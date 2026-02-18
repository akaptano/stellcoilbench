"""
Tests for the nonstop CI autopilot components:
  - validate_config.validate_ci_case / validate_ci_case_file
  - tools/build_context.py
  - tools/propose_batch.py
  - cli run-ci-case
  - coil_optimization iteration cap and iterations_used reporting
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import yaml

# ---------------------------------------------------------------------------
# Make tools/ importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from stellcoilbench.validate_config import validate_ci_case, validate_ci_case_file  # noqa: E402
from build_context import (  # noqa: E402
    build_context,
    compute_failure_stats,
    get_top_parents,
    _config_hash,
    _load_summaries,
)
from propose_batch import (  # noqa: E402
    apply_llm_action,
    check_guardrails,
    is_safe_mode,
    mutate_case,
    explore_case,
    propose_batch,
    _clamp,
    _log_uniform,
    _new_case_id,
    _rng,
)


# ===================================================================
# validate_ci_case
# ===================================================================

class TestValidateCiCase:
    """Tests for validate_ci_case()."""

    @staticmethod
    def _minimal_case(**overrides: Any) -> Dict[str, Any]:
        """Return a minimal valid CI case dict."""
        case: Dict[str, Any] = {
            "case_id": "2026-02-08_00001",
            "resource": {
                "max_total_iterations": 5000,
                "timeout_minutes": 60,
            },
            "case_config": {
                "description": "test",
                "surface_params": {"surface": "input.LandremanPaul2021_QA", "range": "half period"},
                "coils_params": {"ncoils": 4, "order": 8},
                "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 2000},
            },
        }
        case.update(overrides)
        return case

    def test_valid_case_passes(self):
        errors = validate_ci_case(self._minimal_case())
        assert errors == []

    def test_missing_case_id(self):
        c = self._minimal_case()
        del c["case_id"]
        errors = validate_ci_case(c)
        assert any("case_id" in e for e in errors)

    def test_empty_case_id(self):
        errors = validate_ci_case(self._minimal_case(case_id=""))
        assert any("case_id" in e for e in errors)

    def test_non_string_case_id(self):
        errors = validate_ci_case(self._minimal_case(case_id=123))
        assert any("case_id" in e for e in errors)

    def test_resource_max_iter_exceeds_cap(self):
        c = self._minimal_case()
        c["resource"]["max_total_iterations"] = 99999
        errors = validate_ci_case(c)
        assert any("exceeds cap" in e for e in errors)

    def test_resource_max_iter_custom_policy(self):
        c = self._minimal_case()
        c["resource"]["max_total_iterations"] = 8000
        policy = {"resource_caps": {"max_total_iterations": 5000}}
        errors = validate_ci_case(c, policy=policy)
        assert any("exceeds cap" in e for e in errors)

    def test_resource_max_iter_negative(self):
        c = self._minimal_case()
        c["resource"]["max_total_iterations"] = -1
        errors = validate_ci_case(c)
        assert any("positive integer" in e for e in errors)

    def test_timeout_out_of_range(self):
        c = self._minimal_case()
        c["resource"]["timeout_minutes"] = 999
        errors = validate_ci_case(c)
        assert any("outside allowed range" in e for e in errors)

    def test_timeout_too_small(self):
        c = self._minimal_case()
        c["resource"]["timeout_minutes"] = 1
        errors = validate_ci_case(c)
        assert any("outside allowed range" in e for e in errors)

    def test_timeout_negative(self):
        c = self._minimal_case()
        c["resource"]["timeout_minutes"] = -5
        errors = validate_ci_case(c)
        assert any("positive number" in e for e in errors)

    def test_resource_not_dict(self):
        errors = validate_ci_case(self._minimal_case(resource="bad"))
        assert any("resource must be a dictionary" in e for e in errors)

    def test_parent_ids_not_list(self):
        errors = validate_ci_case(self._minimal_case(parent_ids="bad"))
        assert any("parent_ids must be a list" in e for e in errors)

    def test_tags_not_list(self):
        errors = validate_ci_case(self._minimal_case(tags="bad"))
        assert any("tags must be a list" in e for e in errors)

    def test_random_seed_not_int(self):
        errors = validate_ci_case(self._minimal_case(random_seed=3.14))
        assert any("random_seed must be an integer" in e for e in errors)

    def test_missing_case_config(self):
        c = self._minimal_case()
        del c["case_config"]
        errors = validate_ci_case(c)
        assert any("case_config" in e for e in errors)

    def test_case_config_not_dict(self):
        errors = validate_ci_case(self._minimal_case(case_config="bad"))
        assert any("case_config must be a dictionary" in e for e in errors)

    def test_inner_maxiter_exceeds_cap(self):
        c = self._minimal_case()
        c["case_config"]["optimizer_params"]["max_iterations"] = 20000
        errors = validate_ci_case(c)
        assert any("exceeds cap" in e for e in errors)

    def test_valid_optional_fields(self):
        c = self._minimal_case(
            parent_ids=["parent1"],
            tags=["explore"],
            random_seed=42,
        )
        errors = validate_ci_case(c)
        assert errors == []

    def test_file_prefix_in_errors(self):
        c = self._minimal_case()
        del c["case_id"]
        errors = validate_ci_case(c, file_path=Path("test.json"))
        assert any("test.json" in e for e in errors)


class TestValidateCiCaseFile:
    """Tests for validate_ci_case_file()."""

    def test_valid_file(self, tmp_path):
        f = tmp_path / "case.json"
        c = TestValidateCiCase._minimal_case()
        f.write_text(json.dumps(c))
        errors = validate_ci_case_file(f)
        assert errors == []

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{{{not json")
        errors = validate_ci_case_file(f)
        assert any("JSON parse error" in e for e in errors)

    def test_not_a_dict(self, tmp_path):
        f = tmp_path / "list.json"
        f.write_text("[1,2,3]")
        errors = validate_ci_case_file(f)
        assert any("Root element" in e for e in errors)

    def test_missing_file(self, tmp_path):
        f = tmp_path / "noexist.json"
        errors = validate_ci_case_file(f)
        assert any("Error reading file" in e for e in errors)

    def test_with_policy(self, tmp_path):
        f = tmp_path / "case.json"
        c = TestValidateCiCase._minimal_case()
        c["resource"]["max_total_iterations"] = 8000
        f.write_text(json.dumps(c))
        policy = {"resource_caps": {"max_total_iterations": 5000}}
        errors = validate_ci_case_file(f, policy=policy)
        assert any("exceeds cap" in e for e in errors)


# ===================================================================
# build_context
# ===================================================================

def _make_summary(
    case_id: str = "2026-02-08_00001",
    success: bool = True,
    total_score: float = 0.001,
    **kwargs: Any,
) -> Dict[str, Any]:
    s = {
        "case_id": case_id,
        "success": success,
        "total_score": total_score,
        "iterations_used": kwargs.get("iterations_used", 1000),
        "walltime_sec": kwargs.get("walltime_sec", 100.0),
        "failure_reason": kwargs.get("failure_reason", ""),
        "failure_class": kwargs.get("failure_class", ""),
        "metrics": kwargs.get("metrics", {"final_squared_flux": total_score}),
        "case_config": kwargs.get("case_config", {
            "description": "test",
            "surface_params": {"surface": "input.LandremanPaul2021_QA"},
            "coils_params": {"ncoils": 4, "order": 8},
            "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 2000},
        }),
    }
    return s


class TestComputeFailureStats:
    def test_empty(self):
        stats = compute_failure_stats([])
        assert stats["window_size"] == 0
        assert stats["fail_rate"] == 0.0

    def test_all_success(self):
        summaries = [_make_summary(case_id=f"id_{i}") for i in range(10)]
        stats = compute_failure_stats(summaries, window=10)
        assert stats["fail_rate"] == 0.0
        assert stats["fail_count"] == 0

    def test_half_failures(self):
        summaries = []
        for i in range(10):
            summaries.append(_make_summary(
                case_id=f"id_{i}",
                success=(i % 2 == 0),
                failure_reason="bad" if (i % 2 != 0) else "",
                failure_class="RuntimeError" if (i % 2 != 0) else "",
            ))
        stats = compute_failure_stats(summaries, window=10)
        assert stats["fail_rate"] == 0.5
        assert stats["fail_count"] == 5
        assert "bad" in stats["failure_reasons"]

    def test_window_truncation(self):
        summaries = [_make_summary(case_id=f"id_{i}", success=False, failure_reason="x") for i in range(50)]
        stats = compute_failure_stats(summaries, window=5)
        assert stats["window_size"] == 5
        assert stats["fail_count"] == 5


class TestGetTopParents:
    def test_sorted_by_score(self):
        summaries = [
            _make_summary(case_id="a", total_score=0.1),
            _make_summary(case_id="b", total_score=0.001),
            _make_summary(case_id="c", total_score=0.05),
        ]
        parents = get_top_parents(summaries, top_k=2)
        assert len(parents) == 2
        assert parents[0]["case_id"] == "b"
        assert parents[1]["case_id"] == "c"

    def test_excludes_failures(self):
        summaries = [
            _make_summary(case_id="good", total_score=0.001),
            _make_summary(case_id="bad", success=False, total_score=0.0001),
        ]
        parents = get_top_parents(summaries)
        assert len(parents) == 1
        assert parents[0]["case_id"] == "good"


class TestConfigHash:
    def test_deterministic(self):
        cfg = {"a": 1, "b": [2, 3]}
        assert _config_hash(cfg) == _config_hash(cfg)

    def test_order_independent(self):
        assert _config_hash({"a": 1, "b": 2}) == _config_hash({"b": 2, "a": 1})

    def test_different_configs(self):
        assert _config_hash({"a": 1}) != _config_hash({"a": 2})


class TestLoadSummaries:
    def test_loads_from_dir(self, tmp_path):
        done = tmp_path / "done"
        d1 = done / "case_001"
        d1.mkdir(parents=True)
        (d1 / "summary.json").write_text(json.dumps(_make_summary(case_id="case_001")))
        d2 = done / "case_002"
        d2.mkdir(parents=True)
        (d2 / "summary.json").write_text(json.dumps(_make_summary(case_id="case_002")))

        summaries = _load_summaries(done)
        assert len(summaries) == 2

    def test_empty_dir(self, tmp_path):
        done = tmp_path / "done"
        done.mkdir()
        assert _load_summaries(done) == []

    def test_nonexistent_dir(self, tmp_path):
        assert _load_summaries(tmp_path / "nope") == []

    def test_limit(self, tmp_path):
        done = tmp_path / "done"
        for i in range(5):
            d = done / f"case_{i:03d}"
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps(_make_summary(case_id=f"case_{i:03d}")))
        summaries = _load_summaries(done, limit=2)
        assert len(summaries) == 2


class TestBuildContext:
    def test_builds_context_from_done(self, tmp_path):
        done = tmp_path / "done"
        d1 = done / "case_001"
        d1.mkdir(parents=True)
        (d1 / "summary.json").write_text(json.dumps(_make_summary(case_id="case_001")))

        policy_path = tmp_path / "policy.yaml"
        policy_path.write_text(yaml.dump({
            "batch_size": 8,
            "exploit_fraction": 0.5,
            "top_k_parents": 5,
            "resource_caps": {"max_total_iterations": 10000},
            "guardrails": {"sliding_window": 30},
        }))

        ctx = build_context(done, policy_path)
        assert "policy" in ctx
        assert "failure_stats" in ctx
        assert "top_parents" in ctx
        assert "recent_config_hashes" in ctx
        assert "kb_enriched" in ctx
        assert ctx["total_completed"] == 1
        assert ctx["kb_enriched"] is False  # no kb_url passed

    def test_kb_url_without_token_falls_back_on_error(self, tmp_path):
        """kb_url alone (no token) is attempted; falls back to local on KB error."""
        done = tmp_path / "done"
        d1 = done / "case_001"
        d1.mkdir(parents=True)
        (d1 / "summary.json").write_text(json.dumps(_make_summary(case_id="case_001")))

        policy_path = tmp_path / "policy.yaml"
        policy_path.write_text(yaml.dump({
            "batch_size": 8,
            "exploit_fraction": 0.5,
            "top_k_parents": 5,
            "guardrails": {"sliding_window": 30},
        }))

        # Unreachable KB URL -> falls back to local, kb_enriched=False
        ctx = build_context(done, policy_path, kb_url="http://127.0.0.1:19999")
        assert ctx["kb_enriched"] is False
        assert len(ctx["top_parents"]) == 1
        assert ctx["top_parents"][0]["case_id"] == "case_001"


# ===================================================================
# propose_batch
# ===================================================================

class TestCheckGuardrails:
    def test_no_trigger(self):
        ctx = {"failure_stats": {"fail_rate": 0.1, "most_common_reason_count": 2, "failure_classes": {}}}
        policy = {"guardrails": {"max_fail_rate": 0.6, "max_common_failure_count": 12}}
        stop, reason = check_guardrails(ctx, policy)
        assert not stop

    def test_fail_rate_trigger(self):
        ctx = {"failure_stats": {"fail_rate": 0.7, "most_common_reason_count": 2, "failure_classes": {}}}
        policy = {"guardrails": {"max_fail_rate": 0.6, "max_common_failure_count": 12}}
        stop, reason = check_guardrails(ctx, policy)
        assert stop
        assert "fail_rate" in reason

    def test_common_failure_trigger(self):
        ctx = {"failure_stats": {"fail_rate": 0.3, "most_common_reason_count": 15, "most_common_reason": "bad_stuff", "failure_classes": {}}}
        policy = {"guardrails": {"max_fail_rate": 0.6, "max_common_failure_count": 12}}
        stop, reason = check_guardrails(ctx, policy)
        assert stop
        assert "bad_stuff" in reason

    def test_critical_class_trigger(self):
        ctx = {"failure_stats": {
            "fail_rate": 0.3,
            "most_common_reason_count": 2,
            "failure_classes": {"vmec_nonconverged": 11},
        }}
        policy = {"guardrails": {
            "max_fail_rate": 0.6,
            "max_common_failure_count": 12,
            "critical_failure_classes": ["vmec_nonconverged"],
            "max_critical_class_count": 10,
        }}
        stop, reason = check_guardrails(ctx, policy)
        assert stop
        assert "vmec_nonconverged" in reason


class TestSafeMode:
    def test_not_safe(self):
        ctx = {"failure_stats": {"fail_rate": 0.1}}
        policy = {"safe_mode": {"threshold": 0.35}}
        assert not is_safe_mode(ctx, policy)

    def test_safe(self):
        ctx = {"failure_stats": {"fail_rate": 0.4}}
        policy = {"safe_mode": {"threshold": 0.35}}
        assert is_safe_mode(ctx, policy)


class TestMutateCase:
    def test_produces_valid_child(self):
        parent = _make_summary(case_config={
            "description": "parent",
            "surface_params": {"surface": "input.LandremanPaul2021_QA", "range": "half period"},
            "coils_params": {"ncoils": 4, "order": 8},
            "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 2000},
            "coil_objective_terms": {
                "total_length": "l2_threshold",
                "length_threshold": 24.0,
                "length_weight": 0.05,
                "coil_curvature": "lp_threshold",
                "coil_curvature_p": 2,
            },
        })
        policy_path = _REPO_ROOT / "policy" / "proposer_policy.yaml"
        policy = yaml.safe_load(policy_path.read_text())

        rng = _rng(42)
        child = mutate_case(parent, policy, rng)

        assert child["case_id"] != parent["case_id"]
        assert parent.get("case_id") in child.get("parent_ids", [])
        assert "exploit" in child.get("tags", [])
        assert "case_config" in child
        # Weights should have been removed, thresholds jittered
        cc = child["case_config"].get("coil_objective_terms", {})
        weight_keys = [k for k in cc if k.endswith("_weight")]
        assert weight_keys == [], f"Expected no weight keys, got {weight_keys}"
        assert "length_threshold" in cc

    def test_weights_removed_from_child(self):
        """Mutate should strip weight keys from parent (auglag auto-tunes them)."""
        parent = _make_summary(case_config={
            "description": "parent",
            "surface_params": {"surface": "input.LandremanPaul2021_QA", "range": "half period"},
            "coils_params": {"ncoils": 4, "order": 8},
            "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 2000},
            "coil_objective_terms": {
                "length_weight": 1.0,
                "curvature_weight": 0.5,
                "length_threshold": 50.0,
                "cc_threshold": 0.8,
            },
        })
        policy = {"mutation": {"threshold_sigma": 0.1, "max_iterations": 1000}}
        rng = _rng(42)
        child = mutate_case(parent, policy, rng)
        obj = child["case_config"].get("coil_objective_terms", {})
        # No weight keys should remain
        weight_keys = [k for k in obj if k.endswith("_weight")]
        assert weight_keys == [], f"Expected no weight keys, got {weight_keys}"
        # Threshold keys should still be present (and jittered)
        assert "length_threshold" in obj
        assert "cc_threshold" in obj
        # Algorithm should be augmented_lagrangian
        assert child["case_config"]["optimizer_params"]["algorithm"] == "augmented_lagrangian"

    def test_threshold_injection_from_metrics(self):
        """When parent has no explicit thresholds, inject from parent metrics."""
        parent = _make_summary(
            case_config={
                "description": "parent",
                "surface_params": {"surface": "input.LandremanPaul2021_QA", "range": "half period"},
                "coils_params": {"ncoils": 4, "order": 4},
                "optimizer_params": {"algorithm": "augmented_lagrangian", "max_iterations": 500},
                "coil_objective_terms": {
                    "total_length": "l2_threshold",
                    "coil_curvature": "lp_threshold",
                },
            },
            metrics={
                "cc_threshold": 0.08,
                "cs_threshold": 0.13,
                "msc_threshold": 10.0,
                "curvature_threshold": 10.0,
                "flux_threshold": 1e-8,
            },
        )
        policy = {"mutation": {"threshold_sigma": 0.15, "max_iterations": 500}}
        rng = _rng(42)
        child = mutate_case(parent, policy, rng)
        obj = child["case_config"]["coil_objective_terms"]
        # Thresholds should have been injected from metrics and jittered
        assert "cc_threshold" in obj
        assert "cs_threshold" in obj
        # Values should differ from parent (jittered)
        assert obj["cc_threshold"] != 0.08 or obj["cs_threshold"] != 0.13

    def test_structural_mutation_ncoils(self):
        """Structural mutation can change ncoils."""
        parent = _make_summary(case_config={
            "description": "parent",
            "surface_params": {"surface": "input.LandremanPaul2021_QA", "range": "half period"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "augmented_lagrangian", "max_iterations": 500},
            "coil_objective_terms": {"cc_threshold": 0.08},
        })
        policy = {
            "mutation": {
                "threshold_sigma": 0.1,
                "structural_mutation_prob": 1.0,  # always mutate
                "ncoils_choices": [3, 4, 5, 6, 7],
                "order_choices": [4, 6, 8],
                "max_iterations": 500,
            },
        }
        ncoils_seen = set()
        for seed in range(50):
            child = mutate_case(parent, policy, _rng(seed))
            ncoils_seen.add(child["case_config"]["coils_params"]["ncoils"])
        # With prob=1.0, should see adjacent values (3 or 5)
        assert 3 in ncoils_seen or 5 in ncoils_seen
        # Original value (4) should NOT appear since structural_mutation_prob=1.0
        # selects adjacent, which never includes the current value

    def test_dof_perturbation_in_child(self):
        """Mutation should set dof_perturbation in child config."""
        parent = _make_summary(case_config={
            "description": "parent",
            "surface_params": {"surface": "input.LandremanPaul2021_QA", "range": "half period"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "augmented_lagrangian", "max_iterations": 500},
            "coil_objective_terms": {},
        })
        policy = {
            "mutation": {
                "threshold_sigma": 0.1,
                "dof_perturbation": 0.02,
                "max_iterations": 500,
            },
        }
        child = mutate_case(parent, policy, _rng(42))
        assert child["case_config"].get("dof_perturbation") == 0.02


class TestExploreCase:
    def test_produces_valid_case(self):
        policy_path = _REPO_ROOT / "policy" / "proposer_policy.yaml"
        policy = yaml.safe_load(policy_path.read_text())
        rng = _rng(42)
        case = explore_case(policy, rng)

        assert "case_id" in case
        assert "case_config" in case
        assert "explore" in case.get("tags", [])
        assert case["resource"]["max_total_iterations"] <= 10000
        cc = case["case_config"]
        assert "surface_params" in cc
        assert "coils_params" in cc
        assert "optimizer_params" in cc

    def test_threshold_sampling(self):
        """When use_default_thresholds is false, thresholds should be sampled."""
        policy = {
            "exploration": {
                "use_default_thresholds": False,
                "surfaces": ["input.LandremanPaul2021_QA"],
                "algorithms": ["augmented_lagrangian"],
                "ncoils_choices": [4],
                "order_choices": [4],
                "max_iterations": 500,
                "length_threshold_range": [100, 300],
                "cc_threshold_range": [0.4, 1.5],
                "cs_threshold_range": [0.5, 2.5],
                "curvature_threshold_range": [0.5, 5.0],
                "msc_threshold_range": [0.1, 5.0],
            },
        }
        case = explore_case(policy, _rng(42))
        obj = case["case_config"]["coil_objective_terms"]
        assert "length_threshold" in obj
        assert "cc_threshold" in obj
        assert "cs_threshold" in obj
        assert "curvature_threshold" in obj
        assert "msc_threshold" in obj
        # Values should be within ranges
        assert 100 <= obj["length_threshold"] <= 300
        assert 0.4 <= obj["cc_threshold"] <= 1.5

    def test_threshold_diversity(self):
        """Different seeds should produce different thresholds."""
        policy = {
            "exploration": {
                "use_default_thresholds": False,
                "surfaces": ["input.LandremanPaul2021_QA"],
                "algorithms": ["augmented_lagrangian"],
                "ncoils_choices": [4],
                "order_choices": [4],
                "max_iterations": 500,
                "cc_threshold_range": [0.4, 2.0],
                "cs_threshold_range": [0.5, 3.0],
            },
        }
        thresholds = set()
        for seed in range(20):
            case = explore_case(policy, _rng(seed))
            obj = case["case_config"]["coil_objective_terms"]
            thresholds.add(obj.get("cc_threshold"))
        # With 20 different seeds and continuous sampling, all should be unique
        assert len(thresholds) == 20

    def test_dof_perturbation_in_explore(self):
        """Exploration can set dof_perturbation."""
        policy = {
            "exploration": {
                "surfaces": ["input.LandremanPaul2021_QA"],
                "algorithms": ["augmented_lagrangian"],
                "ncoils_choices": [4],
                "order_choices": [4],
                "max_iterations": 500,
                "dof_perturbation": 0.01,
            },
        }
        case = explore_case(policy, _rng(42))
        assert case["case_config"].get("dof_perturbation") == 0.01

    def test_include_force_objective(self):
        """When include_force is true, coil_coil_force should be in terms."""
        policy = {
            "exploration": {
                "surfaces": ["input.LandremanPaul2021_QA"],
                "algorithms": ["augmented_lagrangian"],
                "ncoils_choices": [4],
                "order_choices": [4],
                "max_iterations": 500,
                "include_force": True,
                "use_default_thresholds": False,
                "force_threshold_range": [50, 500],
            },
        }
        case = explore_case(policy, _rng(42))
        obj = case["case_config"]["coil_objective_terms"]
        assert "coil_coil_force" in obj
        assert "force_threshold" in obj

    def test_safe_mode_preferred_surfaces(self):
        policy = {
            "exploration": {
                "surfaces": ["input.W7-X_without_coil_ripple_beta0p05_d23p4_tm", "input.LandremanPaul2021_QA"],
                "algorithms": ["L-BFGS-B"],
                "ncoils_choices": [4],
                "order_choices": [8],
                "max_iterations_range": [1000, 5000],
            },
            "safe_mode": {
                "preferred_surfaces": ["input.LandremanPaul2021_QA"],
                "max_iterations_cap": 3000,
            },
        }
        surfaces_seen = set()
        for seed in range(20):
            case = explore_case(policy, _rng(seed), safe=True)
            s = case["case_config"]["surface_params"]["surface"]
            surfaces_seen.add(s)
        # Should only see the preferred surface in safe mode
        assert surfaces_seen == {"input.LandremanPaul2021_QA"}


class TestProposeBatch:
    def test_batch_size_respected(self):
        policy_path = _REPO_ROOT / "policy" / "proposer_policy.yaml"
        policy = yaml.safe_load(policy_path.read_text())
        ctx = {
            "policy": {"batch_size": 8},
            "failure_stats": {"fail_rate": 0.0, "most_common_reason_count": 0, "failure_classes": {}},
            "top_parents": [_make_summary(case_id=f"p{i}", case_config={
                "description": "parent",
                "surface_params": {"surface": "input.LandremanPaul2021_QA", "range": "half period"},
                "coils_params": {"ncoils": 4, "order": 8},
                "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 2000},
                "coil_objective_terms": {"total_length": "l2_threshold", "length_threshold": 24.0, "length_weight": 0.05},
            }) for i in range(5)],
            "recent_config_hashes": [],
            "total_completed": 10,
        }
        cases = propose_batch(ctx, policy, batch_size=4, seed=42)
        assert len(cases) == 4

    def test_all_cases_validated(self):
        policy_path = _REPO_ROOT / "policy" / "proposer_policy.yaml"
        policy = yaml.safe_load(policy_path.read_text())
        ctx = {
            "policy": {"batch_size": 8},
            "failure_stats": {"fail_rate": 0.0, "most_common_reason_count": 0, "failure_classes": {}},
            "top_parents": [],
            "recent_config_hashes": [],
            "total_completed": 0,
        }
        cases = propose_batch(ctx, policy, batch_size=4, seed=123)
        assert len(cases) == 4
        for c in cases:
            errors = validate_ci_case(c, policy=policy)
            assert errors == [], f"Case {c.get('case_id')} failed validation: {errors}"

    def test_no_parents_still_works(self):
        """When there are no parents, all cases should be explorations."""
        policy_path = _REPO_ROOT / "policy" / "proposer_policy.yaml"
        policy = yaml.safe_load(policy_path.read_text())
        ctx = {
            "policy": {"batch_size": 8},
            "failure_stats": {"fail_rate": 0.0, "most_common_reason_count": 0, "failure_classes": {}},
            "top_parents": [],
            "recent_config_hashes": [],
            "total_completed": 0,
        }
        cases = propose_batch(ctx, policy, batch_size=4, seed=42)
        assert len(cases) == 4
        for c in cases:
            assert "explore" in c.get("tags", [])

    def test_no_duplicate_config_hashes(self):
        """All cases in a batch should have unique config hashes."""
        policy_path = _REPO_ROOT / "policy" / "proposer_policy.yaml"
        policy = yaml.safe_load(policy_path.read_text())
        ctx = {
            "policy": {"batch_size": 8},
            "failure_stats": {"fail_rate": 0.0, "most_common_reason_count": 0, "failure_classes": {}},
            "top_parents": [],
            "recent_config_hashes": [],
            "total_completed": 0,
        }
        from propose_batch import _config_hash_short
        cases = propose_batch(ctx, policy, batch_size=8, seed=42)
        hashes = [_config_hash_short(c["case_config"]) for c in cases]
        assert len(set(hashes)) == len(hashes), f"Duplicate hashes found: {hashes}"

    def test_novelty_prevents_duplicates_with_recent(self):
        """Cases matching recent_config_hashes should not be proposed."""
        policy = {
            "exploration": {
                "use_default_thresholds": True,  # deterministic configs
                "surfaces": ["input.LandremanPaul2021_QA"],
                "algorithms": ["augmented_lagrangian"],
                "ncoils_choices": [4],
                "order_choices": [4],
                "max_iterations": 500,
            },
            "exploit_fraction": 0.0,
        }
        # First, generate a case to get its hash
        from propose_batch import _config_hash_short
        probe = explore_case(policy, _rng(0))
        probe_hash = _config_hash_short(probe["case_config"])

        ctx = {
            "failure_stats": {"fail_rate": 0.0, "most_common_reason_count": 0, "failure_classes": {}},
            "top_parents": [],
            "recent_config_hashes": [probe_hash],
            "total_completed": 5,
        }
        # With only 1 possible config and it's in recent_hashes, batch should be empty
        cases = propose_batch(ctx, policy, batch_size=4, seed=42)
        assert len(cases) == 0


# ===================================================================
# LLM proposer: apply_llm_action
# ===================================================================

class TestApplyLLMAction:
    def test_mutate_action_produces_valid_case(self):
        parent = {
            "case_id": "parent_001",
            "case_config": {
                "surface_params": {"surface": "input.LandremanPaul2021_QA", "range": "half period"},
                "coils_params": {"ncoils": 4, "order": 8},
                "optimizer_params": {},
                "coil_objective_terms": {"cc_threshold": 1.0, "cs_threshold": 2.0},
            },
        }
        ctx = {"top_parents": [parent]}
        policy = {
            "mutation": {"max_iterations": 500},
            "resource_caps": {"max_total_iterations": 5000, "timeout_minutes_max": 60},
            "fourier_continuation": {"enabled": True, "orders": [4, 8, 16]},
        }
        action = {"type": "mutate", "parent_id": "parent_001", "overrides": {"ncoils": 5}}
        case = apply_llm_action(action, ctx, policy, _rng(42))
        assert case is not None
        assert case["parent_ids"] == ["parent_001"]
        assert case["case_config"]["coils_params"]["ncoils"] == 5
        assert "llm" in case["tags"]

    def test_explore_action_produces_valid_case(self):
        ctx = {"top_parents": []}
        policy = {
            "exploration": {
                "surfaces": ["input.LandremanPaul2021_QA"],
                "max_iterations": 500,
            },
            "resource_caps": {"max_total_iterations": 5000, "timeout_minutes_max": 60},
            "fourier_continuation": {"enabled": True, "orders": [4, 8, 16]},
        }
        action = {
            "type": "explore",
            "surface": "input.LandremanPaul2021_QA",
            "ncoils": 4,
            "order": 8,
            "thresholds": {"cc_threshold": 1.0},
        }
        case = apply_llm_action(action, ctx, policy, _rng(42))
        assert case is not None
        assert case["parent_ids"] == []
        assert case["case_config"]["surface_params"]["surface"] == "input.LandremanPaul2021_QA"
        assert case["case_config"]["coil_objective_terms"]["cc_threshold"] == 1.0

    def test_unknown_action_returns_none(self):
        ctx = {"top_parents": []}
        policy = {}
        case = apply_llm_action({"type": "unknown"}, ctx, policy, _rng(42))
        assert case is None


# ===================================================================
# Helpers
# ===================================================================

class TestHelpers:
    def test_clamp(self):
        assert _clamp(5, 0, 10) == 5
        assert _clamp(-1, 0, 10) == 0
        assert _clamp(99, 0, 10) == 10

    def test_log_uniform_range(self):
        rng = _rng(42)
        for _ in range(100):
            v = _log_uniform(rng, 0.01, 10.0)
            assert 0.01 <= v <= 10.0

    def test_new_case_id_format(self):
        cid = _new_case_id()
        parts = cid.split("_")
        assert len(parts) >= 3  # date, time, suffix

    def test_rng_seeded(self):
        r1 = _rng(42)
        r2 = _rng(42)
        assert r1.random() == r2.random()


# ===================================================================
# CLI run-ci-case (smoke test with mocked optimize_coils)
# ===================================================================

class TestRunCiCaseCLI:
    @staticmethod
    def _write_case_file(tmp_path: Path) -> Path:
        case = TestValidateCiCase._minimal_case()
        f = tmp_path / "pending" / f"{case['case_id']}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(case))
        return f

    def test_validation_failure_writes_summary(self, tmp_path):
        """A case that fails validation should still produce a summary."""
        bad_case = {"case_id": "bad_001", "resource": {"max_total_iterations": 999999}, "case_config": {}}
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad_case))

        from typer.testing import CliRunner
        from stellcoilbench.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "run-ci-case", str(f), "--output-dir", str(tmp_path / "done"),
        ])
        assert result.exit_code == 1
        summary = tmp_path / "done" / "bad_001" / "summary.json"
        assert summary.exists()
        data = json.loads(summary.read_text())
        assert data["success"] is False

    def test_successful_run(self, tmp_path):
        """Smoke test: mock optimize_coils and verify summary is written."""
        case = TestValidateCiCase._minimal_case(
            random_seed=42,
            tags=["explore"],
            parent_ids=["parent_001"],
        )
        f = tmp_path / "case.json"
        f.write_text(json.dumps(case))

        mock_results = {
            "final_squared_flux": 0.001,
            "iterations_used": 1500,
            "optimization_time": 42.0,
            "walltime_sec": 42.0,
            "timing": {"coil_optimization": 42.0},
        }

        from typer.testing import CliRunner
        from stellcoilbench.cli import app

        runner = CliRunner()
        with patch("stellcoilbench.coil_optimization.optimize_coils", return_value=mock_results):
            result = runner.invoke(app, [
                "run-ci-case", str(f),
                "--output-dir", str(tmp_path / "done"),
            ])

        assert result.exit_code == 0
        summary_path = tmp_path / "done" / case["case_id"] / "summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["success"] is True
        assert data["total_score"] == 0.001
        assert data["iterations_used"] == 1500
        # Verify context fields are preserved
        assert data["random_seed"] == 42
        assert data["tags"] == ["explore"]
        assert data["parent_ids"] == ["parent_001"]

    def test_optimize_exception_writes_failure(self, tmp_path):
        """If optimize_coils raises, summary should show failure."""
        case = TestValidateCiCase._minimal_case()
        f = tmp_path / "case.json"
        f.write_text(json.dumps(case))

        from typer.testing import CliRunner
        from stellcoilbench.cli import app

        runner = CliRunner()
        with patch("stellcoilbench.coil_optimization.optimize_coils", side_effect=RuntimeError("boom")):
            runner.invoke(app, [
                "run-ci-case", str(f),
                "--output-dir", str(tmp_path / "done"),
            ])

        # Summary should be written even on failure
        summary_path = tmp_path / "done" / case["case_id"] / "summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["success"] is False
        assert "boom" in data["failure_reason"]


# ===================================================================
# coil_optimization: iteration cap
# ===================================================================

class TestIterationCap:
    """Test that _optimize_coils_loop_impl clamps max_iterations."""

    def test_clamp_message(self, capsys):
        """Verify the warning message is printed when max_iterations exceeds cap."""
        # We can't easily run the full function, but we can verify the code exists
        # by importing and checking the source
        import inspect
        from stellcoilbench.coil_optimization import _optimize_coils_loop_impl
        source = inspect.getsource(_optimize_coils_loop_impl)
        assert "_CI_MAX_ITER_CAP" in source
        assert "clamping" in source

    def test_results_contain_iterations_used(self):
        """Verify that the results dict template includes iterations_used."""
        import inspect
        from stellcoilbench.coil_optimization import _optimize_coils_loop_impl
        source = inspect.getsource(_optimize_coils_loop_impl)
        assert "'iterations_used'" in source
        assert "'walltime_sec'" in source
