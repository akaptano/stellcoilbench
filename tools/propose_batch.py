#!/usr/bin/env python3
"""
Non-LLM batch proposer for the nonstop CI autopilot.

Reads recent results, applies guardrails, then generates a batch of cases
using mutation (exploit) and exploration operators.  Every proposed case is
validated before being written to ``cases/pending/``.

Usage::

    python tools/propose_batch.py --batch-size 8

This script is designed to be called from a scheduled CI workflow
(``propose_cases.yml``).  It will exit with code 0 even when it decides
*not* to propose (e.g. guardrail triggered or pending dir not empty).
"""
from __future__ import annotations

import argparse
import copy
import os
import hashlib
import json
import math
import random as _random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# --- sibling imports (when run as script, add parent to path) ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from build_context import build_context  # noqa: E402
from stellcoilbench.validate_config import validate_ci_case  # noqa: E402


def _rng(seed: int | None = None) -> _random.Random:
    """Return a seeded Random instance."""
    return _random.Random(seed)


def _log_uniform(rng: _random.Random, lo: float, hi: float) -> float:
    """Sample from a log-uniform distribution in [lo, hi]."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _new_case_id() -> str:
    """Generate a unique case_id string with timestamp + random suffix."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = _random.randint(10000, 99999)
    return f"{ts}_{suffix}"


def _config_hash_short(cfg: Dict[str, Any]) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Guardrail checks
# ---------------------------------------------------------------------------


def check_guardrails(
    ctx: Dict[str, Any],
    policy: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check guardrails.  Returns (should_stop, reason)."""
    gr = policy.get("guardrails", {})
    stats = ctx.get("failure_stats", {})

    fail_rate = stats.get("fail_rate", 0.0)
    max_fail_rate = gr.get("max_fail_rate", 0.6)
    if fail_rate > max_fail_rate:
        return True, f"fail_rate {fail_rate:.2f} > {max_fail_rate}"

    mcrc = stats.get("most_common_reason_count", 0)
    max_mcrc = gr.get("max_common_failure_count", 12)
    if mcrc > max_mcrc:
        reason = stats.get("most_common_reason", "?")
        return True, f"failure reason '{reason}' repeated {mcrc} > {max_mcrc} times"

    # Check critical failure classes
    critical_classes = set(gr.get("critical_failure_classes", []))
    max_crit = gr.get("max_critical_class_count", 10)
    for cls, cnt in stats.get("failure_classes", {}).items():
        if cls in critical_classes and cnt >= max_crit:
            return True, f"critical failure class '{cls}' repeated {cnt} >= {max_crit}"

    return False, ""


def is_safe_mode(ctx: Dict[str, Any], policy: Dict[str, Any]) -> bool:
    """Return True if the proposer should operate in safe mode."""
    sm = policy.get("safe_mode", {})
    threshold = sm.get("threshold", 0.35)
    fail_rate = ctx.get("failure_stats", {}).get("fail_rate", 0.0)
    return fail_rate > threshold


# ---------------------------------------------------------------------------
# Mutation (exploit) operator
# ---------------------------------------------------------------------------


def mutate_case(
    parent: Dict[str, Any],
    policy: Dict[str, Any],
    rng: _random.Random,
    *,
    safe: bool = False,
) -> Dict[str, Any]:
    """Create a mutated child case from a parent's ``case_config``.

    The child gets a new ``case_id``, ``random_seed``, and jittered thresholds.
    Weights are NOT mutated -- augmented_lagrangian auto-tunes them.
    Any leftover weight keys from the parent are removed.
    The parent ``case_id`` is stored in ``parent_ids``.

    If the parent has no explicit thresholds (because auto-scaling was used),
    thresholds are injected from the parent's metrics so they can be jittered.
    Structural mutations (ncoils, order) are applied with configurable probability.
    """
    parent_cfg = parent.get("case_config", {})
    child_cfg = copy.deepcopy(parent_cfg)

    mut = policy.get("mutation", {})

    t_sigma = float(mut.get("threshold_sigma", 0.10))

    # Ensure algorithm is augmented_lagrangian
    opt = child_cfg.get("optimizer_params", {})
    opt["algorithm"] = "augmented_lagrangian"

    # Remove any weight keys from parent (augmented_lagrangian handles weights)
    obj = child_cfg.get("coil_objective_terms", {})
    if obj is None:
        obj = {}
    weight_keys = [k for k in obj if k.endswith("_weight")]
    for wk in weight_keys:
        del obj[wk]

    # If the parent has no explicit thresholds (auto-scaling was used),
    # inject thresholds from the parent's metrics so we can jitter them.
    threshold_keys = [k for k in obj if k.endswith("_threshold")]
    if not threshold_keys:
        parent_metrics = parent.get("metrics", {})
        for tname in ["cc_threshold", "cs_threshold", "msc_threshold",
                       "curvature_threshold", "flux_threshold",
                       "force_threshold", "torque_threshold"]:
            if tname in parent_metrics:
                val = parent_metrics[tname]
                if isinstance(val, (int, float)) and val > 0:
                    obj[tname] = val
        threshold_keys = [k for k in obj if k.endswith("_threshold")]

    # Jitter thresholds (only numeric values, skip string term-type values)
    for tk in threshold_keys:
        old = obj[tk]
        if isinstance(old, (int, float)) and not isinstance(old, bool) and old > 0:
            new_val = old * math.exp(rng.gauss(0, t_sigma))
            obj[tk] = round(max(1e-6, new_val), 6)

    child_cfg["coil_objective_terms"] = obj

    # --- Structural mutations (ncoils, order) ---
    struct_prob = float(mut.get("structural_mutation_prob", 0.2))
    coils_params = child_cfg.get("coils_params", {})

    # Mutate ncoils with probability struct_prob
    if rng.random() < struct_prob:
        ncoils_choices = mut.get("ncoils_choices", [3, 4, 5, 6, 7])
        current_ncoils = coils_params.get("ncoils", 4)
        # Pick an adjacent value (+-1) if possible, otherwise random
        adjacent = [n for n in ncoils_choices if abs(n - current_ncoils) == 1]
        if adjacent:
            coils_params["ncoils"] = rng.choice(adjacent)
        elif len(ncoils_choices) > 1:
            others = [n for n in ncoils_choices if n != current_ncoils]
            coils_params["ncoils"] = rng.choice(others)

    # Mutate order with probability struct_prob
    if rng.random() < struct_prob:
        order_choices = mut.get("order_choices", [4, 6, 8])
        current_order = coils_params.get("order", 4)
        adjacent = [o for o in order_choices if abs(o - current_order) <= 2 and o != current_order]
        if adjacent:
            coils_params["order"] = rng.choice(adjacent)
        elif len(order_choices) > 1:
            others = [o for o in order_choices if o != current_order]
            coils_params["order"] = rng.choice(others)

    child_cfg["coils_params"] = coils_params

    # Update description to reflect mutations
    surface = child_cfg.get("surface_params", {}).get("surface", "unknown")
    ncoils = coils_params.get("ncoils", 4)
    order = coils_params.get("order", 4)
    child_cfg["description"] = f"Mutation: {surface} ncoils={ncoils} order={order}"

    # Apply Fourier continuation from policy if present (overwrite parent)
    fc = policy.get("fourier_continuation", {})
    if fc and fc.get("enabled") and fc.get("orders"):
        child_cfg["fourier_continuation"] = {
            "enabled": True,
            "orders": list(fc["orders"]),
        }

    # Fix max_iterations to policy default (not scanned)
    mut_policy = policy.get("mutation", {})
    opt["max_iterations"] = mut_policy.get("max_iterations", 1000)
    opt["verbose"] = True
    # Don't set max_iter_subopt -- let the code default to max_iterations // 50
    opt.pop("max_iter_subopt", None)
    child_cfg["optimizer_params"] = opt

    # Enable DOF perturbation to break determinism
    dof_perturbation = float(mut.get("dof_perturbation", 0.01))
    if dof_perturbation > 0:
        child_cfg["dof_perturbation"] = dof_perturbation

    # New seed always
    new_seed = rng.randint(0, 2**31 - 1)

    # Resource block — respect policy caps
    caps = policy.get("resource_caps", {})
    max_iter = opt.get("max_iterations", 2000)
    resource = {
        "max_total_iterations": min(max_iter, caps.get("max_total_iterations", 10000)),
        "timeout_minutes": caps.get("timeout_minutes_max", 60),
    }

    return {
        "case_id": _new_case_id(),
        "parent_ids": [parent.get("case_id", "unknown")],
        "tags": ["exploit"],
        "resource": resource,
        "case_config": child_cfg,
        "random_seed": new_seed,
    }


# ---------------------------------------------------------------------------
# Exploration operator
# ---------------------------------------------------------------------------


def explore_case(
    policy: Dict[str, Any],
    rng: _random.Random,
    *,
    safe: bool = False,
) -> Dict[str, Any]:
    """Generate a random exploration case from the policy parameter ranges.

    Uses augmented_lagrangian by default (auto-tunes weights).
    Thresholds are sampled from log-uniform ranges to create diversity —
    different threshold combinations push the optimizer to different local minima.
    """
    expl = policy.get("exploration", {})
    sm = policy.get("safe_mode", {})

    # Pick surface
    if safe:
        surfaces = sm.get("preferred_surfaces", expl.get("surfaces", ["input.LandremanPaul2021_QA"]))
    else:
        surfaces = expl.get("surfaces", ["input.LandremanPaul2021_QA"])
    surface = rng.choice(surfaces)

    # Pick algorithm (default: augmented_lagrangian only)
    algorithms = expl.get("algorithms", ["augmented_lagrangian"])
    algorithm = rng.choice(algorithms)

    # Pick coils/order
    ncoils = rng.choice(expl.get("ncoils_choices", [4]))
    order = rng.choice(expl.get("order_choices", [8]))

    # Max iterations (fixed, not scanned)
    max_iterations = expl.get("max_iterations", 1000)

    # Build coil_objective_terms -- NO weights (augmented_lagrangian handles them)
    # Only specify which objective terms to use and their types.
    coil_objective_terms: Dict[str, Any] = {
        "total_length": "l2_threshold",
        "coil_curvature": "lp_threshold",
        "coil_curvature_p": 2,
        "coil_mean_squared_curvature": "l2_threshold",
        "coil_arclength_variation": "l2_threshold",
        "linking_number": "",
    }

    # Optionally include force objective
    include_force = expl.get("include_force", False)
    if include_force:
        coil_objective_terms["coil_coil_force"] = "lp_threshold"

    # Sample thresholds from log-uniform ranges to create solution diversity.
    # When use_default_thresholds is true, thresholds are omitted and the code
    # auto-scales — but this produces identical results for the same (ncoils, order).
    use_defaults = expl.get("use_default_thresholds", True)
    if not use_defaults:
        coil_objective_terms["length_threshold"] = round(
            _log_uniform(rng, *expl.get("length_threshold_range", [100, 300])), 2)
        coil_objective_terms["cc_threshold"] = round(
            _log_uniform(rng, *expl.get("cc_threshold_range", [0.4, 1.5])), 3)
        coil_objective_terms["cs_threshold"] = round(
            _log_uniform(rng, *expl.get("cs_threshold_range", [0.5, 2.5])), 3)
        coil_objective_terms["curvature_threshold"] = round(
            _log_uniform(rng, *expl.get("curvature_threshold_range", [0.5, 5.0])), 3)
        coil_objective_terms["msc_threshold"] = round(
            _log_uniform(rng, *expl.get("msc_threshold_range", [0.1, 5.0])), 3)
        if include_force:
            coil_objective_terms["force_threshold"] = round(
                _log_uniform(rng, *expl.get("force_threshold_range", [50, 500])), 1)

    case_config: Dict[str, Any] = {
        "description": f"Exploration case: {surface} ncoils={ncoils} order={order}",
        "surface_params": {
            "surface": surface,
            "range": "half period",
        },
        "coils_params": {
            "ncoils": ncoils,
            "order": order,
        },
        "optimizer_params": {
            "algorithm": algorithm,
            "max_iterations": max_iterations,
            "verbose": True,
        },
        "coil_objective_terms": coil_objective_terms,
    }

    # Enable DOF perturbation to break determinism
    dof_perturbation = float(expl.get("dof_perturbation", 0.0))
    if dof_perturbation > 0:
        case_config["dof_perturbation"] = dof_perturbation

    # Apply Fourier continuation from policy if present
    fc = policy.get("fourier_continuation", {})
    if fc and fc.get("enabled") and fc.get("orders"):
        case_config["fourier_continuation"] = {
            "enabled": True,
            "orders": list(fc["orders"]),
        }

    new_seed = rng.randint(0, 2**31 - 1)
    caps = policy.get("resource_caps", {})
    resource = {
        "max_total_iterations": min(max_iterations, caps.get("max_total_iterations", 10000)),
        "timeout_minutes": caps.get("timeout_minutes_max", 60),
    }

    return {
        "case_id": _new_case_id(),
        "parent_ids": [],
        "tags": ["explore"],
        "resource": resource,
        "case_config": case_config,
        "random_seed": new_seed,
    }


# ---------------------------------------------------------------------------
# Batch composition
# ---------------------------------------------------------------------------


def propose_batch(
    ctx: Dict[str, Any],
    policy: Dict[str, Any],
    batch_size: int = 8,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """Propose a batch of cases using mutation + exploration.

    Parameters
    ----------
    ctx : dict
        Context payload from :func:`build_context.build_context`.
    policy : dict
        Full proposer policy.
    batch_size : int
        Number of cases to propose.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        List of validated CI case dicts ready to be written to
        ``cases/pending/``.
    """
    rng = _rng(seed)
    safe = is_safe_mode(ctx, policy)

    exploit_frac = policy.get("exploit_fraction", 0.5)
    exploit_count = int(math.floor(exploit_frac * batch_size))
    explore_count = batch_size - exploit_count

    parents = ctx.get("top_parents", [])
    recent_hashes = set(ctx.get("recent_config_hashes", []))

    cases: List[Dict[str, Any]] = []
    seen_hashes: set = set()

    # --- exploit (mutation) ---
    attempts = 0
    while len(cases) < exploit_count and attempts < exploit_count * 5:
        attempts += 1
        if not parents:
            break
        parent = rng.choice(parents)
        child = mutate_case(parent, policy, rng, safe=safe)

        # Novelty check
        h = _config_hash_short(child.get("case_config", {}))
        if h in recent_hashes or h in seen_hashes:
            continue

        # Validate
        errors = validate_ci_case(child, policy=policy)
        if errors:
            continue

        seen_hashes.add(h)
        cases.append(child)

    # --- explore ---
    attempts = 0
    while len(cases) < exploit_count + explore_count and attempts < explore_count * 5:
        attempts += 1
        child = explore_case(policy, rng, safe=safe)

        h = _config_hash_short(child.get("case_config", {}))
        if h in recent_hashes or h in seen_hashes:
            continue

        errors = validate_ci_case(child, policy=policy)
        if errors:
            continue

        seen_hashes.add(h)
        cases.append(child)

    # If we still don't have enough, fill with more explore (with novelty check)
    extra_attempts = 0
    while len(cases) < batch_size and extra_attempts < batch_size * 5:
        extra_attempts += 1
        child = explore_case(policy, rng, safe=safe)

        h = _config_hash_short(child.get("case_config", {}))
        if h in recent_hashes or h in seen_hashes:
            continue

        errors = validate_ci_case(child, policy=policy)
        if not errors:
            seen_hashes.add(h)
            cases.append(child)

    return cases[:batch_size]


# ---------------------------------------------------------------------------
# LLM proposer: apply mutation actions from KB /propose
# ---------------------------------------------------------------------------


def apply_llm_action(
    action: Dict[str, Any],
    ctx: Dict[str, Any],
    policy: Dict[str, Any],
    rng: _random.Random,
) -> Dict[str, Any] | None:
    """Convert one LLM mutation/exploration action to a CI case dict.

    Supports two action types:
    - "mutate": Clone a parent from top_parents and apply overrides (surface,
      ncoils, order, thresholds). Parent must exist in context.
    - "explore": Create a new exploration case with surface, ncoils, order,
      and optional thresholds. Surface must be in policy's allowed list.

    Parameters
    ----------
    action : dict
        LLM output: {"type": "mutate"|"explore", "parent_id": "...", "overrides": {...}}
        or {"type": "explore", "surface": "...", "ncoils": 4, "order": 8, ...}.
    ctx : dict
        Build context with top_parents.
    policy : dict
        Proposer policy for resource caps, fourier_continuation, etc.
    rng : Random
        Seeded random for case_id and random_seed.

    Returns
    -------
    dict | None
        Valid CI case dict, or None if action is invalid or parent not found.
    """
    action_type = action.get("type", "")
    if action_type == "mutate":
        parent_id = action.get("parent_id", "")
        overrides = action.get("overrides", {})
        parents = {p.get("case_id"): p for p in ctx.get("top_parents", [])}
        parent = parents.get(parent_id)
        if not parent:
            return None
        child_cfg = copy.deepcopy(parent.get("case_config", {}))
        # Apply overrides
        if "surface" in overrides:
            sp = child_cfg.setdefault("surface_params", {})
            sp["surface"] = overrides["surface"]
        if "ncoils" in overrides:
            cp = child_cfg.setdefault("coils_params", {})
            cp["ncoils"] = int(overrides["ncoils"])
        if "order" in overrides:
            cp = child_cfg.setdefault("coils_params", {})
            cp["order"] = int(overrides["order"])
        obj = child_cfg.get("coil_objective_terms") or {}
        for k in ["cc_threshold", "cs_threshold", "curvature_threshold", "msc_threshold",
                   "length_threshold", "flux_threshold", "force_threshold", "torque_threshold"]:
            if k in overrides and isinstance(overrides[k], (int, float)):
                obj[k] = overrides[k]
        child_cfg["coil_objective_terms"] = obj
        child_cfg["description"] = f"LLM mutate: {parent_id}"
        opt = child_cfg.get("optimizer_params", {})
        opt["algorithm"] = "augmented_lagrangian"
        opt["verbose"] = True
        mut = policy.get("mutation", {})
        opt["max_iterations"] = mut.get("max_iterations", 500)
        child_cfg["optimizer_params"] = opt
        fc = policy.get("fourier_continuation", {})
        if fc and fc.get("enabled") and fc.get("orders"):
            child_cfg["fourier_continuation"] = {"enabled": True, "orders": list(fc["orders"])}
        caps = policy.get("resource_caps", {})
        case = {
            "case_id": _new_case_id(),
            "parent_ids": [parent_id],
            "tags": ["exploit", "llm"],
            "resource": {
                "max_total_iterations": min(opt.get("max_iterations", 500), caps.get("max_total_iterations", 10000)),
                "timeout_minutes": caps.get("timeout_minutes_max", 60),
            },
            "case_config": child_cfg,
            "random_seed": rng.randint(0, 2**31 - 1),
        }
        return case

    if action_type == "explore":
        surface = action.get("surface")
        ncoils = action.get("ncoils", 4)
        order = action.get("order", 8)
        thresholds = action.get("thresholds", {})
        expl = policy.get("exploration", {})
        surfaces = expl.get("surfaces", ["input.LandremanPaul2021_QA"])
        if surface not in surfaces:
            surface = surfaces[0] if surfaces else "input.LandremanPaul2021_QA"
        coil_objective_terms: Dict[str, Any] = {
            "total_length": "l2_threshold",
            "coil_curvature": "lp_threshold",
            "coil_curvature_p": 2,
            "coil_mean_squared_curvature": "l2_threshold",
            "coil_arclength_variation": "l2_threshold",
            "linking_number": "",
        }
        for k, v in thresholds.items():
            if isinstance(v, (int, float)):
                coil_objective_terms[k] = v
        max_iterations = expl.get("max_iterations", 500)
        case_config: Dict[str, Any] = {
            "description": f"LLM explore: {surface} ncoils={ncoils} order={order}",
            "surface_params": {"surface": surface, "range": "half period"},
            "coils_params": {"ncoils": int(ncoils), "order": int(order)},
            "optimizer_params": {"algorithm": "augmented_lagrangian", "max_iterations": max_iterations, "verbose": True},
            "coil_objective_terms": coil_objective_terms,
        }
        fc = policy.get("fourier_continuation", {})
        if fc and fc.get("enabled") and fc.get("orders"):
            case_config["fourier_continuation"] = {"enabled": True, "orders": list(fc["orders"])}
        caps = policy.get("resource_caps", {})
        case = {
            "case_id": _new_case_id(),
            "parent_ids": [],
            "tags": ["explore", "llm"],
            "resource": {
                "max_total_iterations": min(max_iterations, caps.get("max_total_iterations", 10000)),
                "timeout_minutes": caps.get("timeout_minutes_max", 60),
            },
            "case_config": case_config,
            "random_seed": rng.randint(0, 2**31 - 1),
        }
        return case

    return None


def propose_batch_llm(
    ctx: Dict[str, Any],
    policy: Dict[str, Any],
    kb_url: str,
    kb_token: str | None,
    batch_size: int = 8,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """Propose a batch of cases using the KB's LLM-powered /propose endpoint.

    Calls the Knowledge Base server to generate mutation/exploration actions
    via an LLM, then converts each action to a validated CI case. Falls back
    to the rule-based proposer if the KB is unreachable, the LLM returns an
    error, or not enough valid cases are produced.

    Parameters
    ----------
    ctx : dict
        Build context from build_context().
    policy : dict
        Proposer policy.
    kb_url : str
        KB server base URL (e.g. http://localhost:8000).
    kb_token : str | None
        Bearer token for KB auth; None for local/unauthenticated KB.
    batch_size : int, optional
        Number of cases to propose (default 8).
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        List of validated CI case dicts ready for cases/pending/.
    """
    try:
        from knowledge.services.kb_client import KBClient
        client = KBClient(base_url=kb_url, token=kb_token)
        resp = client.propose(context=ctx, policy=policy, batch_size=batch_size)
    except Exception as e:
        print(f"KB propose failed: {e}. Falling back to rule-based proposer.", file=sys.stderr)
        return propose_batch(ctx, policy, batch_size=batch_size, seed=seed)

    actions = resp.get("actions", [])
    if resp.get("error"):
        print(f"LLM propose error: {resp['error']}. Falling back to rule-based.", file=sys.stderr)
        return propose_batch(ctx, policy, batch_size=batch_size, seed=seed)

    if not actions:
        return propose_batch(ctx, policy, batch_size=batch_size, seed=seed)

    rng = _rng(seed)
    recent_hashes = set(ctx.get("recent_config_hashes", []))
    cases: List[Dict[str, Any]] = []
    seen_hashes: set = set()

    for action in actions[:batch_size]:
        case = apply_llm_action(action, ctx, policy, rng)
        if not case:
            continue
        h = _config_hash_short(case.get("case_config", {}))
        if h in recent_hashes or h in seen_hashes:
            continue
        errors = validate_ci_case(case, policy=policy)
        if errors:
            continue
        seen_hashes.add(h)
        cases.append(case)

    # Fill remainder with rule-based if LLM didn't produce enough
    if len(cases) < batch_size:
        extra = propose_batch(ctx, policy, batch_size=batch_size - len(cases), seed=seed)
        for c in extra:
            h = _config_hash_short(c.get("case_config", {}))
            if h not in seen_hashes:
                seen_hashes.add(h)
                cases.append(c)
                if len(cases) >= batch_size:
                    break

    return cases[:batch_size]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Number of cases to propose."
    )
    parser.add_argument(
        "--done-dir", type=Path, default=Path("cases/done"),
        help="Directory containing completed case summaries."
    )
    parser.add_argument(
        "--pending-dir", type=Path, default=Path("cases/pending"),
        help="Directory for new pending cases."
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("policy/proposer_policy.yaml"),
        help="Path to proposer_policy.yaml."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print proposed cases to stdout without writing files."
    )
    parser.add_argument(
        "--kb-url", type=str, default=None,
        help="KB base URL for cloud context (e.g. https://kb.example.com)."
    )
    parser.add_argument(
        "--kb-token", type=str, default=None,
        help="Bearer token for KB API (or set KB_TOKEN env var)."
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Use LLM proposer (KB POST /propose). Requires --kb-url."
    )
    args = parser.parse_args()

    # ---- check PAUSE_AUTORUN ----
    pause_file = _REPO_ROOT / "PAUSE_AUTORUN"
    if pause_file.exists():
        print("PAUSE_AUTORUN file exists. Exiting without proposing.", file=sys.stderr)
        return 0

    # ---- barrier check: pending must be empty (skip for dry-run) ----
    pending = args.pending_dir
    if not args.dry_run and pending.is_dir() and any(pending.glob("*.json")):
        print(
            "Pending directory is not empty. Waiting for current batch to finish.",
            file=sys.stderr,
        )
        return 0

    # ---- load policy ----
    if not args.policy.exists():
        print(f"ERROR: policy file not found: {args.policy}", file=sys.stderr)
        return 1
    policy = yaml.safe_load(args.policy.read_text())

    # ---- build context ----
    kb_url = args.kb_url or os.environ.get("KB_URL")
    kb_token = args.kb_token or os.environ.get("KB_TOKEN")
    ctx = build_context(
        args.done_dir,
        args.policy,
        kb_url=kb_url,
        kb_token=kb_token,
    )

    # ---- guardrails ----
    should_stop, reason = check_guardrails(ctx, policy)
    if should_stop:
        print(f"GUARDRAIL TRIGGERED: {reason}", file=sys.stderr)
        cooldown = policy.get("cooldown", {})
        if cooldown.get("write_pause_file", False):
            pause_file.write_text(f"Guardrail: {reason}\n")
            print(f"Created {pause_file}", file=sys.stderr)
        return 0

    # ---- propose ----
    kb_url = args.kb_url or os.environ.get("KB_URL")
    kb_token = args.kb_token or os.environ.get("KB_TOKEN")
    if args.llm:
        if not kb_url:
            print("ERROR: --llm requires --kb-url (or KB_URL env)", file=sys.stderr)
            return 1
        cases = propose_batch_llm(ctx, policy, kb_url, kb_token, batch_size=args.batch_size, seed=args.seed)
    else:
        cases = propose_batch(ctx, policy, batch_size=args.batch_size, seed=args.seed)

    if args.dry_run:
        if args.llm:
            print("Proposer: LLM (KB /propose)", file=sys.stderr)
        elif ctx.get("kb_enriched"):
            print("KB enriched: yes (context from KB server)", file=sys.stderr)
        elif kb_url or os.environ.get("KB_URL"):
            print("KB enriched: no (using local summaries)", file=sys.stderr)
        print(json.dumps(cases, indent=2))
        return 0

    # ---- write cases to pending ----
    pending.mkdir(parents=True, exist_ok=True)
    for case in cases:
        cid = case["case_id"]
        out_path = pending / f"{cid}.json"
        out_path.write_text(json.dumps(case, indent=2))
        print(f"Wrote {out_path}")

    print(f"Proposed {len(cases)} cases.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
