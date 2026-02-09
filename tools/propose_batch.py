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

    The child gets a new ``case_id``, ``random_seed``, and jittered weights/
    thresholds.  The parent ``case_id`` is stored in ``parent_ids``.
    """
    parent_cfg = parent.get("case_config", {})
    child_cfg = copy.deepcopy(parent_cfg)

    mut = policy.get("mutation", {})
    sm = policy.get("safe_mode", {})

    w_sigma = sm.get("weight_sigma_override", mut.get("weight_sigma", 0.15)) if safe else mut.get("weight_sigma", 0.15)
    w_min = float(mut.get("weight_min", 1e-6))
    w_max = float(mut.get("weight_max", 1e4))
    t_sigma = float(mut.get("threshold_sigma", 0.10))
    t_min = float(mut.get("threshold_min", 0.01))
    t_max = float(mut.get("threshold_max", 1000.0))

    # Jitter weights
    obj = child_cfg.get("coil_objective_terms", {})
    if obj is None:
        obj = {}
    weight_keys = [k for k in obj if k.endswith("_weight")]
    for wk in weight_keys:
        old = obj[wk]
        if isinstance(old, (int, float)) and not isinstance(old, bool) and old > 0:
            new_val = old * math.exp(rng.gauss(0, w_sigma))
            obj[wk] = round(_clamp(new_val, w_min, w_max), 6)

    # Jitter thresholds (only numeric values, skip string term-type values)
    threshold_keys = [k for k in obj if k.endswith("_threshold")]
    for tk in threshold_keys:
        old = obj[tk]
        if isinstance(old, (int, float)) and not isinstance(old, bool) and old > 0:
            new_val = old * math.exp(rng.gauss(0, t_sigma))
            obj[tk] = round(_clamp(new_val, t_min, t_max), 6)

    child_cfg["coil_objective_terms"] = obj

    # Possibly change max_iterations slightly
    opt = child_cfg.get("optimizer_params", {})
    if "max_iterations" in opt:
        old_iter = opt["max_iterations"]
        # jitter ±20%
        new_iter = int(old_iter * math.exp(rng.gauss(0, 0.10)))
        iter_cap = sm.get("max_iterations_cap", 10000) if safe else 10000
        new_iter = max(100, min(iter_cap, new_iter))
        opt["max_iterations"] = new_iter
    child_cfg["optimizer_params"] = opt

    # New seed always
    new_seed = rng.randint(0, 2**31 - 1)

    # Resource block
    max_iter = opt.get("max_iterations", 2000)
    resource = {
        "max_total_iterations": min(max_iter, 10000),
        "timeout_minutes": 120,
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
    """Generate a random exploration case from the policy parameter ranges."""
    expl = policy.get("exploration", {})
    sm = policy.get("safe_mode", {})

    # Pick surface
    if safe:
        surfaces = sm.get("preferred_surfaces", expl.get("surfaces", ["input.LandremanPaul2021_QA"]))
    else:
        surfaces = expl.get("surfaces", ["input.LandremanPaul2021_QA"])
    surface = rng.choice(surfaces)

    # Pick algorithm
    algorithms = expl.get("algorithms", ["L-BFGS-B"])
    algorithm = rng.choice(algorithms)

    # Pick coils/order
    ncoils = rng.choice(expl.get("ncoils_choices", [4]))
    order = rng.choice(expl.get("order_choices", [8]))

    # Max iterations
    iter_range = expl.get("max_iterations_range", [1000, 10000])
    iter_cap = sm.get("max_iterations_cap", 10000) if safe else 10000
    max_iterations = rng.randint(iter_range[0], min(iter_range[1], iter_cap))

    # Sample weights (log-uniform)
    def _sample_range(key: str, default: list) -> float:
        lo, hi = expl.get(key, default)
        return round(_log_uniform(rng, lo, hi), 6)

    length_w = _sample_range("length_weight_range", [0.005, 0.10])
    curvature_w = _sample_range("curvature_weight_range", [0.02, 0.30])
    msc_w = _sample_range("msc_weight_range", [0.01, 0.20])
    arclength_w = _sample_range("arclength_variation_weight_range", [0.01, 0.20])

    # Sample thresholds (log-uniform)
    length_t = round(_log_uniform(rng, *expl.get("length_threshold_range", [10, 100])), 2)
    cc_t = round(_log_uniform(rng, *expl.get("cc_threshold_range", [0.1, 2.0])), 3)
    cs_t = round(_log_uniform(rng, *expl.get("cs_threshold_range", [0.1, 3.0])), 3)
    curvature_t = round(_log_uniform(rng, *expl.get("curvature_threshold_range", [0.5, 10.0])), 3)
    msc_t = round(_log_uniform(rng, *expl.get("msc_threshold_range", [0.1, 5.0])), 3)

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
        "coil_objective_terms": {
            "total_length": "l2_threshold",
            "length_threshold": length_t,
            "length_weight": length_w,
            "cc_threshold": cc_t,
            "cs_threshold": cs_t,
            "coil_curvature": "lp_threshold",
            "coil_curvature_p": 2,
            "curvature_threshold": curvature_t,
            "curvature_weight": curvature_w,
            "coil_mean_squared_curvature": "l2_threshold",
            "msc_threshold": msc_t,
            "msc_weight": msc_w,
            "coil_arclength_variation": "l2_threshold",
            "arclength_variation_weight": arclength_w,
            "linking_number": "",
        },
    }

    # Add augmented_lagrangian-specific options
    if algorithm == "augmented_lagrangian":
        case_config["optimizer_params"]["max_iter_subopt"] = max(10, max_iterations // 50)

    new_seed = rng.randint(0, 2**31 - 1)
    resource = {
        "max_total_iterations": min(max_iterations, 10000),
        "timeout_minutes": 120,
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

    # If we still don't have enough, fill with more explore
    extra_attempts = 0
    while len(cases) < batch_size and extra_attempts < batch_size * 3:
        extra_attempts += 1
        child = explore_case(policy, rng, safe=safe)
        errors = validate_ci_case(child, policy=policy)
        if not errors:
            cases.append(child)

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
    args = parser.parse_args()

    # ---- check PAUSE_AUTORUN ----
    pause_file = _REPO_ROOT / "PAUSE_AUTORUN"
    if pause_file.exists():
        print("PAUSE_AUTORUN file exists. Exiting without proposing.", file=sys.stderr)
        return 0

    # ---- barrier check: pending must be empty ----
    pending = args.pending_dir
    if pending.is_dir() and any(pending.glob("*.json")):
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
    ctx = build_context(args.done_dir, args.policy)

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
    cases = propose_batch(ctx, policy, batch_size=args.batch_size, seed=args.seed)

    if args.dry_run:
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
