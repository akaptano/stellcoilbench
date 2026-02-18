#!/usr/bin/env python3
"""
Build a compact context payload from recent CI results.

Used by the proposer (and eventually an LLM) to decide what to run next.
Output is a JSON dict written to stdout or a file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_policy(policy_path: Path) -> Dict[str, Any]:
    """Load proposer_policy.yaml."""
    return yaml.safe_load(policy_path.read_text())


def _load_summaries(done_dir: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """Load completed case summaries from cases/done/*/summary.json.

    Returns summaries sorted newest-first (by case_id which embeds a date).
    If *limit* is given, return at most that many.
    """
    summaries: List[Dict[str, Any]] = []
    if not done_dir.is_dir():
        return summaries

    for summary_file in sorted(done_dir.glob("*/summary.json"), reverse=True):
        try:
            data = json.loads(summary_file.read_text())
            summaries.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    if limit is not None:
        summaries = summaries[:limit]
    return summaries


def _config_hash(cfg: Dict[str, Any]) -> str:
    """Deterministic hash of a case_config dict for novelty checking."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def compute_failure_stats(
    summaries: List[Dict[str, Any]], window: int = 30,
) -> Dict[str, Any]:
    """Compute failure statistics over a sliding window of recent summaries."""
    recent = summaries[:window]
    if not recent:
        return {
            "window_size": 0,
            "fail_count": 0,
            "fail_rate": 0.0,
            "failure_reasons": {},
            "failure_classes": {},
            "most_common_reason": None,
            "most_common_reason_count": 0,
        }

    fail_count = sum(1 for s in recent if not s.get("success", True))
    fail_rate = fail_count / len(recent) if recent else 0.0

    reasons: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    for s in recent:
        if not s.get("success", True):
            reason = s.get("failure_reason", "unknown")
            cls = s.get("failure_class", "unknown")
            if reason:
                reasons[reason] += 1
            if cls:
                classes[cls] += 1

    most_common = reasons.most_common(1)
    return {
        "window_size": len(recent),
        "fail_count": fail_count,
        "fail_rate": round(fail_rate, 4),
        "failure_reasons": dict(reasons.most_common(10)),
        "failure_classes": dict(classes.most_common(10)),
        "most_common_reason": most_common[0][0] if most_common else None,
        "most_common_reason_count": most_common[0][1] if most_common else 0,
    }


def get_top_parents(
    summaries: List[Dict[str, Any]], top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Return the top-K feasible (success=True) parents sorted by total_score ascending."""
    feasible = [s for s in summaries if s.get("success", False)]
    # Lower total_score is better (it's squared flux)
    feasible.sort(key=lambda s: s.get("total_score", float("inf")))
    parents = []
    for s in feasible[:top_k]:
        parents.append({
            "case_id": s.get("case_id", ""),
            "total_score": s.get("total_score"),
            "iterations_used": s.get("iterations_used"),
            "walltime_sec": s.get("walltime_sec"),
            "metrics": {
                k: v for k, v in s.get("metrics", {}).items()
                if isinstance(v, (int, float))
            },
            "case_config": s.get("case_config", {}),
        })
    return parents


def get_recent_config_hashes(
    summaries: List[Dict[str, Any]], last_n: int = 50,
) -> List[str]:
    """Return config hashes of the last *last_n* runs for novelty checking."""
    hashes = []
    for s in summaries[:last_n]:
        cfg = s.get("case_config", {})
        if cfg:
            hashes.append(_config_hash(cfg))
    return hashes


def _summary_to_parent(s: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a summary dict to top_parents format."""
    return {
        "case_id": s.get("case_id", ""),
        "total_score": s.get("total_score"),
        "iterations_used": s.get("iterations_used"),
        "walltime_sec": s.get("walltime_sec"),
        "metrics": {
            k: v for k, v in s.get("metrics", {}).items()
            if isinstance(v, (int, float))
        },
        "case_config": s.get("case_config", {}),
    }


def build_context(
    done_dir: Path,
    policy_path: Path,
    *,
    max_summaries: int = 200,
    kb_url: str | None = None,
    kb_token: str | None = None,
) -> Dict[str, Any]:
    """Build the full context payload for the proposer.

    Loads summaries from cases/done, computes failure stats and top parents,
    and optionally enriches from the Knowledge Base when kb_url is set.

    Parameters
    ----------
    done_dir : Path
        Directory containing completed case summaries (cases/done/*/summary.json).
    policy_path : Path
        Path to proposer_policy.yaml.
    max_summaries : int, optional
        Maximum number of summaries to load (default 200).
    kb_url : str | None, optional
        KB server URL (e.g. http://localhost:8000). If set, fetches top_parents
        and failure_stats from the KB when available.
    kb_token : str | None, optional
        Bearer token for KB auth. Optional for local KB.

    Returns
    -------
    dict
        Context with policy, failure_stats, top_parents, recent_config_hashes,
        surface_exploration_counts, total_completed, kb_enriched.
    """
    policy = _load_policy(policy_path)
    summaries = _load_summaries(done_dir, limit=max_summaries)

    window = policy.get("guardrails", {}).get("sliding_window", 30)
    top_k = policy.get("top_k_parents", 10)

    failure_stats = compute_failure_stats(summaries, window=window)
    top_parents = get_top_parents(summaries, top_k=top_k)
    config_hashes = get_recent_config_hashes(summaries)
    kb_enriched = False

    # Optionally enrich from KB (cloud or local). Token optional for local KB.
    if kb_url:
        try:
            from knowledge.services.kb_client import KBClient
            client = KBClient(base_url=kb_url, token=kb_token)
            kb_runs = client.get_top_runs(k=top_k, success_only=True)
            if kb_runs:
                top_parents = [_summary_to_parent(r) for r in kb_runs]
            kb_stats = client.stats_recent(window_days=1)
            if kb_stats and "failure_classes" in kb_stats:
                failure_stats = {
                    "window_size": kb_stats.get("total", window),
                    "fail_count": kb_stats.get("fail_count", 0),
                    "fail_rate": kb_stats.get("fail_rate", 0.0),
                    "failure_reasons": kb_stats.get("failure_reasons", {}),
                    "failure_classes": kb_stats.get("failure_classes", {}),
                    "most_common_reason": None,
                    "most_common_reason_count": 0,
                }
                if kb_stats.get("failure_reasons"):
                    mc = max(kb_stats["failure_reasons"].items(), key=lambda x: x[1])
                    failure_stats["most_common_reason"] = mc[0]
                    failure_stats["most_common_reason_count"] = mc[1]
            # Merge config_hashes from KB runs
            for r in kb_runs:
                cfg = r.get("case_config", {})
                if cfg:
                    h = _config_hash(cfg)
                    if h not in config_hashes:
                        config_hashes.append(h)
            kb_enriched = True
        except Exception:
            pass  # Fall back to local context on any KB error

    # Surfaces explored so far (count per surface)
    surface_counts: Counter[str] = Counter()
    for s in summaries:
        cfg = s.get("case_config", {})
        sp = cfg.get("surface_params", {})
        surface_counts[sp.get("surface", "unknown")] += 1

    ctx: Dict[str, Any] = {
        "policy": {
            "batch_size": policy.get("batch_size", 8),
            "exploit_fraction": policy.get("exploit_fraction", 0.5),
            "resource_caps": policy.get("resource_caps", {}),
        },
        "failure_stats": failure_stats,
        "top_parents": top_parents,
        "recent_config_hashes": config_hashes,
        "surface_exploration_counts": dict(surface_counts.most_common()),
        "total_completed": len(summaries),
        "kb_enriched": kb_enriched,
    }
    return ctx


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--done-dir",
        type=Path,
        default=Path("cases/done"),
        help="Directory containing completed case summaries.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("policy/proposer_policy.yaml"),
        help="Path to proposer_policy.yaml.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write context JSON to this file (default: stdout).",
    )
    args = parser.parse_args()

    ctx = build_context(args.done_dir, args.policy)
    text = json.dumps(ctx, indent=2)

    if args.out:
        args.out.write_text(text)
        print(f"Wrote context to {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
