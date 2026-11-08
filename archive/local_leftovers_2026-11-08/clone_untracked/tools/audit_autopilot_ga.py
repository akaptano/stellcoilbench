#!/usr/bin/env python3
"""
Detailed performance audit of autopilot submissions (genetic-algorithm proposer).

Loads successful runs from submissions/<surface>/auto/ and failures from
policy/autopilot_failures.json, then produces a structured performance report
covering composite scores, flux, reactor-scale metrics, N_turns, failure rates,
exploit vs explore ratios, and trends.

Usage:
    python -m tools.audit_autopilot_ga [--repo-root .] [--out audit_report.json]
    python -m tools.audit_autopilot_ga [--plots-dir docs/]  # also write PNG plots
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import statistics
from statistics import median, quantiles
from typing import Any, Dict, List

# Add repo root for imports
_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from stellcoilbench.update_db import (
    build_leaderboard_json,
    build_methods_json,
    build_surface_leaderboards,
)
from stellcoilbench.update_db._plot_composite_score import plot_composite_score_vs_date

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
# Suppress update_db verbose constraint warnings during audit
logging.getLogger("stellcoilbench.update_db._metrics_extraction").setLevel(logging.ERROR)


def _load_autopilot_failures(failures_path: Path) -> List[Dict[str, Any]]:
    """Load failure summaries from autopilot_failures.json."""
    if not failures_path.exists():
        return []
    try:
        data = json.loads(failures_path.read_text())
        failures = (
            data
            if isinstance(data, list)
            else (data.get("failures", []) if isinstance(data, dict) else [])
        )
        return failures if isinstance(failures, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _filter_autopilot_entries(methods: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract autopilot (contact=auto) entries from methods dict."""
    entries = []
    for method_key, md in methods.items():
        contact = md.get("contact", "")
        if contact == "auto":
            entries.append({**md, "method_key": method_key})
    return entries


def _parse_case_id_date(case_id: str) -> str | None:
    """Extract YYYY-MM-DD from case_id (e.g. 2026-03-18_110609_87569 -> 2026-03-18)."""
    if not case_id:
        return None
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", case_id)
    if match:
        return match.group(0)
    return None


def _classify_tag(tags: list[str] | None) -> str:
    """Return 'exploit', 'explore', or 'unknown' from tags list."""
    if not tags:
        return "unknown"
    tags_lower = [t.lower() for t in tags if isinstance(t, str)]
    if "exploit" in tags_lower:
        return "exploit"
    if "explore" in tags_lower:
        return "explore"
    return "unknown"


def _load_tags_from_submission(repo_root: Path, path_str: str) -> list[str]:
    """Load metadata.tags from a submission results.json. Returns [] if unavailable."""
    if not path_str:
        return []
    full_path = repo_root / path_str if not Path(path_str).is_absolute() else Path(path_str)
    if not full_path.exists() or full_path.name != "results.json":
        return []
    try:
        data = json.loads(full_path.read_text())
        meta = data.get("metadata") or {}
        tags = meta.get("tags")
        return tags if isinstance(tags, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _load_parent_ids_from_submission(repo_root: Path, path_str: str) -> list[str]:
    """Load metadata.parent_ids from a submission results.json. Returns [] if unavailable."""
    if not path_str:
        return []
    full_path = repo_root / path_str if not Path(path_str).is_absolute() else Path(path_str)
    if not full_path.exists() or full_path.name != "results.json":
        return []
    try:
        data = json.loads(full_path.read_text())
        meta = data.get("metadata") or {}
        pids = meta.get("parent_ids")
        return pids if isinstance(pids, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _entry_sort_key(e: Dict[str, Any]) -> tuple:
    """Return (datetime_ordinal, case_id) for chronological ordering of entries."""
    run_date = e.get("run_date", "")
    case_id = ""
    path_str = e.get("path", "")
    if path_str:
        parts = path_str.replace("\\", "/").split("/")
        case_id = parts[-2] if len(parts) >= 2 and parts[-1] == "results.json" else ""
    date_str = _parse_case_id_date(case_id) or _run_date_to_str(run_date) or ""
    try:
        if run_date and "T" in run_date:
            dt = datetime.fromisoformat(run_date.replace("Z", "+00:00"))
        elif date_str:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        else:
            dt = datetime.min
        return (dt.timestamp(), case_id)
    except (ValueError, TypeError):
        return (0.0, case_id)


def _run_date_to_str(run_date: str) -> str | None:
    """Parse run_date and return YYYY-MM-DD string."""
    if not run_date:
        return None
    try:
        if "T" in run_date:
            dt = datetime.fromisoformat(run_date.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(run_date[:10], "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _compute_audit(
    repo_root: Path,
    submissions_root: Path,
    failures_path: Path,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Build methods, filter autopilot entries, load failures, compute audit stats.

    Returns
    -------
    audit : dict
        Structured audit report.
    leaderboard : dict
        Leaderboard with entries and excluded_entries (for plotting).
    """
    methods = build_methods_json(
        submissions_root=submissions_root, repo_root=repo_root
    )
    leaderboard = build_leaderboard_json(methods)

    auto_entries = _filter_autopilot_entries(methods)
    failures = _load_autopilot_failures(failures_path)

    # Count GA vs LLM from failures (proposer_mode in failure records)
    ga_failures = []
    llm_failures = []
    for f in failures:
        mode = f.get("proposer_mode") or "ga"
        if "llm" in str(mode).lower():
            llm_failures.append(f)
        else:
            ga_failures.append(f)

    total_success = len(auto_entries)
    total_failures = len(failures)
    total_runs = total_success + total_failures
    fail_rate = total_failures / total_runs if total_runs else 0.0

    # Sliding window (last N runs) failure rate
    recent_window = 50
    recent_failures = failures[:recent_window]
    # Approximate recent: we don't have strict ordering, use failure recency
    recent_fail_count = len(recent_failures)
    recent_total_approx = min(recent_window, total_runs)
    recent_fail_rate = recent_fail_count / recent_total_approx if recent_total_approx else 0.0

    # Failure class distribution
    failure_classes: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    for f in failures:
        cls = f.get("failure_class") or "unknown"
        reason = (f.get("failure_reason") or "unknown")[:80]
        failure_classes[cls] += 1
        failure_reasons[reason] += 1

    # Composite score distribution (successful runs)
    composite_scores = [
        float(e.get("composite_score", 0) or 0)
        for e in auto_entries
        if e.get("composite_score") is not None
    ]
    score_primary_list = [
        float(e.get("score_primary", float("inf")) or float("inf"))
        for e in auto_entries
        if e.get("score_primary") is not None
    ]

    # Flux (lower is better)
    flux_vals = [m for m in score_primary_list if m < 1e10]
    flux_vals.sort()

    # Reactor-scale metrics
    n_turns_list: List[float] = []
    for e in auto_entries:
        rs = e.get("reactor_scale_metrics") or {}
        nt = rs.get("N_turns_per_coil")
        if isinstance(nt, list) and nt:
            n_turns_list.append(max(nt))
        elif isinstance(nt, (int, float)):
            n_turns_list.append(float(nt))

    # Per-surface breakdown
    by_surface: Dict[str, List[Dict[str, Any]]] = {}
    for e in auto_entries:
        method_key = e.get("method_key", "")
        parts = method_key.split(":")
        surface = parts[1] if len(parts) >= 2 else "unknown"
        by_surface.setdefault(surface, []).append(e)

    surface_stats: Dict[str, Dict[str, Any]] = {}
    for surface, entries in by_surface.items():
        scores = [
            float(e.get("composite_score", 0) or 0)
            for e in entries
            if e.get("composite_score") is not None
        ]
        fluxes = [
            float(e.get("score_primary", float("inf")) or float("inf"))
            for e in entries
            if e.get("score_primary") is not None and e.get("score_primary") < 1e10
        ]
        surface_stats[surface] = {
            "count": len(entries),
            "composite_score_median": round(median(scores), 4) if scores else None,
            "composite_score_best": round(max(scores), 4) if scores else None,
            "flux_best": min(fluxes) if fluxes else None,
            "flux_median": round(median(fluxes), 6) if fluxes else None,
        }

    # Top entries by composite score
    ranked = sorted(
        auto_entries,
        key=lambda e: (
            -(float(e.get("composite_score") or 0)),
            float(e.get("score_primary") or float("inf")),
        ),
    )
    top_10 = [
        {
            "method_key": e.get("method_key"),
            "composite_score": e.get("composite_score"),
            "score_primary": e.get("score_primary"),
            "num_coils": e.get("metrics", {}).get("num_coils"),
            "fourier_continuation_orders": e.get("metrics", {}).get("fourier_continuation_orders"),
            "N_turns_max": (
                max(e.get("reactor_scale_metrics", {}).get("N_turns_per_coil") or [0])
                if e.get("reactor_scale_metrics")
                else None
            ),
        }
        for e in ranked[:10]
    ]

    # Excluded (constraint failures) count
    excluded = [
        e for e in (leaderboard.get("excluded_entries") or [])
        if e.get("contact") == "auto"
    ]
    passes_constraints_count = sum(1 for e in auto_entries if e.get("passes_constraints", True))
    constraint_fail_count = len(auto_entries) - passes_constraints_count

    # Exploit vs explore: failures have tags; successes need metadata.tags from results.json
    exploit_count = 0
    explore_count = 0
    unknown_count = 0
    by_date: Dict[str, Dict[str, int]] = {}  # date -> {"exploit": n, "explore": n, "unknown": n}

    for f in failures:
        case_id = f.get("case_id", "")
        tags = f.get("tags") or []
        kind = _classify_tag(tags)
        if kind == "exploit":
            exploit_count += 1
        elif kind == "explore":
            explore_count += 1
        else:
            unknown_count += 1
        date_str = _parse_case_id_date(case_id)
        if date_str:
            by_date.setdefault(date_str, {"exploit": 0, "explore": 0, "unknown": 0})
            by_date[date_str][kind] += 1

    for e in auto_entries:
        path_str = e.get("path", "")
        parts = path_str.replace("\\", "/").split("/")
        case_id = parts[-2] if len(parts) >= 2 and parts[-1] == "results.json" else ""
        tags = _load_tags_from_submission(repo_root, path_str) if path_str else []
        kind = _classify_tag(tags)
        if kind == "exploit":
            exploit_count += 1
        elif kind == "explore":
            explore_count += 1
        else:
            unknown_count += 1
        date_str = _parse_case_id_date(case_id) or _run_date_to_str(e.get("run_date", ""))
        if date_str:
            by_date.setdefault(date_str, {"exploit": 0, "explore": 0, "unknown": 0})
            by_date[date_str][kind] += 1

    exploit_explore: Dict[str, Any] = {
        "exploit_count": exploit_count,
        "explore_count": explore_count,
        "unknown_count": unknown_count,
        "by_date": dict(sorted(by_date.items())),
    }

    # Enrich entries with tag and parent_ids for progress analysis
    for e in auto_entries:
        path_str = e.get("path", "")
        e["_tag"] = _classify_tag(
            _load_tags_from_submission(repo_root, path_str) if path_str else []
        )
        e["_parent_ids"] = (
            _load_parent_ids_from_submission(repo_root, path_str) if path_str else []
        )
        parts = path_str.replace("\\", "/").split("/") if path_str else []
        e["_case_id"] = parts[-2] if len(parts) >= 2 and parts[-1] == "results.json" else ""

    # Sort by run date for temporal analysis
    sorted_entries = sorted(auto_entries, key=_entry_sort_key)

    # 1. Temporal progress: by-date stats, cumulative best, first vs last N
    WINDOW_N = 200
    by_date_scores: Dict[str, List[float]] = {}
    cumulative_best: List[tuple] = []  # (date_str, running_max) for JSON; (dt, running_max) for plot
    cumulative_best_dts: List[datetime] = []
    running_max = float("-inf")

    for e in sorted_entries:
        comp = e.get("composite_score")
        if comp is not None:
            comp_val = float(comp)
            if comp_val > running_max:
                running_max = comp_val
            date_str = (
                _parse_case_id_date(e.get("_case_id", ""))
                or _run_date_to_str(e.get("run_date", ""))
                or ""
            )
            if date_str:
                by_date_scores.setdefault(date_str, []).append(comp_val)
                cumulative_best.append((date_str, running_max))
                try:
                    rd = e.get("run_date", "")
                    if rd and "T" in rd:
                        cumulative_best_dts.append(
                            datetime.fromisoformat(rd.replace("Z", "+00:00"))
                        )
                    else:
                        cumulative_best_dts.append(
                            datetime.strptime(date_str, "%Y-%m-%d")
                        )
                except (ValueError, TypeError):
                    cumulative_best_dts.append(datetime.strptime(date_str, "%Y-%m-%d"))

    temporal_by_date: Dict[str, Dict[str, Any]] = {}
    for d, scores in sorted(by_date_scores.items()):
        temporal_by_date[d] = {
            "median": round(median(scores), 4),
            "best": round(max(scores), 4),
            "count": len(scores),
        }

    first_n = [
        float(e.get("composite_score"))
        for e in sorted_entries[:WINDOW_N]
        if e.get("composite_score") is not None
    ]
    last_n = [
        float(e.get("composite_score"))
        for e in sorted_entries[-WINDOW_N:]
        if e.get("composite_score") is not None
    ]
    first_n_passes = sum(
        1 for e in sorted_entries[:WINDOW_N] if e.get("passes_constraints", True)
    )
    last_n_passes = sum(
        1 for e in sorted_entries[-WINDOW_N:] if e.get("passes_constraints", True)
    )

    temporal_progress: Dict[str, Any] = {
        "by_date": temporal_by_date,
        "cumulative_best_dates": [cb[0] for cb in cumulative_best],
        "cumulative_best_values": [cb[1] for cb in cumulative_best],
        "cumulative_best_dts": [d.isoformat() for d in cumulative_best_dts],
        "first_n": {
            "median": round(median(first_n), 4) if first_n else None,
            "best": round(max(first_n), 4) if first_n else None,
            "pass_rate": round(first_n_passes / min(WINDOW_N, len(sorted_entries)), 4)
            if sorted_entries
            else None,
            "count": len(first_n),
        },
        "last_n": {
            "median": round(median(last_n), 4) if last_n else None,
            "best": round(max(last_n), 4) if last_n else None,
            "pass_rate": round(last_n_passes / min(WINDOW_N, len(sorted_entries)), 4)
            if sorted_entries
            else None,
            "count": len(last_n),
        },
        "window_size": WINDOW_N,
    }

    # 2. Exploit vs explore performance (only tagged runs)
    exploit_scores = [
        float(e.get("composite_score", 0) or 0)
        for e in auto_entries
        if e.get("_tag") == "exploit" and e.get("composite_score") is not None
    ]
    explore_scores = [
        float(e.get("composite_score", 0) or 0)
        for e in auto_entries
        if e.get("_tag") == "explore" and e.get("composite_score") is not None
    ]
    exploit_passes = sum(
        1 for e in auto_entries if e.get("_tag") == "exploit" and e.get("passes_constraints", True)
    )
    explore_passes = sum(
        1 for e in auto_entries if e.get("_tag") == "explore" and e.get("passes_constraints", True)
    )
    exploit_tagged = sum(1 for e in auto_entries if e.get("_tag") == "exploit")
    explore_tagged = sum(1 for e in auto_entries if e.get("_tag") == "explore")

    exploit_vs_explore_perf: Dict[str, Any] = {
        "exploit": {
            "median_composite": round(median(exploit_scores), 4) if exploit_scores else None,
            "count": len(exploit_scores),
            "pass_rate": round(exploit_passes / exploit_tagged, 4) if exploit_tagged else None,
            "tagged_total": exploit_tagged,
        },
        "explore": {
            "median_composite": round(median(explore_scores), 4) if explore_scores else None,
            "count": len(explore_scores),
            "pass_rate": round(explore_passes / explore_tagged, 4) if explore_tagged else None,
            "tagged_total": explore_tagged,
        },
    }

    # 3. Parent-child lineage (exploit runs with resolvable parents)
    case_id_to_entry: Dict[str, Dict[str, Any]] = {}
    for e in auto_entries:
        cid = e.get("_case_id", "")
        if cid:
            case_id_to_entry[cid] = e

    parent_child_deltas: List[float] = []
    improvement_count = 0
    total_comparable = 0

    for e in auto_entries:
        if e.get("_tag") != "exploit":
            continue
        parent_ids = e.get("_parent_ids", [])
        if not parent_ids:
            continue
        child_comp = e.get("composite_score")
        if child_comp is None:
            continue
        child_comp = float(child_comp)
        for pid in parent_ids:
            parent = case_id_to_entry.get(pid)
            if parent is None:
                continue
            parent_comp = parent.get("composite_score")
            if parent_comp is None:
                continue
            parent_comp = float(parent_comp)
            total_comparable += 1
            delta = child_comp - parent_comp
            parent_child_deltas.append(delta)
            if child_comp >= parent_comp:
                improvement_count += 1

    parent_child: Dict[str, Any] = {
        "comparable_pairs": total_comparable,
        "improvement_count": improvement_count,
        "improvement_rate": round(improvement_count / total_comparable, 4) if total_comparable else None,
        "mean_delta": round(sum(parent_child_deltas) / len(parent_child_deltas), 4)
        if parent_child_deltas
        else None,
        "deltas": [round(d, 4) for d in parent_child_deltas],
        "deltas_sample": [round(d, 4) for d in parent_child_deltas[:20]],
    }

    # 4. Convergence: rolling variance (window=50)
    comp_chrono: List[tuple] = []
    for e in sorted_entries:
        comp = e.get("composite_score")
        if comp is None:
            continue
        date_str = (
            _parse_case_id_date(e.get("_case_id", ""))
            or _run_date_to_str(e.get("run_date", ""))
            or ""
        )
        comp_chrono.append((date_str, float(comp)))

    rolling_window = 50
    rolling_stds: List[float] = []
    rolling_dates: List[str] = []
    for i in range(rolling_window - 1, len(comp_chrono)):
        window = [c[1] for c in comp_chrono[i - rolling_window + 1 : i + 1]]
        if len(window) >= 2:
            rolling_stds.append(statistics.stdev(window))
            rolling_dates.append(comp_chrono[i][0])

    convergence: Dict[str, Any] = {
        "rolling_window": rolling_window,
        "rolling_stds": [round(s, 4) for s in rolling_stds[-50:]],
        "rolling_dates": rolling_dates[-50:],
        "recent_std": round(rolling_stds[-1], 4) if rolling_stds else None,
        "early_std": round(rolling_stds[0], 4) if rolling_stds else None,
    }

    # 5. LandremanPaul2021_QA only (per-surface trends)
    qa_entries = [e for e in auto_entries if "LandremanPaul2021_QA" in str(e.get("method_key", ""))]
    qa_sorted = sorted(qa_entries, key=_entry_sort_key)
    qa_by_date: Dict[str, List[float]] = {}
    for e in qa_sorted:
        comp = e.get("composite_score")
        if comp is None:
            continue
        date_str = (
            _parse_case_id_date(e.get("_case_id", ""))
            or _run_date_to_str(e.get("run_date", ""))
            or ""
        )
        if date_str:
            qa_by_date.setdefault(date_str, []).append(float(comp))

    qa_temporal: Dict[str, Dict[str, Any]] = {}
    for d, scores in sorted(qa_by_date.items()):
        qa_temporal[d] = {
            "median": round(median(scores), 4),
            "best": round(max(scores), 4),
            "count": len(scores),
        }

    # 6. Constraint satisfaction trend by date
    pass_rate_by_date: Dict[str, List[bool]] = {}
    for e in sorted_entries:
        date_str = (
            _parse_case_id_date(e.get("_case_id", ""))
            or _run_date_to_str(e.get("run_date", ""))
            or ""
        )
        if date_str:
            pass_rate_by_date.setdefault(date_str, []).append(e.get("passes_constraints", True))

    constraint_trend: Dict[str, Any] = {
        "by_date": {
            d: round(sum(passes) / len(passes), 4) if passes else None
            for d, passes in sorted(pass_rate_by_date.items())
        },
    }

    audit: Dict[str, Any] = {
        "summary": {
            "total_successful_runs": total_success,
            "total_failures": total_failures,
            "total_runs": total_runs,
            "overall_fail_rate": round(fail_rate, 4),
            "recent_fail_rate_approx": round(recent_fail_rate, 4),
            "constraint_failures_among_success": constraint_fail_count,
            "excluded_entries_count": len(excluded),
        },
        "composite_score": {
            "count": len(composite_scores),
            "median": round(median(composite_scores), 4) if composite_scores else None,
            "best": round(max(composite_scores), 4) if composite_scores else None,
            "worst": round(min(composite_scores), 4) if composite_scores else None,
            "quartiles": (
                [round(q, 4) for q in quantiles(composite_scores, n=4)]
                if len(composite_scores) >= 4
                else None
            ),
        },
        "flux": {
            "count": len(flux_vals),
            "best": min(flux_vals) if flux_vals else None,
            "median": round(median(flux_vals), 8) if flux_vals else None,
            "quartiles": (
                [round(q, 8) for q in quantiles(flux_vals, n=4)]
                if len(flux_vals) >= 4
                else None
            ),
        },
        "N_turns": {
            "count": len(n_turns_list),
            "min": round(min(n_turns_list), 1) if n_turns_list else None,
            "max": round(max(n_turns_list), 1) if n_turns_list else None,
            "median": round(median(n_turns_list), 1) if n_turns_list else None,
            "over_300_count": sum(1 for n in n_turns_list if n > 300),
        },
        "failure_analysis": {
            "by_class": dict(failure_classes.most_common(10)),
            "by_reason": dict(failure_reasons.most_common(5)),
        },
        "exploit_vs_explore": exploit_explore,
        "by_surface": surface_stats,
        "top_10_by_composite_score": top_10,
        "temporal_progress": temporal_progress,
        "exploit_vs_explore_performance": exploit_vs_explore_perf,
        "parent_child_lineage": parent_child,
        "convergence": convergence,
        "landreman_paul_qa_temporal": qa_temporal,
        "constraint_trend": constraint_trend,
    }
    return audit, leaderboard


def _format_report(audit: Dict[str, Any]) -> str:
    """Produce human-readable report text."""
    lines = [
        "=" * 60,
        "Autopilot (Genetic Algorithm) Performance Audit",
        "=" * 60,
        "",
    ]
    s = audit.get("summary", {})
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Successful runs:     {s.get('total_successful_runs', 0)}")
    lines.append(f"  Failures:            {s.get('total_failures', 0)}")
    lines.append(f"  Total runs:          {s.get('total_runs', 0)}")
    lines.append(f"  Overall fail rate:    {s.get('overall_fail_rate', 0):.2%}")
    lines.append(f"  Constraint failures: {s.get('constraint_failures_among_success', 0)}")
    lines.append("")

    cs = audit.get("composite_score", {})
    lines.append("COMPOSITE SCORE")
    lines.append("-" * 40)
    lines.append(f"  Count:   {cs.get('count', 0)}")
    lines.append(f"  Median:  {cs.get('median')}")
    lines.append(f"  Best:    {cs.get('best')}")
    lines.append(f"  Worst:   {cs.get('worst')}")
    if cs.get("quartiles"):
        lines.append(f"  Q1-Q3:   {cs['quartiles']}")
    lines.append("")

    fl = audit.get("flux", {})
    lines.append("FLUX (score_primary, lower is better)")
    lines.append("-" * 40)
    lines.append(f"  Count:   {fl.get('count', 0)}")
    lines.append(f"  Best:    {fl.get('best')}")
    lines.append(f"  Median:  {fl.get('median')}")
    if fl.get("quartiles"):
        lines.append(f"  Q1-Q3:   {fl['quartiles']}")
    lines.append("")

    nt = audit.get("N_turns", {})
    lines.append("N_TURNS (reactor-scale max per coil)")
    lines.append("-" * 40)
    lines.append(f"  Count:     {nt.get('count', 0)}")
    lines.append(f"  Min/Max:   {nt.get('min')} / {nt.get('max')}")
    lines.append(f"  Median:    {nt.get('median')}")
    lines.append(f"  Over 300: {nt.get('over_300_count', 0)}")
    lines.append("")

    fa = audit.get("failure_analysis", {})
    lines.append("FAILURE ANALYSIS")
    lines.append("-" * 40)
    for cls, cnt in list(fa.get("by_class", {}).items())[:5]:
        lines.append(f"  {cls}: {cnt}")
    lines.append("")

    ee = audit.get("exploit_vs_explore", {})
    lines.append("EXPLOIT VS EXPLORE")
    lines.append("-" * 40)
    lines.append(f"  Exploit:  {ee.get('exploit_count', 0)}")
    lines.append(f"  Explore:  {ee.get('explore_count', 0)}")
    lines.append(f"  Unknown: {ee.get('unknown_count', 0)} (historical successes without tags)")
    lines.append("")

    lines.append("BY SURFACE")
    lines.append("-" * 40)
    for surface, stats in audit.get("by_surface", {}).items():
        lines.append(f"  {surface}:")
        lines.append(f"    count={stats.get('count')}, comp_median={stats.get('composite_score_median')}, flux_best={stats.get('flux_best')}")
    lines.append("")

    lines.append("TOP 10 BY COMPOSITE SCORE")
    lines.append("-" * 40)
    for i, t in enumerate(audit.get("top_10_by_composite_score", [])[:10], 1):
        lines.append(f"  {i}. {t.get('method_key', '')[:50]}...")
        lines.append(f"     composite={t.get('composite_score')}, flux={t.get('score_primary')}, N={t.get('num_coils')}, FC={t.get('fourier_continuation_orders')}, N_turns_max={t.get('N_turns_max')}")
    lines.append("")

    # PROGRESS ASSESSMENT
    lines.append("PROGRESS ASSESSMENT")
    lines.append("-" * 40)
    tp = audit.get("temporal_progress", {})
    fn = tp.get("first_n", {})
    ln = tp.get("last_n", {})
    ws = tp.get("window_size", 200)
    lines.append(f"  First {ws} runs: median={fn.get('median')}, best={fn.get('best')}, pass_rate={fn.get('pass_rate')}")
    lines.append(f"  Last  {ws} runs: median={ln.get('median')}, best={ln.get('best')}, pass_rate={ln.get('pass_rate')}")
    median_improved = (
        fn.get("median") is not None
        and ln.get("median") is not None
        and ln["median"] > fn["median"]
    )
    lines.append(f"  Median composite improved (last vs first): {median_improved}")

    eep = audit.get("exploit_vs_explore_performance", {})
    ex = eep.get("exploit", {})
    er = eep.get("explore", {})
    lines.append("  Exploit vs Explore (tagged runs only):")
    lines.append(f"    Exploit: median={ex.get('median_composite')}, pass_rate={ex.get('pass_rate')}, n={ex.get('count')}")
    lines.append(f"    Explore:  median={er.get('median_composite')}, pass_rate={er.get('pass_rate')}, n={er.get('count')}")
    exploit_better = (
        ex.get("median_composite") is not None
        and er.get("median_composite") is not None
        and ex["median_composite"] > er["median_composite"]
    )
    lines.append(f"  Exploit median > Explore median: {exploit_better}")

    pc = audit.get("parent_child_lineage", {})
    lines.append(f"  Parent-child (exploit): comparable_pairs={pc.get('comparable_pairs')}, improvement_rate={pc.get('improvement_rate')}, mean_delta={pc.get('mean_delta')}")

    conv = audit.get("convergence", {})
    lines.append(f"  Convergence: early_std={conv.get('early_std')}, recent_std={conv.get('recent_std')} (rolling window=50)")

    lines.append("")
    return "\n".join(lines)


def _plot_composite_score_autopilot(
    leaderboard: Dict[str, Any],
    submissions_root: Path,
    plasma_surfaces_dir: Path,
    plots_dir: Path,
) -> bool:
    """Generate composite score vs date plot for autopilot submissions only."""
    auto_entries = [e for e in (leaderboard.get("entries") or []) if e.get("contact") == "auto"]
    if not auto_entries:
        return False
    auto_leaderboard = {"entries": auto_entries}
    surface_leaderboards = build_surface_leaderboards(
        auto_leaderboard, submissions_root, plasma_surfaces_dir
    )
    out_path = plots_dir / "autopilot_composite_score_vs_date.png"
    return plot_composite_score_vs_date(
        surface_leaderboards,
        out_path,
        figsize=(10, 6),
        dpi=150,
    )


def _plot_exploit_vs_explore(audit: Dict[str, Any], plots_dir: Path) -> bool:
    """Generate exploit vs explore stacked bar chart over time."""
    by_date = audit.get("exploit_vs_explore", {}).get("by_date", {})
    if not by_date:
        return False

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping exploit vs explore plot")
        return False

    dates = sorted(by_date.keys())
    exploit_vals = [by_date[d].get("exploit", 0) for d in dates]
    explore_vals = [by_date[d].get("explore", 0) for d in dates]
    unknown_vals = [by_date[d].get("unknown", 0) for d in dates]

    x = np.arange(len(dates))
    width = 0.8

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    bottom = np.zeros(len(dates))
    if any(exploit_vals):
        ax.bar(x, exploit_vals, width, label="Exploit", color="#0173b2", bottom=bottom)
        bottom += np.array(exploit_vals)
    if any(explore_vals):
        ax.bar(x, explore_vals, width, label="Explore", color="#029e73", bottom=bottom)
        bottom += np.array(explore_vals)
    if any(unknown_vals):
        ax.bar(x, unknown_vals, width, label="Unknown", color="#949494", bottom=bottom, alpha=0.7)

    ax.set_ylabel("Cases per day", fontsize=11)
    ax.set_title(
        "Autopilot: Exploit vs Explore over time",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = plots_dir / "autopilot_exploit_vs_explore.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote exploit vs explore plot to %s", out_path)
    return True


def _plot_cumulative_best(audit: Dict[str, Any], plots_dir: Path) -> bool:
    """Generate cumulative best composite score vs date."""
    tp = audit.get("temporal_progress", {})
    values = tp.get("cumulative_best_values", [])
    if not values:
        return False

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping cumulative best plot")
        return False

    dts_iso = tp.get("cumulative_best_dts", [])
    dates = tp.get("cumulative_best_dates", [])
    if dts_iso and len(dts_iso) == len(values):
        dts = []
        for s in dts_iso:
            try:
                dts.append(datetime.fromisoformat(s.replace("Z", "+00:00")))
            except (ValueError, TypeError):
                dts.append(datetime.now())
    elif dates and len(dates) == len(values):
        dts = []
        for d in dates:
            try:
                dts.append(datetime.strptime(d, "%Y-%m-%d"))
            except (ValueError, TypeError):
                dts.append(datetime.now())
    else:
        return False

    if not dts:
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    ax.step(dts, values, where="post", color="#0173b2", linewidth=1.5)
    ax.fill_between(dts, values, alpha=0.2, step="post", color="#0173b2")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative best composite score", fontsize=11)
    ax.set_title(
        "Autopilot: Cumulative best composite score over time",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    plt.xticks(rotation=35, ha="right")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = plots_dir / "autopilot_cumulative_best.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote cumulative best plot to %s", out_path)
    return True


def _plot_median_by_date(audit: Dict[str, Any], plots_dir: Path) -> bool:
    """Generate daily median composite score trend."""
    tp = audit.get("temporal_progress", {}).get("by_date", {})
    if not tp:
        return False

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping median-by-date plot")
        return False

    dates = sorted(tp.keys())
    medians = [tp[d]["median"] for d in dates]
    bests = [tp[d]["best"] for d in dates]
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    ax.plot(dts, medians, "o-", color="#0173b2", label="Daily median", markersize=6)
    ax.plot(dts, bests, "s--", color="#029e73", label="Daily best", markersize=5, alpha=0.8)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Composite score", fontsize=11)
    ax.set_title(
        "Autopilot: Daily median and best composite score",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    plt.xticks(rotation=35, ha="right")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = plots_dir / "autopilot_median_by_date.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote median-by-date plot to %s", out_path)
    return True


def _plot_exploit_vs_explore_scores(audit: Dict[str, Any], leaderboard: Dict[str, Any], repo_root: Path, plots_dir: Path) -> bool:
    """Generate boxplot of composite score by exploit/explore tag."""
    auto_entries = [e for e in (leaderboard.get("entries") or []) if e.get("contact") == "auto"]
    if not auto_entries:
        return False

    exploit_scores = []
    explore_scores = []
    for e in auto_entries:
        path_str = e.get("path", "")
        tags = _load_tags_from_submission(repo_root, path_str) if path_str else []
        kind = _classify_tag(tags)
        comp = e.get("composite_score")
        if comp is None:
            continue
        if kind == "exploit":
            exploit_scores.append(float(comp))
        elif kind == "explore":
            explore_scores.append(float(comp))

    if not exploit_scores and not explore_scores:
        return False

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping exploit vs explore scores plot")
        return False

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    data = []
    labels = []
    if exploit_scores:
        data.append(exploit_scores)
        labels.append(f"Exploit (n={len(exploit_scores)})")
    if explore_scores:
        data.append(explore_scores)
        labels.append(f"Explore (n={len(explore_scores)})")

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
    )
    if len(data) >= 1:
        bp["boxes"][0].set_facecolor("#0173b2")
        bp["boxes"][0].set_alpha(0.6)
    if len(data) >= 2:
        bp["boxes"][1].set_facecolor("#029e73")
        bp["boxes"][1].set_alpha(0.6)

    ax.set_ylabel("Composite score", fontsize=11)
    ax.set_title(
        "Autopilot: Composite score by Exploit vs Explore",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = plots_dir / "autopilot_exploit_vs_explore_scores.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote exploit vs explore scores plot to %s", out_path)
    return True


def _plot_parent_child_delta(audit: Dict[str, Any], plots_dir: Path) -> bool:
    """Generate histogram of (child - parent) composite score for exploit runs."""
    pc = audit.get("parent_child_lineage", {})
    deltas = pc.get("deltas", []) or pc.get("deltas_sample", [])
    if not deltas:
        return False

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping parent-child delta plot")
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    ax.hist(deltas, bins=min(20, max(5, len(deltas) // 2)), color="#0173b2", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="#333", linestyle="--", linewidth=1)
    ax.axvline(pc.get("mean_delta") or 0, color="#de8f05", linestyle="-", linewidth=1.5, label=f"Mean delta={pc.get('mean_delta')}")

    ax.set_xlabel("Child - Parent composite score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Autopilot: Parent-child composite score delta (exploit mutations)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = plots_dir / "autopilot_parent_child_delta.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote parent-child delta plot to %s", out_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON audit to this file",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress human-readable report to stdout",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Write PNG plots (composite score vs date, exploit vs explore) to this directory",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    submissions_root = repo_root / "submissions"
    failures_path = repo_root / "policy" / "autopilot_failures.json"
    plasma_surfaces_dir = repo_root / "plasma_surfaces"

    if not submissions_root.is_dir():
        logger.error("Submissions directory not found: %s", submissions_root)
        return 1

    audit, leaderboard = _compute_audit(repo_root, submissions_root, failures_path)

    if args.plots_dir:
        plots_dir = args.plots_dir.resolve()
        plots_dir.mkdir(parents=True, exist_ok=True)
        _plot_composite_score_autopilot(
            leaderboard, submissions_root, plasma_surfaces_dir, plots_dir
        )
        _plot_exploit_vs_explore(audit, plots_dir)
        _plot_cumulative_best(audit, plots_dir)
        _plot_median_by_date(audit, plots_dir)
        _plot_exploit_vs_explore_scores(audit, leaderboard, repo_root, plots_dir)
        _plot_parent_child_delta(audit, plots_dir)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(audit, indent=2))
        print(f"Wrote JSON audit to {args.out}", file=sys.stderr)

    if not args.quiet:
        print(_format_report(audit))

    return 0


if __name__ == "__main__":
    sys.exit(main())
