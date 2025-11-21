# src/coilbench/update_db.py
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Tuple


def _load_submissions(submissions_root: Path) -> Iterable[Tuple[str, Path, Dict[str, Any]]]:
    """
    Iterate over all submission results.json files under submissions_root.

    Yields
    ------
    (method_key, path, data)
        method_key: "method_name:version_or_run_id"
        path: path to results.json
        data: parsed JSON dict
    """
    if not submissions_root.exists():
        return  # nothing to do

    for path in submissions_root.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            # Skip malformed JSON; you could also log/raise if you prefer.
            continue

        meta = data.get("metadata") or {}
        method_name = meta.get("method_name", "UNKNOWN")
        # Use explicit method_version if present, otherwise fall back to dir name.
        version = meta.get("method_version") or path.parent.name
        method_key = f"{method_name}:{version}"

        yield method_key, path, data


def build_methods_json(
    submissions_root: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    """
    Build the per-method summary dictionary.

    Returns
    -------
    dict
        Keys are "method_name:version", values hold metadata + per-case metrics.
    """
    def _numeric_fields(values: Dict[str, Any]) -> Dict[str, float]:
        return {
            key: float(value)
            for key, value in values.items()
            if isinstance(value, (int, float))
        }

    methods: Dict[str, Any] = {}

    for method_key, path, data in _load_submissions(submissions_root):
        meta = data.get("metadata") or {}
        cases = data.get("cases") or []

        per_case: Dict[str, Dict[str, float]] = {}
        primary_scores = []

        for c in cases:
            cid = c.get("case_id")
            if not cid:
                continue

            c_metrics = c.get("metrics") or {}
            c_scores = c.get("scores") or {}

            metrics_numeric = _numeric_fields(c_metrics)
            scores_numeric = _numeric_fields(c_scores)

            primary_score = scores_numeric.get("score_primary")
            if primary_score is None:
                fallback = c_metrics.get("final_flux")
                if isinstance(fallback, (int, float)):
                    primary_score = float(fallback)
                    scores_numeric.setdefault("score_primary", primary_score)

            if isinstance(primary_score, (int, float)):
                primary_scores.append(float(primary_score))

            per_case[cid] = {**metrics_numeric, **scores_numeric}

        if not per_case:
            # If no valid cases, skip this submission.
            continue

        # Convert path to absolute if it's relative
        abs_path = path if path.is_absolute() else (repo_root / path).resolve()
        rel_path = str(abs_path.relative_to(repo_root.resolve()))

        methods[method_key] = {
            "method_name": meta.get("method_name", "UNKNOWN"),
            "method_version": meta.get("method_version", path.parent.name),
            "contact": meta.get("contact", ""),
            "hardware": meta.get("hardware", ""),
            "run_date": meta.get("run_date", ""),
            "path": rel_path,
            "num_cases": len(per_case),
            "mean_score_primary": float(mean(primary_scores)) if primary_scores else None,
            "per_case": per_case,
        }

    return methods


def build_cases_json(methods: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build per-case summary, including best-so-far entries by score and χ²_Bn.

    Parameters
    ----------
    methods:
        Output of build_methods_json.

    Returns
    -------
    dict
        Keys are case_id, values contain best-by-* entries.
    """
    cases: Dict[str, Any] = {}

    for method_key, md in methods.items():
        per_case = md.get("per_case") or {}
        for cid, cm in per_case.items():
            entry = cases.setdefault(cid, {})

            # Best by score_primary (maximize)
            sp = cm.get("score_primary")
            if isinstance(sp, (int, float)):
                best = entry.get("best_by_score_primary")
                if best is None or sp > best["score_primary"]:
                    entry["best_by_score_primary"] = {
                        "method_key": method_key,
                        "score_primary": float(sp),
                        "chi2_Bn": cm.get("chi2_Bn"),
                    }

            # Best by chi2_Bn (minimize)
            chi2 = cm.get("chi2_Bn")
            if isinstance(chi2, (int, float)):
                best = entry.get("best_by_chi2_Bn")
                if best is None or chi2 < best["chi2_Bn"]:
                    entry["best_by_chi2_Bn"] = {
                        "method_key": method_key,
                        "chi2_Bn": float(chi2),
                        "score_primary": cm.get("score_primary"),
                    }

    return cases


def build_leaderboard_json(methods: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a simple leaderboard summary from methods.json-style data.

    Sorting by mean_score_primary (descending).
    """

    def _case_primary_score(metrics: Dict[str, Any]) -> float | None:
        score = metrics.get("score_primary")
        if isinstance(score, (int, float)):
            return float(score)
        fallback = metrics.get("final_flux")
        if isinstance(fallback, (int, float)):
            return float(fallback)
        return None

    entries = []
    case_entries: Dict[str, list[Dict[str, Any]]] = {}

    for method_key, md in methods.items():
        mean_sp = md.get("mean_score_primary")
        per_case = md.get("per_case") or {}
        cases = []
        for cid in sorted(per_case.keys()):
            cases.append(
                {
                    "case_id": cid,
                    "metrics": per_case[cid],
                }
            )

            case_entry = {
                "method_key": method_key,
                "method_name": md.get("method_name", "UNKNOWN"),
                "method_version": md.get("method_version", ""),
                "run_date": md.get("run_date", ""),
                "contact": md.get("contact", ""),
                "hardware": md.get("hardware", ""),
                "score_primary": _case_primary_score(per_case[cid]),
                "metrics": per_case[cid],
            }
            case_entries.setdefault(cid, []).append(case_entry)

        if isinstance(mean_sp, (int, float)):
            entries.append(
                {
                    "method_key": method_key,
                    "method_name": md.get("method_name", "UNKNOWN"),
                    "method_version": md.get("method_version", ""),
                    "mean_score_primary": float(mean_sp),
                    "num_cases": int(md.get("num_cases", 0)),
                    "run_date": md.get("run_date", ""),
                    "contact": md.get("contact", ""),
                    "hardware": md.get("hardware", ""),
                    "cases": cases,
                }
            )

    entries.sort(key=lambda e: e["mean_score_primary"], reverse=True)
    for i, e in enumerate(entries, start=1):
        e["rank"] = i

    for cid, case_list in case_entries.items():
        case_list.sort(
            key=lambda entry: (
                0 if isinstance(entry.get("score_primary"), (int, float)) else 1,
                -(entry["score_primary"] or 0.0) if entry.get("score_primary") is not None else 0.0,
            )
        )
        for i, entry in enumerate(case_list, start=1):
            entry["rank"] = i

    return {"entries": entries, "cases": case_entries}


def write_markdown_leaderboard(leaderboard: Dict[str, Any], out_md: Path) -> None:
    """
    Write a simple markdown leaderboard table to out_md, using leaderboard JSON.
    """
    entries = leaderboard.get("entries") or []
    case_entries = leaderboard.get("cases") or {}

    def _format_case_block(case: Dict[str, Any]) -> str:
        metrics = case.get("metrics") or {}
        if not metrics:
            return f"**{case.get('case_id', 'unknown')}** — _no metrics_"
        parts = []
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}: {value:.6g}")
            else:
                parts.append(f"{key}: {value}")
        return f"**{case.get('case_id', 'unknown')}** — " + ", ".join(parts)

    lines = ["# CoilBench Leaderboard", ""]

    if case_entries:
        lines.append("## Case-by-case leaderboards")
        for cid in sorted(case_entries.keys()):
            lines.append(f"- [{cid}](leaderboards/{cid}.md)")
        lines.append("")

    lines.append("## Overall leaderboard")

    if not entries:
        lines.append("_No valid submissions found._")
    else:
        lines.append(
            "| Rank | Method | Run date | Mean primary score | Cases & metrics | Contact | Hardware |"
        )
        lines.append(
            "|:----:|:-------|:---------|:-------------------|:---------------|:--------|:---------|"
        )
        for e in entries:
            cases = e.get("cases") or []
            case_block = "<br>".join(_format_case_block(c) for c in cases) if cases else "_No cases_"
            run_date = e.get("run_date") or "_unknown_"
            lines.append(
                f"| {e['rank']} | {e['method_name']} ({e['method_version']}) | {run_date} | "
                f"{e['mean_score_primary']:.3f} | {case_block} | {e.get('contact','')} | {e.get('hardware','')} |"
            )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def write_case_leaderboards(leaderboard: Dict[str, Any], docs_dir: Path) -> list[str]:
    """
    Write per-case leaderboard markdown files under docs/leaderboards.
    """
    case_entries = leaderboard.get("cases") or {}
    if not case_entries:
        # Ensure directory exists even if empty for downstream tooling.
        (docs_dir / "leaderboards").mkdir(parents=True, exist_ok=True)
        return []

    def _format_metrics(metrics: Dict[str, Any]) -> str:
        if not metrics:
            return "_no metrics_"
        ordered = []
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, float):
                ordered.append(f"{key}: {value:.6g}")
            else:
                ordered.append(f"{key}: {value}")
        return "<br>".join(ordered)

    case_dir = docs_dir / "leaderboards"
    case_dir.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(case_entries.keys())
    for cid in case_ids:
        entries = case_entries[cid]
        lines = [
            f"# {cid} Leaderboard",
            "",
            "[Back to overall leaderboard](../leaderboard.md)",
            "",
        ]

        if not entries:
            lines.append("_No valid submissions found for this case._")
        else:
            lines.append("| Rank | Method | Run date | Primary score | Metrics | Contact | Hardware |")
            lines.append("|:----:|:-------|:---------|:--------------|:--------|:--------|:---------|")
            for entry in entries:
                run_date = entry.get("run_date") or "_unknown_"
                score = entry.get("score_primary")
                score_str = f"{score:.6f}" if isinstance(score, (int, float)) else "_n/a_"
                metrics_block = _format_metrics(entry.get("metrics") or {})
                lines.append(
                    f"| {entry.get('rank', '-') } | {entry.get('method_name', 'UNKNOWN')} "
                    f"({entry.get('method_version', '')}) | {run_date} | {score_str} | "
                    f"{metrics_block} | {entry.get('contact', '')} | {entry.get('hardware', '')} |"
                )

        (case_dir / f"{cid}.md").write_text("\n".join(lines))

    return case_ids


def write_case_leaderboard_index(case_ids: list[str], docs_dir: Path) -> None:
    """
    Write docs/leaderboards.md linking to all per-case leaderboards.
    """
    index_path = docs_dir / "leaderboards.md"
    lines = ["# Case Leaderboards", ""]

    if not case_ids:
        lines.append("_No per-case leaderboards yet. Run `stellcoilbench update-db` after adding submissions._")
    else:
        lines.append("Jump directly to any benchmark case:")
        lines.append("")
        for cid in case_ids:
            lines.append(f"- [{cid}](leaderboards/{cid}.md)")

    index_path.write_text("\n".join(lines))


def update_database(
    repo_root: Path,
    submissions_root: Path | None = None,
    db_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> None:
    """
    High-level entry point to rebuild the on-repo database.

    It does three things:
      1. Scans submissions_root for results.json files
      2. Writes db/methods.json, db/cases.json, db/leaderboard.json
      3. Writes docs/leaderboard.md

    Parameters
    ----------
    repo_root:
        Root of the git repo (e.g. Path.cwd() when called from repo root).
    submissions_root:
        Directory containing per-method submissions. Defaults to repo_root / "submissions".
    db_dir:
        Directory where JSON database files are stored. Defaults to repo_root / "db".
    docs_dir:
        Directory where docs/leaderboard.md is written. Defaults to repo_root / "docs".
    """
    submissions_root = submissions_root or (repo_root / "submissions")
    db_dir = db_dir or (repo_root / "db")
    docs_dir = docs_dir or (repo_root / "docs")

    db_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    methods = build_methods_json(submissions_root=submissions_root, repo_root=repo_root)
    cases = build_cases_json(methods)
    leaderboard = build_leaderboard_json(methods)

    # Ensure all JSON files have .json extension
    methods_file = db_dir / "methods.json"
    cases_file = db_dir / "cases.json"
    leaderboard_file = db_dir / "leaderboard.json"
    
    methods_file.write_text(json.dumps(methods, indent=2))
    cases_file.write_text(json.dumps(cases, indent=2))
    leaderboard_file.write_text(json.dumps(leaderboard, indent=2))

    write_markdown_leaderboard(leaderboard, out_md=docs_dir / "leaderboard.md")
    case_ids = write_case_leaderboards(leaderboard, docs_dir=docs_dir)
    write_case_leaderboard_index(case_ids, docs_dir=docs_dir)

