# src/coilbench/update_db.py
from __future__ import annotations

import json
import yaml
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
        import sys
        print(f"Warning: Submissions directory does not exist: {submissions_root}", file=sys.stderr)
        return  # nothing to do

    found_count = 0
    for path in submissions_root.rglob("*.json"):
        # Skip files that are clearly not submission results
        if path.name != "results.json":
            continue
            
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            import sys
            print(f"Warning: Failed to parse JSON from {path}: {e}", file=sys.stderr)
            continue

        meta = data.get("metadata") or {}
        method_name = meta.get("method_name", "UNKNOWN")
        # Use explicit method_version if present, otherwise fall back to dir name.
        version = meta.get("method_version") or path.parent.name
        method_key = f"{method_name}:{version}"
        
        found_count += 1
        yield method_key, path, data
    
    import sys
    if found_count == 0:
        print(f"Warning: No results.json files found in {submissions_root}", file=sys.stderr)
    else:
        print(f"Found {found_count} submission(s) in {submissions_root}", file=sys.stderr)


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


def write_markdown_leaderboard(leaderboard: Dict[str, Any], out_md: Path, case_ids: list[str] | None = None) -> None:
    """
    Write a beautiful markdown leaderboard table to out_md, using leaderboard JSON.
    Combines overall leaderboard with case-by-case navigation.
    """
    entries = leaderboard.get("entries") or []
    case_entries = leaderboard.get("cases") or {}
    case_ids = case_ids or sorted(case_entries.keys())

    lines = [
        "# CoilBench Leaderboard",
        "",
        "Welcome to the CoilBench leaderboard! Compare coil optimization methods across different benchmark cases and plasma surfaces.",
        "",
        "---",
        "",
    ]

    # Navigation links
    nav_lines = []
    if case_ids:
        nav_lines.append("- [Case-by-case leaderboards](#case-by-case-leaderboards)")
    nav_lines.append("- [Plasma surface leaderboards](surfaces.md)")
    if nav_lines:
        lines.append("## Quick Navigation")
        lines.extend(nav_lines)
        lines.append("")

    lines.append("## Overall Leaderboard")
    lines.append("")

    if not entries:
        lines.append("_No valid submissions found._")
        lines.append("")
        lines.append("To add submissions, place `results.json` files in the `submissions/` directory following the format:")
        lines.append("```json")
        lines.append("{")
        lines.append('  "metadata": {')
        lines.append('    "method_name": "your_method",')
        lines.append('    "method_version": "v1.0.0",')
        lines.append('    "contact": "your@email.com",')
        lines.append('    "hardware": "your_hardware"')
        lines.append("  },")
        lines.append('  "cases": [...]')
        lines.append("}")
        lines.append("```")
    else:
        lines.append(
            "| Rank | Method | Version | Run Date | Mean Primary Score | Num Cases | Contact | Hardware |"
        )
        lines.append(
            "|:----:|:-------|:--------|:---------|:-------------------|:----------|:--------|:---------|"
        )
        for e in entries:
            run_date = e.get("run_date") or "_unknown_"
            # Format date nicely if it's an ISO string
            if run_date != "_unknown_" and "T" in run_date:
                run_date = run_date.split("T")[0]
            lines.append(
                f"| {e['rank']} | **{e['method_name']}** | {e['method_version']} | {run_date} | "
                f"**{e['mean_score_primary']:.6f}** | {e.get('num_cases', 0)} | "
                f"{e.get('contact','')} | {e.get('hardware','')} |"
            )

    # Add case-by-case leaderboards section
    if case_ids:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Case-by-case Leaderboards")
        lines.append("")
        lines.append("Jump directly to any benchmark case:")
        lines.append("")
        for cid in case_ids:
            lines.append(f"- [{cid}](leaderboards/{cid}.md)")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Last updated: Run `stellcoilbench update-db` to refresh.*")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def _load_case_to_surface_map(cases_root: Path, plasma_surfaces_dir: Path) -> Dict[str, str]:
    """
    Map case_id -> surface filename by reading case.yaml files.
    
    Returns dict mapping case_id to surface filename (e.g., "input.muse").
    """
    case_to_surface: Dict[str, str] = {}
    
    # Try to find case.yaml files
    for case_yaml in cases_root.rglob("case.yaml"):
        try:
            data = yaml.safe_load(case_yaml.read_text())
            case_id = data.get("case_id")
            surface = data.get("surface_params", {}).get("surface", "")
            if case_id and surface:
                # Extract just the filename if it's a path
                surface_name = Path(surface).name
                case_to_surface[case_id] = surface_name
        except Exception:
            continue
    
    return case_to_surface


def _get_all_metric_keys(case_entries: Dict[str, list[Dict[str, Any]]]) -> list[str]:
    """
    Collect all unique metric keys across all entries.
    Returns sorted list of metric names (excluding score_primary which gets its own column).
    """
    all_keys = set()
    for entries in case_entries.values():
        for entry in entries:
            metrics = entry.get("metrics") or {}
            for key in metrics.keys():
                if key != "score_primary":  # score_primary gets its own column
                    all_keys.add(key)
    return sorted(all_keys)


def write_case_leaderboards(leaderboard: Dict[str, Any], docs_dir: Path, repo_root: Path) -> list[str]:
    """
    Write per-case leaderboard markdown files under docs/leaderboards.
    Now with separate columns for each metric.
    """
    case_entries = leaderboard.get("cases") or {}
    if not case_entries:
        # Ensure directory exists even if empty for downstream tooling.
        (docs_dir / "leaderboards").mkdir(parents=True, exist_ok=True)
        return []

    # Get all unique metric keys to create columns
    all_metric_keys = _get_all_metric_keys(case_entries)

    case_dir = docs_dir / "leaderboards"
    case_dir.mkdir(parents=True, exist_ok=True)

    def _format_value(value: Any) -> str:
        """Format a metric value nicely."""
        if isinstance(value, float):
            # Use scientific notation for very small/large numbers
            if abs(value) < 1e-3 or abs(value) > 1e6:
                return f"{value:.4e}"
            return f"{value:.6f}"
        elif isinstance(value, int):
            return str(value)
        return str(value)

    case_ids = sorted(case_entries.keys())
    for cid in case_ids:
        entries = case_entries[cid]
        lines = [
            f"# {cid} Leaderboard",
            "",
            "[← Back to overall leaderboard](../leaderboard.md)",
            "",
        ]

        if not entries:
            lines.append("_No valid submissions found for this case._")
        else:
            # Build header with separate columns for each metric
            header_cols = ["Rank", "Method", "Version", "Run Date", "Primary Score"]
            header_cols.extend(all_metric_keys)
            header_cols.extend(["Contact", "Hardware"])
            
            # Create header row
            lines.append("| " + " | ".join(header_cols) + " |")
            
            # Create separator row
            sep_parts = []
            for col in header_cols:
                if col == "Rank":
                    sep_parts.append(":----:")
                elif col in ["Method", "Version", "Contact", "Hardware"]:
                    sep_parts.append(":-------")
                else:
                    sep_parts.append(":--------:")
            lines.append("| " + " | ".join(sep_parts) + " |")
            
            # Add data rows
            for entry in entries:
                row_parts = [
                    str(entry.get("rank", "-")),
                    entry.get("method_name", "UNKNOWN"),
                    entry.get("method_version", ""),
                    entry.get("run_date", "_unknown_"),
                ]
                
                # Primary score
                score = entry.get("score_primary")
                row_parts.append(_format_value(score) if isinstance(score, (int, float)) else "_n/a_")
                
                # All metrics
                metrics = entry.get("metrics") or {}
                for key in all_metric_keys:
                    value = metrics.get(key)
                    row_parts.append(_format_value(value) if value is not None else "—")
                
                # Contact and hardware
                row_parts.append(entry.get("contact", ""))
                row_parts.append(entry.get("hardware", ""))
                
                lines.append("| " + " | ".join(row_parts) + " |")

        (case_dir / f"{cid}.md").write_text("\n".join(lines))

    return case_ids




def build_surface_leaderboards(
    leaderboard: Dict[str, Any],
    case_to_surface: Dict[str, str],
    plasma_surfaces_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Group case entries by plasma surface and build per-surface leaderboards.
    
    Returns dict mapping surface_name -> {"entries": [...], "cases": [...]}
    """
    case_entries = leaderboard.get("cases") or {}
    surface_leaderboards: Dict[str, Dict[str, Any]] = {}
    
    # Get all surfaces from plasma_surfaces directory
    all_surfaces = set()
    if plasma_surfaces_dir.exists():
        for surface_file in plasma_surfaces_dir.iterdir():
            if surface_file.is_file():
                all_surfaces.add(surface_file.name)
    
    # Initialize leaderboards for all surfaces
    for surface in sorted(all_surfaces):
        surface_leaderboards[surface] = {"entries": [], "cases": {}}
    
    # Group cases by surface
    for case_id, entries in case_entries.items():
        surface = case_to_surface.get(case_id)
        if not surface:
            # Try to infer from case_id or skip
            continue
        
        if surface not in surface_leaderboards:
            surface_leaderboards[surface] = {"entries": [], "cases": {}}
        
        surface_leaderboards[surface]["cases"][case_id] = entries
    
    # For each surface, aggregate entries across all cases
    for surface, surf_data in surface_leaderboards.items():
        all_surface_entries: Dict[str, Dict[str, Any]] = {}
        
        for case_id, entries in surf_data["cases"].items():
            for entry in entries:
                method_key = entry.get("method_key", "")
                if method_key not in all_surface_entries:
                    all_surface_entries[method_key] = {
                        "method_key": method_key,
                        "method_name": entry.get("method_name", "UNKNOWN"),
                        "method_version": entry.get("method_version", ""),
                        "run_date": entry.get("run_date", ""),
                        "contact": entry.get("contact", ""),
                        "hardware": entry.get("hardware", ""),
                        "cases": [],
                        "scores": [],
                    }
                
                all_surface_entries[method_key]["cases"].append({
                    "case_id": case_id,
                    "metrics": entry.get("metrics", {}),
                })
                score = entry.get("score_primary")
                if isinstance(score, (int, float)):
                    all_surface_entries[method_key]["scores"].append(float(score))
        
        # Calculate mean scores and rank
        for method_key, entry_data in all_surface_entries.items():
            scores = entry_data["scores"]
            if scores:
                entry_data["mean_score_primary"] = float(mean(scores))
                surf_data["entries"].append(entry_data)
        
        # Sort by mean score
        surf_data["entries"].sort(
            key=lambda e: e.get("mean_score_primary", 0.0),
            reverse=True
        )
        for i, entry in enumerate(surf_data["entries"], start=1):
            entry["rank"] = i
    
    return surface_leaderboards


def write_surface_leaderboards(
    surface_leaderboards: Dict[str, Dict[str, Any]],
    docs_dir: Path,
    repo_root: Path,
) -> list[str]:
    """
    Write per-surface leaderboard markdown files with beautiful formatting.
    Each metric gets its own column.
    """
    surface_dir = docs_dir / "surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    
    def _format_value(value: Any) -> str:
        """Format a metric value nicely."""
        if isinstance(value, float):
            if abs(value) < 1e-3 or abs(value) > 1e6:
                return f"{value:.4e}"
            return f"{value:.6f}"
        elif isinstance(value, int):
            return str(value)
        return str(value)
    
    def _get_all_metrics_for_surface(surf_data: Dict[str, Any]) -> list[str]:
        """Get all unique metric keys for a surface."""
        all_keys = set()
        for entry in surf_data.get("entries", []):
            for case_data in entry.get("cases", []):
                metrics = case_data.get("metrics", {})
                for key in metrics.keys():
                    if key != "score_primary":
                        all_keys.add(key)
        return sorted(all_keys)
    
    surface_names = sorted(surface_leaderboards.keys())
    
    for surface_name in surface_names:
        surf_data = surface_leaderboards[surface_name]
        entries = surf_data.get("entries", [])
        
        # Get all metrics for this surface
        all_metric_keys = _get_all_metrics_for_surface(surf_data)
        
        # Create nice display name
        display_name = surface_name.replace("input.", "").replace("_", " ").title()
        
        lines = [
            f"# {display_name} Leaderboard",
            "",
            f"**Plasma Surface:** `{surface_name}`",
            "",
            "[← Back to overall leaderboard](../leaderboard.md) | [View all surfaces](surfaces.md)",
            "",
            "---",
            "",
        ]
        
        if not entries:
            lines.append("_No submissions found for this plasma surface yet._")
            lines.append("")
            lines.append("Submit results using cases that reference this surface to appear on this leaderboard.")
        else:
            # Build header
            header_cols = ["Rank", "Method", "Version", "Run Date", "Mean Score"]
            header_cols.extend(all_metric_keys)
            header_cols.extend(["Contact", "Hardware"])
            
            lines.append("| " + " | ".join(header_cols) + " |")
            
            # Separator
            sep_parts = []
            for col in header_cols:
                if col == "Rank":
                    sep_parts.append(":----:")
                elif col in ["Method", "Version", "Contact", "Hardware"]:
                    sep_parts.append(":-------")
                else:
                    sep_parts.append(":--------:")
            lines.append("| " + " | ".join(sep_parts) + " |")
            
            # Data rows - aggregate metrics across cases
            for entry in entries:
                # Aggregate metrics across all cases for this method
                aggregated_metrics: Dict[str, Any] = {}
                for case_data in entry.get("cases", []):
                    metrics = case_data.get("metrics", {})
                    for key, value in metrics.items():
                        if key != "score_primary":
                            if key not in aggregated_metrics:
                                aggregated_metrics[key] = []
                            if isinstance(value, (int, float)):
                                aggregated_metrics[key].append(float(value))
                
                # Calculate means for aggregated metrics
                final_metrics: Dict[str, Any] = {}
                for key, values in aggregated_metrics.items():
                    if values:
                        final_metrics[key] = float(mean(values))
                
                row_parts = [
                    str(entry.get("rank", "-")),
                    f"**{entry.get('method_name', 'UNKNOWN')}**",
                    entry.get("method_version", ""),
                    entry.get("run_date", "_unknown_"),
                    f"**{_format_value(entry.get('mean_score_primary', 0.0))}**",
                ]
                
                # Add all metrics
                for key in all_metric_keys:
                    value = final_metrics.get(key)
                    row_parts.append(_format_value(value) if value is not None else "—")
                
                # Contact and hardware
                row_parts.append(entry.get("contact", ""))
                row_parts.append(entry.get("hardware", ""))
                
                lines.append("| " + " | ".join(row_parts) + " |")
        
        # Write file
        safe_filename = surface_name.replace(".", "_")
        (surface_dir / f"{safe_filename}.md").write_text("\n".join(lines))
    
    return surface_names


def write_surface_leaderboard_index(surface_names: list[str], docs_dir: Path) -> None:
    """
    Write docs/surfaces.md linking to all per-surface leaderboards.
    """
    index_path = docs_dir / "surfaces.md"
    lines = [
        "# Plasma Surface Leaderboards",
        "",
        "Leaderboards organized by plasma surface configuration:",
        "",
    ]
    
    if not surface_names:
        lines.append("_No surface leaderboards yet. Run `stellcoilbench update-db` after adding submissions._")
    else:
        for surface_name in surface_names:
            display_name = surface_name.replace("input.", "").replace("_", " ").title()
            safe_filename = surface_name.replace(".", "_")
            lines.append(f"- **{display_name}** — [`{surface_name}`](surfaces/{safe_filename}.md)")
        lines.append("")
        lines.append("[← Back to overall leaderboard](leaderboard.md)")
    
    index_path.write_text("\n".join(lines))


def update_database(
    repo_root: Path,
    submissions_root: Path | None = None,
    db_dir: Path | None = None,
    docs_dir: Path | None = None,
    cases_root: Path | None = None,
    plasma_surfaces_dir: Path | None = None,
) -> None:
    """
    High-level entry point to rebuild the on-repo database.

    It does several things:
      1. Scans submissions_root for results.json files
      2. Writes db/methods.json, db/cases.json, db/leaderboard.json
      3. Writes docs/leaderboard.md (overall)
      4. Writes docs/leaderboards/ (per-case)
      5. Writes docs/surfaces/ (per-surface)

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
    cases_root:
        Directory containing case.yaml files. Defaults to repo_root / "cases".
    plasma_surfaces_dir:
        Directory containing plasma surface files. Defaults to repo_root / "plasma_surfaces".
    """
    submissions_root = submissions_root or (repo_root / "submissions")
    db_dir = db_dir or (repo_root / "db")
    docs_dir = docs_dir or (repo_root / "docs")
    cases_root = cases_root or (repo_root / "cases")
    plasma_surfaces_dir = plasma_surfaces_dir or (repo_root / "plasma_surfaces")

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

    # Write per-case leaderboards first (to get case_ids)
    case_ids = write_case_leaderboards(leaderboard, docs_dir=docs_dir, repo_root=repo_root)
    
    # Write overall leaderboard (combined with case-by-case links)
    write_markdown_leaderboard(leaderboard, out_md=docs_dir / "leaderboard.md", case_ids=case_ids)
    
    # Build and write per-surface leaderboards
    case_to_surface = _load_case_to_surface_map(cases_root, plasma_surfaces_dir)
    surface_leaderboards = build_surface_leaderboards(
        leaderboard, case_to_surface, plasma_surfaces_dir
    )
    surface_names = write_surface_leaderboards(
        surface_leaderboards, docs_dir=docs_dir, repo_root=repo_root
    )
    write_surface_leaderboard_index(surface_names, docs_dir=docs_dir)

