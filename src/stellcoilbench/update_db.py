# src/coilbench/update_db.py
from __future__ import annotations

import json
from pathlib import Path
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
        Keys are "method_name:version", values hold metadata + metrics.
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
        metrics = data.get("metrics") or {}

        if not metrics:
            # Skip submissions with no metrics
            continue

        metrics_numeric = _numeric_fields(metrics)
        
        # Extract primary score
        primary_score = metrics_numeric.get("score_primary")
        if primary_score is None:
            # Try multiple fallback options for primary score
            fallback = metrics.get("final_flux")
            if fallback is None:
                fallback = metrics.get("final_normalized_squared_flux")
            if isinstance(fallback, (int, float)):
                primary_score = float(fallback)
                metrics_numeric["score_primary"] = primary_score

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
            "score_primary": primary_score,
            "metrics": metrics_numeric,
        }

    return methods


def build_leaderboard_json(methods: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a simple leaderboard summary from methods.json-style data.

    Sorting by score_primary (descending).
    """
    entries = []

    for method_key, md in methods.items():
        score_primary = md.get("score_primary")
        metrics = md.get("metrics", {})
        
        if score_primary is None:
            # Skip entries without a primary score
            continue

        entries.append(
            {
                "method_key": method_key,
                "method_name": md.get("method_name", "UNKNOWN"),
                "method_version": md.get("method_version", ""),
                "score_primary": float(score_primary),
                "run_date": md.get("run_date", ""),
                "contact": md.get("contact", ""),
                "hardware": md.get("hardware", ""),
                "path": md.get("path", ""),
                "metrics": metrics,
            }
        )

    entries.sort(key=lambda e: e["score_primary"], reverse=True)
    for i, e in enumerate(entries, start=1):
        e["rank"] = i

    return {"entries": entries}


def _get_all_metrics_from_entries(entries: list[Dict[str, Any]]) -> list[str]:
    """Get all unique metric keys from overall leaderboard entries."""
    all_keys = set()
    for entry in entries:
        metrics = entry.get("metrics", {})
        for key in metrics.keys():
            if key != "score_primary":  # score_primary gets its own column
                all_keys.add(key)
    return sorted(all_keys)


def write_markdown_leaderboard(leaderboard: Dict[str, Any], out_md: Path) -> None:
    """
    Write a beautiful markdown leaderboard table to out_md, using leaderboard JSON.
    """
    entries = leaderboard.get("entries") or []

    lines = [
        "# CoilBench Leaderboard",
        "",
        "Welcome to the CoilBench leaderboard! Compare coil optimization methods across different plasma surfaces.",
        "",
        "---",
        "",
    ]

    # Navigation links
    nav_lines = ["- [Plasma surface leaderboards](surfaces.md)"]
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
        lines.append('  "metrics": {...}')
        lines.append("}")
        lines.append("```")
    else:
        # Get all unique metric keys across all entries
        all_metric_keys = _get_all_metrics_from_entries(entries)
        
        # Build header: Rank, User, Run Date, Primary Score, then all metrics
        header_cols = ["Rank", "User", "Run Date", "Primary Score"]
        header_cols.extend(all_metric_keys)
        
        lines.append("| " + " | ".join(header_cols) + " |")
        
        # Separator row
        sep_parts = []
        for col in header_cols:
            if col == "Rank":
                sep_parts.append(":----:")
            elif col in ["User", "Run Date"]:
                sep_parts.append(":-------")
            else:
                sep_parts.append(":--------:")
        lines.append("| " + " | ".join(sep_parts) + " |")
        
        def _format_value(value: Any) -> str:
            """Format a metric value nicely."""
            if isinstance(value, float):
                if abs(value) < 1e-3 or abs(value) > 1e6:
                    return f"{value:.4e}"
                return f"{value:.6f}"
            elif isinstance(value, int):
                return str(value)
            return str(value)
        
        # Write rows for each entry
        for e in entries:
            metrics = e.get("metrics", {})
            
            run_date = e.get("run_date") or "_unknown_"
            # Format date nicely if it's an ISO string
            if run_date != "_unknown_" and "T" in run_date:
                run_date = run_date.split("T")[0]
            
            # Build row: Rank, User, Run Date, Primary Score, then all metrics
            row_parts = [
                str(e['rank']),
                f"**{e.get('contact', e.get('method_name', 'UNKNOWN'))}**",  # Use contact as user, fallback to method_name
                run_date,
                f"**{_format_value(e.get('score_primary', 0.0))}**",
            ]
            
            # Add all metrics
            for key in all_metric_keys:
                value = metrics.get(key)
                row_parts.append(_format_value(value) if value is not None else "—")
            
            lines.append("| " + " | ".join(row_parts) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Last updated: Run `stellcoilbench update-db` to refresh.*")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))








def build_surface_leaderboards(
    leaderboard: Dict[str, Any],
    submissions_root: Path,
    plasma_surfaces_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Group entries by plasma surface based on submission directory structure.
    
    Submission paths are: submissions/<surface>/<username>/<datetime>/results.json
    
    Returns dict mapping surface_name -> {"entries": [...]}
    """
    entries = leaderboard.get("entries") or []
    surface_leaderboards: Dict[str, Dict[str, Any]] = {}
    
    # Group entries by surface extracted from submission paths
    for entry in entries:
        # Extract surface from path stored in methods data
        # Path format: submissions/<surface>/<username>/<datetime>/results.json
        path_str = entry.get("path", "")
        if not path_str:
            continue
        
        path_parts = Path(path_str).parts
        # Find "submissions" in path and get the next part (surface name)
        try:
            submissions_idx = path_parts.index("submissions")
            if len(path_parts) > submissions_idx + 1:
                surface_name = path_parts[submissions_idx + 1]
            else:
                continue
        except ValueError:
            continue
        
        if surface_name not in surface_leaderboards:
            surface_leaderboards[surface_name] = {"entries": []}
        
        surface_leaderboards[surface_name]["entries"].append(entry)
    
    # Sort entries within each surface by score_primary
    for surface, surf_data in surface_leaderboards.items():
        entries = surf_data["entries"]
        entries.sort(
            key=lambda e: e.get("score_primary", 0.0),
            reverse=True
        )
        for i, entry in enumerate(entries, start=1):
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
            metrics = entry.get("metrics", {})
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
            header_cols = ["Rank", "User", "Run Date", "Primary Score"]
            header_cols.extend(all_metric_keys)
            
            lines.append("| " + " | ".join(header_cols) + " |")
            
            # Separator
            sep_parts = []
            for col in header_cols:
                if col == "Rank":
                    sep_parts.append(":----:")
                elif col == "User":
                    sep_parts.append(":-------")
                else:
                    sep_parts.append(":--------:")
            lines.append("| " + " | ".join(sep_parts) + " |")
            
            # Data rows
            for entry in entries:
                metrics = entry.get("metrics", {})
                
                run_date = entry.get("run_date", "_unknown_")
                # Format date nicely if it's an ISO string
                if run_date != "_unknown_" and "T" in run_date:
                    run_date = run_date.split("T")[0]
                
                row_parts = [
                    str(entry.get("rank", "-")),
                    f"**{entry.get('contact', entry.get('method_name', 'UNKNOWN'))}**",  # Use contact as user
                    run_date,
                    f"**{_format_value(entry.get('score_primary', 0.0))}**",
                ]
                
                # Add all metrics
                for key in all_metric_keys:
                    value = metrics.get(key)
                    row_parts.append(_format_value(value) if value is not None else "—")
                
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
    High-level entry point to rebuild the leaderboard.

    It does several things:
      1. Scans submissions_root for results.json files
      2. Aggregates data from submissions (in-memory)
      3. Writes docs/leaderboard.md (overall)
      4. Writes docs/leaderboards/ (per-case)
      5. Writes docs/surfaces/ (per-surface)
      6. Optionally writes db/leaderboard.json for reference

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

    # Build in-memory data structures from submissions
    methods = build_methods_json(submissions_root=submissions_root, repo_root=repo_root)
    # cases = build_cases_json(methods)
    leaderboard = build_leaderboard_json(methods)

    # Only write leaderboard.json for reference (optional)
    # methods.json and cases.json are intermediate and not needed on disk
    # Ensure leaderboard always has the expected structure
    if not isinstance(leaderboard, dict):
        leaderboard = {"entries": []}
    if "entries" not in leaderboard:
        leaderboard["entries"] = []
    
    leaderboard_file = db_dir / "leaderboard.json"
    leaderboard_json = json.dumps(leaderboard, indent=2)
    leaderboard_file.write_text(leaderboard_json)
    
    # Verify the file was written correctly
    import sys
    if not leaderboard_file.exists() or leaderboard_file.stat().st_size == 0:
        print("ERROR: leaderboard.json was not written correctly!", file=sys.stderr)
        sys.exit(1)

    # Write overall leaderboard
    write_markdown_leaderboard(leaderboard, out_md=docs_dir / "leaderboard.md")
    
    # Build and write per-surface leaderboards
    surface_leaderboards = build_surface_leaderboards(
        leaderboard, submissions_root, plasma_surfaces_dir
    )
    surface_names = write_surface_leaderboards(
        surface_leaderboards, docs_dir=docs_dir, repo_root=repo_root
    )
    write_surface_leaderboard_index(surface_names, docs_dir=docs_dir)

