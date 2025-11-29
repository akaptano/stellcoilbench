# src/coilbench/update_db.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _metric_shorthand(metric_name: str) -> str:
    """
    Convert metric names to compact shorthand/acronyms for display in leaderboard.
    
    Uses LaTeX-style notation where appropriate for compactness.
    """
    shorthand_map = {
        # B-field related
        "max_BdotN_over_B": "max⟨Bn⟩/⟨B⟩",
        "avg_BdotN_over_B": "avg⟨Bn⟩/⟨B⟩",
        "final_normalized_squared_flux": "f_B",
        "initial_B_field": "B0",
        "final_B_field": "Bf",
        "target_B_field": "Bt",
        
        # Curvature
        "final_average_curvature": "κ̄",
        "final_max_curvature": "max(κ)",
        "final_mean_squared_curvature": "MSC",
        
        # Separations
        "final_min_cs_separation": "min(d_cs)",
        "final_min_cc_separation": "min(d_cc)",
        "final_cs_separation": "d_cs",
        "final_cc_separation": "d_cc",
        
        # Length
        "final_total_length": "L",
        
        # Forces/Torques
        "final_max_max_coil_force": "max(F)",
        "final_avg_max_coil_force": "F̄",
        "final_max_max_coil_torque": "max(τ)",
        "final_avg_max_coil_torque": "τ̄",
        
        # Time
        "optimization_time": "t",
        
        # Linking number
        "final_linking_number": "LN",
        
        # Coil parameters
        "coil_order": "n",
        "num_coils": "N",
        
        # Score (keep for sorting but don't display)
        "score_primary": "score",
    }
    
    return shorthand_map.get(metric_name, metric_name.replace("_", " "))


def _metric_definition(metric_name: str) -> str:
    """
    Get detailed mathematical definition for a metric.
    
    Returns a string with LaTeX-style mathematical notation describing the metric.
    Format: symbol = expression - description
    """
    definitions = {
        # B-field related
        "final_normalized_squared_flux": r"$f_B = \frac{1}{|S|} \int_{S} \left(\frac{\mathbf{B} \cdot \mathbf{n}}{|\mathbf{B}|}\right)^2 dS$ - Normalized squared flux error on plasma surface",
        "max_BdotN_over_B": r"$\max\left(\frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}\right)$ - Maximum normalized normal field component",
        "avg_BdotN_over_B": r"$\frac{1}{|S|} \int_{S} \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|} dS$ - Average normalized normal field component",
        
        # Curvature
        "final_average_curvature": r"$\bar{\kappa} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i$ - Mean curvature over all coils, where $\kappa_i = |\mathbf{r}''(s)|$",
        "final_max_curvature": r"$\max(\kappa)$ - Maximum curvature across all coils",
        "final_mean_squared_curvature": r"$\text{MSC} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i^2$ - Mean squared curvature",
        
        # Separations
        "final_min_cs_separation": r"$\min(d_{cs})$ - Minimum coil-to-surface distance",
        "final_min_cc_separation": r"$\min(d_{cc})$ - Minimum coil-to-coil distance",
        "final_cs_separation": r"$d_{cs}$ - Average coil-to-surface separation",
        "final_cc_separation": r"$d_{cc}$ - Average coil-to-coil separation",
        
        # Length
        "final_total_length": r"$L = \sum_{i=1}^{N} \int_{0}^{L_i} ds$ - Total length of all coils",
        
        # Forces/Torques
        "final_max_max_coil_force": r"$\max(|\mathbf{F}_i|)$ - Maximum force magnitude across all coils",
        "final_avg_max_coil_force": r"$\bar{F} = \frac{1}{N} \sum_{i=1}^{N} \max(|\mathbf{F}_i|)$ - Average of maximum force per coil",
        "final_max_max_coil_torque": r"$\max(|\boldsymbol{\tau}_i|)$ - Maximum torque magnitude across all coils",
        "final_avg_max_coil_torque": r"$\bar{\tau} = \frac{1}{N} \sum_{i=1}^{N} \max(|\boldsymbol{\tau}_i|)$ - Average of maximum torque per coil",
        
        # Time
        "optimization_time": r"$t$ - Total optimization time (seconds)",
        
        # Linking number
        "final_linking_number": r"$\text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot (d\mathbf{r}_i \times d\mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^3}$ - Linking number between coil pairs",
        
        # Coil parameters
        "coil_order": r"$n$ - Fourier order of coil representation: $\mathbf{r}(\phi) = \sum_{m=-n}^{n} \mathbf{c}_m e^{im\phi}$",
        "num_coils": r"$N$ - Number of base coils (before applying stellarator symmetry)",
    }
    
    return definitions.get(metric_name, metric_name.replace("_", " ").title())


def _load_submissions(submissions_root: Path) -> Iterable[Tuple[str, Path, Dict[str, Any]]]:
    """
    Iterate over all submission results.json files under submissions_root.
    
    Handles both regular directories and zip files. For zip files, extracts
    results.json and case.yaml temporarily to read them.

    Yields
    ------
    (method_key, path, data)
        method_key: "method_name:version_or_run_id"
        path: path to results.json (or zip file containing it)
        data: parsed JSON dict
    """
    import zipfile
    
    if not submissions_root.exists():
        import sys
        print(f"Warning: Submissions directory does not exist: {submissions_root}", file=sys.stderr)
        return  # nothing to do

    found_count = 0
    
    # First, handle regular JSON files in directories
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
        
        # Extract surface and user from path to make method_key unique
        path_parts = path.parts
        surface = "unknown"
        user = "unknown"
        if "submissions" in path_parts:
            submissions_idx = path_parts.index("submissions")
            if submissions_idx + 1 < len(path_parts):
                surface = path_parts[submissions_idx + 1]
            if submissions_idx + 2 < len(path_parts):
                user = path_parts[submissions_idx + 2]
        
        # Use explicit method_version if present, otherwise fall back to dir name.
        # For zip files, use the zip filename (without .zip extension)
        if path.suffix == ".zip":
            version = meta.get("method_version") or path.stem
        else:
            version = meta.get("method_version") or path.parent.name
        
        # Include surface and user in method_key to ensure uniqueness
        method_key = f"{method_name}:{surface}:{user}:{version}"
        
        found_count += 1
        yield method_key, path, data
    
    # Second, handle zip files (submission directories that were zipped)
    for zip_path in submissions_root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Check if results.json exists in the zip
                if "results.json" not in zf.namelist():
                    continue
                
                # Read results.json from zip
                results_json_content = zf.read("results.json")
                data = json.loads(results_json_content.decode('utf-8'))
                
                meta = data.get("metadata") or {}
                method_name = meta.get("method_name", "UNKNOWN")
                
                # Extract surface and user from path to make method_key unique
                # Path format: submissions/{surface}/{user}/{timestamp}.zip
                path_parts = zip_path.parts
                surface = "unknown"
                user = "unknown"
                if "submissions" in path_parts:
                    submissions_idx = path_parts.index("submissions")
                    if submissions_idx + 1 < len(path_parts):
                        surface = path_parts[submissions_idx + 1]
                    if submissions_idx + 2 < len(path_parts):
                        user = path_parts[submissions_idx + 2]
                
                # Use zip filename (without .zip) as version/directory name
                version = zip_path.stem  # Remove .zip extension
                # Include surface and user in method_key to ensure uniqueness
                method_key = f"{method_name}:{surface}:{user}:{version}"
                
                found_count += 1
                # Yield with zip_path as the path (even though results.json is inside)
                yield method_key, zip_path, data
        except Exception as e:
            import sys
            print(f"Warning: Failed to read zip file {zip_path}: {e}", file=sys.stderr)
            continue
    
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
    import yaml
    
    def _numeric_fields(values: Dict[str, Any]) -> Dict[str, float]:
        return {
            key: float(value)
            for key, value in values.items()
            if isinstance(value, (int, float))
        }

    methods: Dict[str, Any] = {}

    loaded_count = 0
    skipped_no_metrics = 0
    skipped_no_score = 0
    duplicate_keys = {}  # Track duplicate method_keys
    
    for method_key, path, data in _load_submissions(submissions_root):
        loaded_count += 1
        meta = data.get("metadata") or {}
        metrics = data.get("metrics") or {}

        if not metrics:
            # Skip submissions with no metrics
            skipped_no_metrics += 1
            import sys
            print(f"Warning: Skipping {path} - no metrics found", file=sys.stderr)
            continue

        # Track duplicate method_keys (warn but still process - later overwrites earlier)
        if method_key in methods:
            if method_key not in duplicate_keys:
                duplicate_keys[method_key] = [methods[method_key].get('path')]  # Include the first one
            duplicate_keys[method_key].append(str(path))  # Add the duplicate
            import sys
            print(f"Warning: Duplicate method_key '{method_key}'. Previous: {methods[method_key].get('path')}, New: {path} (will overwrite)", file=sys.stderr)

        metrics_numeric = _numeric_fields(metrics)
        
        # Extract coil parameters from case.yaml if available
        # Handle both regular directories and zip files
        import zipfile
        case_yaml_data = None
        
        if path.suffix == ".zip":
            # Read case.yaml from zip file
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    if "case.yaml" in zf.namelist():
                        case_yaml_content = zf.read("case.yaml")
                        case_yaml_data = yaml.safe_load(case_yaml_content.decode('utf-8'))
            except Exception as e:
                import sys
                print(f"Warning: Failed to load case.yaml from zip {path}: {e}", file=sys.stderr)
        else:
            # Read case.yaml from regular directory
            case_yaml_path = path.parent / "case.yaml"
            if case_yaml_path.exists():
                try:
                    case_yaml_data = yaml.safe_load(case_yaml_path.read_text())
                except Exception as e:
                    import sys
                    print(f"Warning: Failed to load case.yaml from {case_yaml_path}: {e}", file=sys.stderr)
        
        if case_yaml_data:
            coils_params = case_yaml_data.get("coils_params", {})
            # Add coil order and number of coils to metrics
            if "order" in coils_params:
                metrics_numeric["coil_order"] = float(coils_params["order"])
            if "ncoils" in coils_params:
                metrics_numeric["num_coils"] = float(coils_params["ncoils"])
        
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
            elif fallback is not None:
                # Log warning if fallback exists but is not numeric
                import sys
                print(f"Warning: fallback score '{fallback}' (type {type(fallback).__name__}) is not numeric for {path}", file=sys.stderr)

        # Convert path to absolute if it's relative
        abs_path = path if path.is_absolute() else (repo_root / path).resolve()
        rel_path = str(abs_path.relative_to(repo_root.resolve()))

        if primary_score is None:
            skipped_no_score += 1
        
        methods[method_key] = {
            "method_name": meta.get("method_name", "UNKNOWN"),
            "method_version": meta.get("method_version", path.stem if path.suffix == ".zip" else path.parent.name),
            "contact": meta.get("contact", ""),
            "hardware": meta.get("hardware", ""),
            "run_date": meta.get("run_date", ""),
            "path": rel_path,
            "score_primary": primary_score,
            "metrics": metrics_numeric,
        }
    
    # Log summary
    import sys
    total_duplicates = sum(len(paths) - 1 for paths in duplicate_keys.values())  # -1 because first one isn't a duplicate
    print(f"Loaded {loaded_count} submissions, skipped {skipped_no_metrics} (no metrics), {skipped_no_score} will be filtered (no score)", file=sys.stderr)
    if duplicate_keys:
        print(f"Found {len(duplicate_keys)} duplicate method_keys ({total_duplicates} overwrites):", file=sys.stderr)
        for key, paths in duplicate_keys.items():
            print(f"  {key}: {len(paths)} total (first kept, {len(paths)-1} overwritten)", file=sys.stderr)
    expected_entries = loaded_count - skipped_no_metrics - total_duplicates
    print(f"Methods dict has {len(methods)} entries (expected: {expected_entries}, loaded: {loaded_count}, skipped: {skipped_no_metrics}, duplicates: {total_duplicates})", file=sys.stderr)

    return methods


def build_leaderboard_json(methods: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a simple leaderboard summary from methods.json-style data.

    Sorting by score_primary (ascending - lower normalized squared flux is better).
    """
    entries = []
    filtered_count = 0

    for method_key, md in methods.items():
        score_primary = md.get("score_primary")
        metrics = md.get("metrics", {})
        path = md.get("path", "")
        
        if score_primary is None:
            # Skip entries without a primary score
            # Log which entries are being filtered out for debugging
            filtered_count += 1
            import sys
            print(f"Warning: Filtering out entry {path} (score_primary is None, metrics keys: {list(metrics.keys())[:5]})", file=sys.stderr)
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

    entries.sort(key=lambda e: e["score_primary"], reverse=False)
    for i, e in enumerate(entries, start=1):
        e["rank"] = i

    import sys
    print(f"Leaderboard: {len(entries)} entries included, {filtered_count} filtered out (no score_primary)", file=sys.stderr)

    return {"entries": entries}


def _get_all_metrics_from_entries(entries: list[Dict[str, Any]]) -> list[str]:
    """Get all unique metric keys from overall leaderboard entries."""
    # Fields to exclude from display
    exclude_fields = {
        "score_primary",  # Used for sorting only
        "initial_B_field",  # B0 - removed per request
        "final_B_field",  # Bf - removed per request
        "target_B_field",  # Bt - removed per request
    }
    
    all_keys = set()
    for entry in entries:
        metrics = entry.get("metrics", {})
        for key in metrics.keys():
            if key not in exclude_fields:
                all_keys.add(key)
    
    # Sort with final_normalized_squared_flux first
    sorted_keys = sorted(all_keys)
    if "final_normalized_squared_flux" in sorted_keys:
        sorted_keys.remove("final_normalized_squared_flux")
        sorted_keys.insert(0, "final_normalized_squared_flux")
    
    return sorted_keys


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
    nav_lines = ["- [Plasma surface leaderboards](leaderboards/)"]
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
        
        # Build header: Rank, User, Date, then all metrics (compact)
        header_cols = ["#", "User", "Date"]
        # Add metric shorthands
        header_cols.extend([_metric_shorthand(key) for key in all_metric_keys])
        
        lines.append("| " + " | ".join(header_cols) + " |")
        
        # Separator row
        sep_parts = []
        for col in header_cols:
            if col == "#":
                sep_parts.append(":-:")
            elif col == "User":
                sep_parts.append(":---")
            elif col == "Date":
                sep_parts.append(":---:")
            else:
                sep_parts.append(":---:")
        lines.append("| " + " | ".join(sep_parts) + " |")
        
        def _format_value(value: Any, metric_key: str = "") -> str:
            """Format a metric value in scientific notation with 2 digits."""
            # Special handling for linking number - use integer format
            if metric_key == "final_linking_number":
                if isinstance(value, (float, int)):
                    return str(int(round(value)))
                return str(value)
            # All other numeric values use scientific notation with 2 digits
            if isinstance(value, (float, int)):
                return f"{float(value):.2e}"
            return str(value)
        
        # Write rows for each entry
        for e in entries:
            metrics = e.get("metrics", {})
            
            run_date = e.get("run_date") or "_unknown_"
            # Format date compactly (just date part)
            if run_date != "_unknown_" and "T" in run_date:
                run_date = run_date.split("T")[0]
            
            # Build row: Rank, User, Date, then all metrics
            row_parts = [
                str(e['rank']),
                e.get('contact', e.get('method_name', '?'))[:15],  # Truncate long names
                run_date,
            ]
            
            # Add all metrics
            for key in all_metric_keys:
                value = metrics.get(key)
                row_parts.append(_format_value(value, metric_key=key) if value is not None else "—")
            
            lines.append("| " + " | ".join(row_parts) + " |")
        
        # Add legend for acronyms
        lines.append("")
        lines.append("### Legend")
        lines.append("")
        
        # Build legend from displayed metrics
        legend_items = []
        for key in all_metric_keys:
            shorthand = _metric_shorthand(key)
            full_name = key.replace("_", " ").title()
            legend_items.append(f"- **{shorthand}**: {full_name}")
        
        lines.extend(legend_items)
        lines.append("")

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
            key=lambda e: e.get("score_primary", float('inf')),  # Use inf for missing scores (sort last)
            reverse=False  # Ascending order - lower normalized squared flux is better
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
    surface_dir = docs_dir / "leaderboards"
    surface_dir.mkdir(parents=True, exist_ok=True)
    if not surface_dir.exists() or not surface_dir.is_dir():
        import sys
        raise RuntimeError(f"Failed to create or access surface_dir: {surface_dir}")
    
    def _format_value(value: Any, metric_key: str = "") -> str:
        """Format a metric value in scientific notation with 2 digits."""
        # Special handling for integer metrics - use integer format
        integer_metrics = {"final_linking_number", "coil_order", "num_coils"}
        if metric_key in integer_metrics:
            if isinstance(value, (float, int)):
                return str(int(round(value)))
            return str(value)
        # All other numeric values use scientific notation with 2 digits
        if isinstance(value, (float, int)):
            return f"{float(value):.2e}"
        return str(value)
    
    def _get_all_metrics_for_surface(surf_data: Dict[str, Any]) -> list[str]:
        """Get all unique metric keys for a surface."""
        # Fields to exclude from display
        exclude_fields = {
            "score_primary",  # Used for sorting only
            "initial_B_field",  # B0 - removed per request
            "final_B_field",  # Bf - removed per request
            "target_B_field",  # Bt - removed per request
        }
        
        all_keys = set()
        for entry in surf_data.get("entries", []):
            metrics = entry.get("metrics", {})
            for key in metrics.keys():
                if key not in exclude_fields:
                    all_keys.add(key)
        
        # Sort with priority order: primary metric first, then coil parameters, then others
        sorted_keys = sorted(all_keys)
        
        # Priority order for display
        priority_order = [
            "final_normalized_squared_flux",  # Primary metric
            "num_coils",  # Coil configuration
            "coil_order",  # Coil configuration
        ]
        
        # Reorder: priority items first, then rest alphabetically
        ordered_keys = []
        for priority_key in priority_order:
            if priority_key in sorted_keys:
                ordered_keys.append(priority_key)
                sorted_keys.remove(priority_key)
        
        # Add remaining keys alphabetically
        ordered_keys.extend(sorted(sorted_keys))
        
        return ordered_keys
    
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
            "[View all surfaces](../leaderboards/)",
            "",
            "---",
            "",
        ]
        
        if not entries:
            lines.append("_No submissions found for this plasma surface yet._")
            lines.append("")
            lines.append("Submit results using cases that reference this surface to appear on this leaderboard.")
        else:
            # Build header (compact)
            header_cols = ["#", "User", "Date"]
            # Add metric shorthands
            header_cols.extend([_metric_shorthand(key) for key in all_metric_keys])
            
            lines.append("| " + " | ".join(header_cols) + " |")
            
            # Separator
            sep_parts = []
            for col in header_cols:
                if col == "#":
                    sep_parts.append(":-:")
                elif col == "User":
                    sep_parts.append(":---")
                elif col == "Date":
                    sep_parts.append(":---:")
                else:
                    sep_parts.append(":---:")
            lines.append("| " + " | ".join(sep_parts) + " |")
            
            # Data rows
            for entry in entries:
                metrics = entry.get("metrics", {})
                
                run_date = entry.get("run_date", "_unknown_")
                # Format date compactly
                if run_date != "_unknown_" and "T" in run_date:
                    run_date = run_date.split("T")[0]
                
                row_parts = [
                    str(entry.get("rank", "-")),
                    entry.get('contact', entry.get('method_name', '?'))[:15],  # Truncate long names
                    run_date,
                ]
                
                # Add all metrics
                for key in all_metric_keys:
                    value = metrics.get(key)
                    row_parts.append(_format_value(value, metric_key=key) if value is not None else "—")
                
                lines.append("| " + " | ".join(row_parts) + " |")
            
            # Add legend with detailed mathematical definitions
            lines.append("")
            lines.append("### Legend")
            lines.append("")
            
            # Build legend from displayed metrics with mathematical definitions
            legend_items = []
            for key in all_metric_keys:
                definition = _metric_definition(key)
                legend_items.append(f"- {definition}")
            
            lines.extend(legend_items)
            lines.append("")
        
        # Write file
        safe_filename = surface_name.replace(".", "_")
        output_file = surface_dir / f"{safe_filename}.md"
        try:
            output_file.write_text("\n".join(lines))
        except Exception as e:
            import sys
            print(f"ERROR: Failed to write {output_file}: {e}", file=sys.stderr)
            raise
    
    return surface_names


def write_surface_leaderboard_index(surface_names: list[str], docs_dir: Path) -> None:
    """
    No longer creates an index file - leaderboards are in docs/leaderboards/ directory.
    This function is kept for API compatibility but does nothing.
    """
    pass


def update_database(
    repo_root: Path,
    submissions_root: Path | None = None,
    docs_dir: Path | None = None,
    cases_root: Path | None = None,
    plasma_surfaces_dir: Path | None = None,
) -> None:
    """
    High-level entry point to rebuild the leaderboard.

    It does several things:
      1. Scans submissions_root for results.json files
      2. Aggregates data from submissions (in-memory)
      3. Writes docs/leaderboards/ (per-surface leaderboards)
      4. Writes docs/leaderboard.json for reference

    Parameters
    ----------
    repo_root:
        Root of the git repo (e.g. Path.cwd() when called from repo root).
    submissions_root:
        Directory containing per-method submissions. Defaults to repo_root / "submissions".
    docs_dir:
        Directory where docs/leaderboards/ leaderboards and leaderboard.json are written. Defaults to repo_root / "docs".
    cases_root:
        Directory containing case.yaml files. Defaults to repo_root / "cases".
    plasma_surfaces_dir:
        Directory containing plasma surface files. Defaults to repo_root / "plasma_surfaces".
    """
    submissions_root = submissions_root or (repo_root / "submissions")
    docs_dir = docs_dir or (repo_root / "docs")
    cases_root = cases_root or (repo_root / "cases")
    plasma_surfaces_dir = plasma_surfaces_dir or (repo_root / "plasma_surfaces")

    docs_dir.mkdir(parents=True, exist_ok=True)

    # Build in-memory data structures from submissions
    methods = build_methods_json(submissions_root=submissions_root, repo_root=repo_root)
    # cases = build_cases_json(methods)
    leaderboard = build_leaderboard_json(methods)

    # Write leaderboard.json for reference
    # methods.json and cases.json are intermediate and not needed on disk
    # Ensure leaderboard always has the expected structure
    if not isinstance(leaderboard, dict):
        leaderboard = {"entries": []}
    if "entries" not in leaderboard:
        leaderboard["entries"] = []
    
    leaderboard_file = docs_dir / "leaderboard.json"
    leaderboard_json = json.dumps(leaderboard, indent=2)
    leaderboard_file.write_text(leaderboard_json)
    
    # Verify the file was written correctly
    import sys
    if not leaderboard_file.exists() or leaderboard_file.stat().st_size == 0:
        print("ERROR: leaderboard.json was not written correctly!", file=sys.stderr)
        sys.exit(1)

    # Build and write per-surface leaderboards
    surface_leaderboards = build_surface_leaderboards(
        leaderboard, submissions_root, plasma_surfaces_dir
    )
    
    import sys
    print(f"Surface leaderboards built: {sorted(surface_leaderboards.keys())}", file=sys.stderr)
    for surface, data in surface_leaderboards.items():
        entries_count = len(data.get('entries', []))
        print(f"  {surface}: {entries_count} entries", file=sys.stderr)
    
    surface_names = write_surface_leaderboards(
        surface_leaderboards, docs_dir=docs_dir, repo_root=repo_root
    )
    write_surface_leaderboard_index(surface_names, docs_dir=docs_dir)
    
    print(f"Generated {len(surface_names)} surface leaderboard files: {sorted(surface_names)}", file=sys.stderr)

