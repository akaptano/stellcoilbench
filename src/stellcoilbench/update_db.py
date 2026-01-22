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
        "final_arclength_variation": "Var(l_i)",
        
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


def _format_date(date_str: str) -> str:
    """
    Format date from ISO format (YYYY-MM-DD) to DD/MM/YY format.
    
    Examples:
    - "2025-12-01" -> "01/12/25"
    - "_unknown_" -> "_unknown_"
    """
    if date_str is None:
        return "_unknown_"
    if not date_str or date_str == "_unknown_":
        return date_str
    
    # Handle ISO format with or without time component
    if "T" in date_str:
        date_str = date_str.split("T")[0]
    
    try:
        # Parse YYYY-MM-DD format
        parts = date_str.split("-")
        if len(parts) == 3:
            year = parts[0]
            month = parts[1]
            day = parts[2]
            # Convert to DD/MM/YY
            return f"{day}/{month}/{year[2:]}"
    except (IndexError, AttributeError):
        pass
    
    # Return as-is if parsing fails
    return date_str


def _shorthand_to_math(shorthand: str) -> str:
    """
    Convert metric shorthand to RST math mode format.
    
    Examples:
    - "min(d_cc)" -> ":math:`\\min(d_{cc})`"
    - "f_B" -> ":math:`f_B`"
    - "κ̄" -> ":math:`\\bar{\\kappa}`"
    - "n" -> ":math:`n`"
    """
    import re
    
    # If it's already a simple variable or Greek letter, wrap it
    if shorthand in ["n", "N", "L", "t"]:
        return f":math:`{shorthand}`"
    
    # Handle special Unicode characters
    unicode_map = {
        "κ̄": r":math:`\bar{\kappa}`",
        "F̄": r":math:`\bar{F}`",
        "τ̄": r":math:`\bar{\tau}`",
        "avg⟨Bn⟩/⟨B⟩": r":math:`\text{avg}\langle B_n \rangle / \langle B \rangle`",
        "max⟨Bn⟩/⟨B⟩": r":math:`\max(\langle B_n \rangle / \langle B \rangle)`",
        "Var(l_i)": r":math:`\mathrm{Var}(l_i)`",
    }
    if shorthand in unicode_map:
        return unicode_map[shorthand]
    
    # Handle function calls like "min(d_cc)", "max(κ)", "max(F)", "max(τ)"
    func_match = re.match(r'(\w+)\(([^)]+)\)', shorthand)
    if func_match:
        func_name = func_match.group(1)
        arg = func_match.group(2)
        # Convert underscores to subscripts
        arg_math = arg.replace("_", "_{").replace("}", "}") + ("}" if "{" in arg.replace("_", "_{") else "")
        # Handle special cases
        if arg == "κ":
            arg_math = r"\kappa"
        elif arg == "F":
            arg_math = r"F"
        elif arg == "τ":
            arg_math = r"\tau"
        elif arg == "d_cc":
            arg_math = r"d_{cc}"
        elif arg == "d_cs":
            arg_math = r"d_{cs}"
        else:
            # Default: convert underscores to subscripts
            parts = arg.split("_")
            if len(parts) == 2:
                arg_math = f"{parts[0]}_{{{parts[1]}}}"
            else:
                arg_math = arg.replace("_", "_{") + "}"
        
        func_math = func_name if func_name in ["min", "max"] else func_name
        return f":math:`\\{func_math}({arg_math})`"
    
    # Handle simple variable names with underscores (e.g., "d_cc", "d_cs")
    if "_" in shorthand:
        parts = shorthand.split("_")
        if len(parts) == 2:
            return f":math:`{parts[0]}_{{{parts[1]}}}`"
        else:
            # Multiple underscores - convert all to subscripts
            result = parts[0]
            for part in parts[1:]:
                result += f"_{{{part}}}"
            return f":math:`{result}`"
    
    # Default: wrap in math mode
    return f":math:`{shorthand}`"


def _metric_definition(metric_name: str) -> str:
    """
    Get detailed mathematical definition for a metric.
    
    Returns a string with LaTeX-style mathematical notation describing the metric.
    Format: symbol = expression - description
    """
    definitions = {
        # B-field related
        "final_normalized_squared_flux": r"Normalized squared flux error $f_B = \frac{1}{|S|} \int_{S} \left(\frac{\mathbf{B} \cdot \mathbf{n}}{|\mathbf{B}|}\right)^2 dS$ on plasma surface (dimensionless)",
        "max_BdotN_over_B": r"Maximum normalized normal field component $\max\left(\frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}\right)$ (dimensionless)",
        "avg_BdotN_over_B": r"Average normalized normal field component $\frac{\int_{S} |\mathbf{B} \cdot \mathbf{n}| dS}{\int_{S} |\mathbf{B}| dS}$ (dimensionless)",
        
        # Curvature
        "final_average_curvature": r"Mean curvature $\bar{\kappa} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i$ over all coils, where $\kappa_i = |\mathbf{r}''(s)|$ ($\text{m}^{-1}$)",
        "final_max_curvature": r"Maximum curvature $\max(\kappa)$ across all coils ($\text{m}^{-1}$)",
        "final_mean_squared_curvature": r"Mean squared curvature $\text{MSC} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i^2$ ($\text{m}^{-2}$)",
        
        # Separations
        "final_min_cs_separation": r"Minimum coil-to-surface distance $\min(d_{cs})$ ($\text{m}$)",
        "final_min_cc_separation": r"Minimum coil-to-coil distance $\min(d_{cc})$ ($\text{m}$)",
        "final_cs_separation": r"Average coil-to-surface separation $d_{cs}$ ($\text{m}$)",
        "final_cc_separation": r"Average coil-to-coil separation $d_{cc}$ ($\text{m}$)",
        
        # Length
        "final_total_length": r"Total length $L = \sum_{i=1}^{N} \int_{0}^{L_i} ds$ of all coils ($\text{m}$)",
        
        # Forces/Torques
        "final_max_max_coil_force": r"Maximum force magnitude $\max(F)$ across all coils ($\text{N}/\text{m}$)",
        "final_avg_max_coil_force": r"Average of maximum force $\bar{F} = \frac{1}{N} \sum_{i=1}^{N} \max(|\mathbf{F}_i|)$ per coil ($\text{N}/\text{m}$)",
        "final_max_max_coil_torque": r"Maximum torque magnitude $\max(\tau)$ across all coils ($\text{N}$)",
        "final_avg_max_coil_torque": r"Average of maximum torque $\bar{\tau} = \frac{1}{N} \sum_{i=1}^{N} \max(|\boldsymbol{\tau}_i|)$ per coil ($\text{N}$)",
        
        # Time
        "optimization_time": r"Total optimization time $t$ ($\text{s}$)",
        
        # Linking number
        "final_linking_number": r"Linking number $\text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot (d\mathbf{r}_i \times d\mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^3}$ between coil pairs (dimensionless)",
        
        # Arclength variation
        "final_arclength_variation": r"Variance of incremental arclength $J = \text{Var}(l_i)$ where $l_i$ is the average incremental arclength on interval $I_i$ from a partition $\{I_i\}_{i=1}^L$ of $[0,1]$ ($\text{m}^2$)",
        
        # Coil parameters
        "coil_order": r"Fourier order $n$ of coil representation: $\mathbf{r}(\phi) = \sum_{m=-n}^{n} \mathbf{c}_m e^{im\phi}$ (dimensionless)",
        "num_coils": r"Number of base coils $N$ (before applying stellarator symmetry) (dimensionless)",
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
        
        # Try to find "submissions" in path
        if "submissions" in path_parts:
            submissions_idx = path_parts.index("submissions")
            if submissions_idx + 1 < len(path_parts):
                surface = path_parts[submissions_idx + 1]
            if submissions_idx + 2 < len(path_parts):
                user = path_parts[submissions_idx + 2]
        else:
            # For test cases or non-standard paths, extract from relative path structure
            # Path format: submissions_root/surface/user/timestamp/results.json
            # Calculate relative to submissions_root
            try:
                rel_path = path.relative_to(submissions_root)
                rel_parts = rel_path.parts
                if len(rel_parts) >= 3:
                    surface = rel_parts[0]  # First part after submissions_root
                    user = rel_parts[1]     # Second part
            except ValueError:
                # If relative path calculation fails, try absolute path structure
                # Assume format: .../surface/user/timestamp/results.json
                if len(path_parts) >= 4:
                    # results.json is last, timestamp is second to last, user is third to last, surface is fourth to last
                    surface = path_parts[-4]
                    user = path_parts[-3]
        
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
                
                # If run_date is missing or all the same, try to extract from zip filename
                # Zip filename format: MM-DD-YYYY_HH-MM-SS.zip
                if not meta.get("run_date") or meta.get("run_date") == "2025-12-01T00:00:00":
                    zip_stem = zip_path.stem  # e.g., "12-01-2025_01-51"
                    # Try to parse timestamp from filename
                    import re
                    match = re.match(r'(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})', zip_stem)
                    if match:
                        month, day, year, hour, minute = match.groups()
                        # Convert to ISO format
                        meta["run_date"] = f"{year}-{month}-{day}T{hour}:{minute}:00"
                
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
        # Threshold parameters - these are configuration, not results
        "flux_threshold",
        "cc_threshold",
        "cs_threshold",
        "msc_threshold",
        "curvature_threshold",
        "force_threshold",
        "torque_threshold",
        "arclength_variation",  # Exclude intermediate arclength variation, keep only final
        "arclength_variation_threshold",  # Exclude threshold parameter
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
        
        # Use HTML table with inline styles for smaller font
        lines.append('<table style="font-size: 0.85em;">')
        lines.append("<thead>")
        lines.append("<tr>")
        for col in header_cols:
            lines.append(f'<th style="font-size: 0.9em; padding: 4px 8px;">{col}</th>')
        lines.append("</tr>")
        lines.append("</thead>")
        lines.append("<tbody>")
        
        def _format_value(value: Any, metric_key: str = "") -> str:
            """Format a metric value in compact scientific notation."""
            # Special handling for linking number - use integer format
            if metric_key == "final_linking_number":
                if isinstance(value, (float, int)):
                    return str(int(round(value)))
                return str(value)
            # All other numeric values use ultra-compact scientific notation
            if isinstance(value, (float, int)):
                val = float(value)
                if abs(val) < 1e-100:
                    return "0"
                # Use ultra-compact format: single digit, no + sign, no leading zero
                s = f"{val:.1e}"
                # Remove + sign for compactness
                s = s.replace("e+", "e")
                # Remove leading zero (e.g., "0.5e-2" -> ".5e-2")
                if s.startswith("0."):
                    s = "." + s[2:]
                elif s.startswith("-0."):
                    s = "-." + s[3:]
                # For very large numbers, use shorter format if possible
                if "e" in s:
                    parts = s.split("e")
                    if len(parts) == 2:
                        base, exp = parts[0], parts[1]
                        # Remove leading zero from exponent if present
                        if exp.startswith("0") and len(exp) > 1:
                            exp = exp[1:]
                        s = base + "e" + exp
                return s
            return str(value)
        
        # Write rows for each entry
        for e in entries:
            metrics = e.get("metrics", {})
            
            run_date = _format_date(e.get("run_date") or "_unknown_")
            
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
            
            lines.append("<tr>")
            for cell in row_parts:
                lines.append(f'<td style="font-size: 0.9em; padding: 4px 8px;">{cell}</td>')
            lines.append("</tr>")
        
        lines.append("</tbody>")
        lines.append("</table>")
        
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


def write_rst_leaderboard(
    leaderboard: Dict[str, Any],
    out_rst: Path,
    surface_leaderboards: Dict[str, Dict[str, Any]],
) -> None:
    """
    Write a comprehensive ReadTheDocs-friendly reStructuredText leaderboard with
    embedded tables for all surfaces.
    """
    entries = leaderboard.get("entries") or []
    surface_names = sorted(surface_leaderboards.keys())

    def _format_value(value: Any, metric_key: str = "") -> str:
        """Format metric values for display in RST tables."""
        integer_metrics = {"final_linking_number", "coil_order", "num_coils"}
        if metric_key in integer_metrics:
            if isinstance(value, (float, int)):
                return str(int(round(value)))
            return str(value)
        if isinstance(value, (float, int)):
            # Use scientific notation with 2 significant digits
            return f"{float(value):.2e}"
        return str(value)

    def _get_metrics_for_surface(entries_for_surface: list[Dict[str, Any]]) -> list[str]:
        """Extract all unique metric keys from entries for a specific surface."""
        exclude_fields = {
            "score_primary",
            "initial_B_field",
            "final_B_field",
            "target_B_field",
            "flux_threshold",
            "cc_threshold",
            "cs_threshold",
            "msc_threshold",
            "curvature_threshold",
            "force_threshold",
            "torque_threshold",
            "arclength_variation",  # Exclude intermediate arclength variation, keep only final
            "arclength_variation_threshold",  # Exclude threshold parameter
        }
        all_keys = set()
        for entry in entries_for_surface:
            metrics = entry.get("metrics", {})
            for key in metrics.keys():
                if key not in exclude_fields:
                    all_keys.add(key)
        
        # Define the desired order: N, n, fB, avg(Bn/B), max(Bn/B), L, min(d_cc), min(d_cs), \bar{kappa}, MSC, \bar{F}, \bar{\tau}, max(F), max(\tau), LN, t
        desired_order = [
            "num_coils",                    # N
            "coil_order",                   # n
            "final_normalized_squared_flux", # fB
            "avg_BdotN_over_B",             # avg(Bn/B)
            "max_BdotN_over_B",             # max(Bn/B)
            "final_total_length",           # L
            "final_arclength_variation",    # Var(l_i)
            "final_min_cc_separation",      # min(d_cc)
            "final_min_cs_separation",      # min(d_cs)
            "final_average_curvature",      # \bar{kappa}
            "final_mean_squared_curvature",  # MSC
            "final_avg_max_coil_force",     # \bar{F}
            "final_avg_max_coil_torque",    # \bar{\tau}
            "final_max_max_coil_force",     # max(F)
            "final_max_max_coil_torque",    # max(\tau)
            "final_linking_number",         # LN
            "optimization_time",            # t
        ]
        
        # Build ordered list: first add metrics in desired order that exist, then add any others
        ordered_keys = []
        for key in desired_order:
            if key in all_keys:
                ordered_keys.append(key)
        
        # Add any remaining keys that weren't in the desired order
        remaining_keys = sorted(all_keys - set(ordered_keys))
        ordered_keys.extend(remaining_keys)
        
        return ordered_keys

    def _get_surface_display_name(surface_name: str) -> str:
        """Convert surface file name to a descriptive display name."""
        # Mapping of file names (with and without extensions) to display names
        name_map = {
            "LandremanPaul2021_QA": "Landreman-Paul QA",
            "input.LandremanPaul2021_QA": "Landreman-Paul QA",
            "circular_tokamak": "Circular Tokamak",
            "input.circular_tokamak": "Circular Tokamak",
            "W7-X_without_coil_ripple_beta0p05_d23p4_tm": "W7-X",
            "input.W7-X_without_coil_ripple_beta0p05_d23p4_tm": "W7-X",
            "HSX_QHS_mn1824_ns101": "HSX",
            "input.HSX_QHS_mn1824_ns101": "HSX",
            "cfqs_2b40": "CFQS",
            "input.cfqs_2b40": "CFQS",
            "rotating_ellipse": "Rotating Ellipse",
            "input.rotating_ellipse": "Rotating Ellipse",
            "c09r00_B_axis_half_tesla_PM4Stell.focus": "0.5 Tesla NCSX Design",
            "c09r00_B_axis_half_tesla_PM4Stell": "0.5 Tesla NCSX Design",
            "muse.focus": "MUSE",
            "muse": "MUSE",
        }
        
        # Check for exact match first
        if surface_name in name_map:
            return name_map[surface_name]
        
        # Check for partial matches (without extension)
        name_no_ext = surface_name.replace(".focus", "").replace("input.", "")
        if name_no_ext in name_map:
            return name_map[name_no_ext]
        
        # Check for partial matches in keys
        for key, display in name_map.items():
            key_base = key.replace(".focus", "").replace("input.", "")
            if key_base in surface_name or surface_name in key_base:
                return display
        
        # Fallback: clean up the name
        return surface_name.replace("input.", "").replace(".focus", "").replace("_", " ").title()

    # Collect all unique metrics across all surfaces for definitions
    all_metric_keys_set = set()
    for surface_name in surface_names:
        entries_for_surface = surface_leaderboards[surface_name].get("entries", [])
        if entries_for_surface:
            surface_metrics = _get_metrics_for_surface(entries_for_surface)
            all_metric_keys_set.update(surface_metrics)
    
    # Also check overall entries if available
    if entries:
        all_metric_keys_set.update(_get_all_metrics_from_entries(entries))
    
    all_metric_keys = sorted(all_metric_keys_set)
    if "final_normalized_squared_flux" in all_metric_keys:
        all_metric_keys.remove("final_normalized_squared_flux")
        all_metric_keys.insert(0, "final_normalized_squared_flux")

    lines = [
        "StellCoilBench Leaderboard",
        "===========================",
        "",
        "The StellCoilBench leaderboard provides a comprehensive comparison of coil optimization",
        "methods across different plasma surfaces. Each submission is evaluated using standardized",
        "metrics that measure both the quality of the magnetic field produced and the engineering",
        "feasibility of the coil designs.",
        "",
        ".. note::",
        "   This page is automatically regenerated by CI after each successful submission.",
        "   For local development, run ``stellcoilbench update-db`` to refresh the leaderboard.",
        "",
    ]

    # Metric definitions (shown once at the top)
    if all_metric_keys:
        lines.append("Metric Definitions")
        lines.append("==================")
        lines.append("")
        lines.append("The following metrics are used to evaluate coil optimization submissions:")
        lines.append("")
        
        # Group metrics logically
        import re
        field_quality = []
        coil_geometry = []
        separations = []
        forces_torques = []
        topology = []
        performance = []
        config = []
        
        for key in all_metric_keys:
            definition = _metric_definition(key)
            shorthand = _metric_shorthand(key)
            
            # Convert all LaTeX math ($...$) to RST math (:math:`...`)
            formatted_def = re.sub(r'\$([^$]+)\$', r':math:`\1`', definition)
            
            if "flux" in key.lower() or "BdotN" in key or "B" in key:
                field_quality.append((shorthand, formatted_def))
            elif "curvature" in key.lower() or "length" in key.lower() or "arclength" in key.lower() or key in ["coil_order", "num_coils"]:
                coil_geometry.append((shorthand, formatted_def))
            elif "separation" in key.lower() or "distance" in key.lower():
                separations.append((shorthand, formatted_def))
            elif "force" in key.lower() or "torque" in key.lower():
                forces_torques.append((shorthand, formatted_def))
            elif "linking" in key.lower():
                topology.append((shorthand, formatted_def))
            elif "time" in key.lower():
                performance.append((shorthand, formatted_def))
            else:
                config.append((shorthand, formatted_def))
        
        if field_quality:
            lines.append("**Field Quality Metrics:**")
            lines.append("")
            for shorthand, rst_def in field_quality:
                lines.append(f"  - {rst_def}")
            lines.append("")
        
        if coil_geometry:
            lines.append("**Coil Geometry Metrics:**")
            lines.append("")
            for shorthand, rst_def in coil_geometry:
                lines.append(f"  - {rst_def}")
            lines.append("")
        
        if separations:
            lines.append("**Separation Metrics:**")
            lines.append("")
            for shorthand, rst_def in separations:
                lines.append(f"  - {rst_def}")
            lines.append("")
        
        if forces_torques:
            lines.append("**Force and Torque Metrics:**")
            lines.append("")
            for shorthand, rst_def in forces_torques:
                lines.append(f"  - {rst_def}")
            lines.append("")
        
        if topology:
            lines.append("**Topology Metrics:**")
            lines.append("")
            for shorthand, rst_def in topology:
                lines.append(f"  - {rst_def}")
            lines.append("")
        
        if performance:
            lines.append("**Performance Metrics:**")
            lines.append("")
            for shorthand, rst_def in performance:
                lines.append(f"  - {rst_def}")
            lines.append("")
        
        if config:
            lines.append("**Configuration Metrics:**")
            lines.append("")
            for shorthand, rst_def in config:
                lines.append(f"  - {rst_def}")
            lines.append("")
        
        # Add visualization link definitions
        lines.append("**Visualization Links:**")
        lines.append("")
        lines.append("  - :math:`i`: Link to 3D visualization plot showing :math:`B_N/|B|` error on plasma surface with initial (pre-optimization) coils")
        lines.append("  - :math:`f`: Link to 3D visualization plot showing :math:`B_N/|B|` error on plasma surface with final (optimized) coils")
        lines.append("")

    lines.append("Surface-Specific Leaderboards")
    lines.append("===============================")
    lines.append("")
    lines.append("Each plasma surface presents unique challenges for coil optimization. The following")
    lines.append("tables show detailed results for each surface, allowing for direct comparison")
    lines.append("of methods on specific configurations.")
    lines.append("")
    
    if not surface_names:
        lines.append("No surface leaderboards generated yet.")
        lines.append("")
    else:
        for surface_name in surface_names:
            display_name = _get_surface_display_name(surface_name)
            # Create a proper RST anchor
            anchor = surface_name.replace(".", "-").replace("_", "-").lower()
            lines.append(f".. _{anchor}:")
            lines.append("")
            lines.append(f"{display_name}")
            lines.append("^" * len(display_name))
            lines.append("")
            lines.append(f"**Surface file:** ``{surface_name}``")
            lines.append("")
            
            # Add surface description if available
            entries_for_surface = surface_leaderboards[surface_name].get("entries", [])
            if entries_for_surface:
                # Extract surface info from first entry if available
                first_entry = entries_for_surface[0]
                metrics = first_entry.get("metrics", {})
                num_coils = metrics.get("num_coils", "N/A")
                coil_order = metrics.get("coil_order", "N/A")
                lines.append(f"This surface has {len(entries_for_surface)} submission(s).")
                lines.append(f"Typical configuration: {int(coil_order) if isinstance(coil_order, (int, float)) else coil_order} Fourier order, {int(num_coils) if isinstance(num_coils, (int, float)) else num_coils} base coils.")
                lines.append("")
            
            if not entries_for_surface:
                lines.append("No submissions found for this surface.")
                lines.append("")
                lines.append("Submit results using cases that reference this surface to appear on this leaderboard.")
                lines.append("")
                continue

            surface_metric_keys = _get_metrics_for_surface(entries_for_surface)
            # Build header columns: metrics first, then Date, User, IC, # at the end
            surface_header_cols = []
            # Wrap metric shorthands in math mode for table headers
            for key in surface_metric_keys:
                shorthand = _metric_shorthand(key)
                # Convert shorthand to math mode (e.g., "min(d_cc)" -> ":math:`\min(d_{cc})`")
                math_shorthand = _shorthand_to_math(shorthand)
                surface_header_cols.append(math_shorthand)
            # Add Date, User, i, f at the end
            surface_header_cols.extend(["Date", "User", "i", "f"])

            # Use list-table for surface leaderboard
            lines.append(f".. list-table:: {display_name} Leaderboard")
            lines.append("   :header-rows: 1")
            lines.append("   :widths: auto")
            lines.append("")
            
            # Header row - each column on separate line
            lines.append("   * - " + surface_header_cols[0])
            for col in surface_header_cols[1:]:
                lines.append("     - " + col)
            
            # Data rows
            for entry in entries_for_surface:
                metrics = entry.get("metrics", {})
                run_date = _format_date(entry.get("run_date", "_unknown_"))
                
                # Find PDF paths for this entry and make initial (i) and final (f) coil visualization links
                rank_num = str(entry.get("rank", "-"))
                entry_path = entry.get("path", "")
                i_link = "—"  # Initial coils link - show dash if PDF doesn't exist
                f_link = rank_num  # Final coils link - show rank number
                if entry_path:
                    # Path format: submissions/{surface}/{user}/{timestamp}.zip
                    # entry_path is relative to repo root
                    path_obj = Path(entry_path)
                    if path_obj.suffix == ".zip":
                        # PDF should be in the same directory as the zip file
                        # Use standard PDF name (files are named bn_error_3d_plot.pdf without timestamp)
                        # path_obj.parent gives us submissions/{surface}/{user}/
                        pdf_path = path_obj.parent / "bn_error_3d_plot.pdf"
                        # Initial coils PDF
                        pdf_path_initial = path_obj.parent / "bn_error_3d_plot_initial.pdf"
                        # Infer repo root from out_rst path (docs/leaderboard.rst -> repo_root)
                        repo_root = out_rst.parent.parent
                        # Convert to absolute paths for existence check
                        full_pdf_path = (repo_root / pdf_path).resolve()
                        full_pdf_path_initial = (repo_root / pdf_path_initial).resolve()
                        if full_pdf_path.exists():
                            # Create relative path from docs/_build/html/ to PDF
                            # HTML files are in docs/_build/html/, PDFs are in submissions/ (project root)
                            # Need to go up 3 levels: html -> _build -> docs -> project root, then into submissions
                            pdf_rel_path = Path("../../..") / pdf_path
                            f_link = f"`{rank_num} <{pdf_rel_path}>`__"
                        if full_pdf_path_initial.exists():
                            # Only show rank number in "i" column if initial PDF exists
                            pdf_rel_path_initial = Path("../../..") / pdf_path_initial
                            i_link = f"`{rank_num} <{pdf_rel_path_initial}>`__"
                
                # Build row: metrics first, then Date, User, i, f at the end
                row_parts = []
                for key in surface_metric_keys:
                    value = metrics.get(key)
                    formatted = _format_value(value, metric_key=key) if value is not None else "—"
                    row_parts.append(formatted)
                # Add Date, User, i, f at the end
                row_parts.extend([
                    run_date,
                    entry.get("contact", entry.get("method_name", "?"))[:15],
                    i_link,
                    f_link,
                ])
                
                # First column
                lines.append("   * - " + row_parts[0])
                # Remaining columns
                for val in row_parts[1:]:
                    lines.append("     - " + val)
            
            lines.append("")
            lines.append("")

    lines.extend([
        "",
        ".. note::",
        "   Last updated: run ``stellcoilbench update-db`` to refresh locally.",
        "",
    ])

    out_rst.parent.mkdir(parents=True, exist_ok=True)
    out_rst.write_text("\n".join(lines))








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
            # Threshold parameters - these are configuration, not results
            "flux_threshold",
            "cc_threshold",
            "cs_threshold",
            "msc_threshold",
            "curvature_threshold",
            "force_threshold",
            "torque_threshold",
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
            
            # Use HTML table with inline styles for smaller font
            lines.append('<table style="font-size: 0.85em;">')
            lines.append("<thead>")
            lines.append("<tr>")
            for col in header_cols:
                lines.append(f'<th style="font-size: 0.9em; padding: 4px 8px;">{col}</th>')
            lines.append("</tr>")
            lines.append("</thead>")
            lines.append("<tbody>")
            
            # Data rows
            for entry in entries:
                metrics = entry.get("metrics", {})
                
                run_date = _format_date(entry.get("run_date", "_unknown_"))
                
                row_parts = [
                    str(entry.get("rank", "-")),
                    entry.get('contact', entry.get('method_name', '?'))[:15],  # Truncate long names
                    run_date,
                ]
                
                # Add all metrics
                for key in all_metric_keys:
                    value = metrics.get(key)
                    row_parts.append(_format_value(value, metric_key=key) if value is not None else "—")
                
                lines.append("<tr>")
                for cell in row_parts:
                    lines.append(f'<td style="font-size: 0.9em; padding: 4px 8px;">{cell}</td>')
                lines.append("</tr>")
            
            lines.append("</tbody>")
            lines.append("</table>")
            
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
    
    # Write ReadTheDocs-friendly leaderboard (includes surface list)
    write_rst_leaderboard(leaderboard, docs_dir / "leaderboard.rst", surface_leaderboards)
    
    print(f"Generated {len(surface_names)} surface leaderboard files: {sorted(surface_names)}", file=sys.stderr)

