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
        "max_BdotN_over_B": "max(B_n)",
        "avg_BdotN_over_B": "B̄_n",
        "final_normalized_squared_flux": "f_B",
        "initial_B_field": "B0",
        "final_B_field": "Bf",
        "target_B_field": "Bt",
        
        # Curvature
        "final_average_curvature": "κ̄",
        "final_max_curvature": "κ_max",
        "final_mean_squared_curvature": "MSC",
        
        # Separations (d_cc and d_cs are already minimum distances)
        "final_min_cs_separation": "d_cs",
        "final_min_cc_separation": "d_cc",
        "final_cs_separation": "d_cs",
        "final_cc_separation": "d_cc",
        
        # Length
        "final_total_length": "L",
        "final_arclength_variation": "Var(l_i)",
        
        # Forces/Torques
        "final_max_max_coil_force": "F_max",
        "final_avg_max_coil_force": "F̄",
        "final_max_max_coil_torque": "τ_max",
        "final_avg_max_coil_torque": "τ̄",
        
        # Time
        "optimization_time": "t",
        
        # Linking number
        "final_linking_number": "LN",
        
        # Coil parameters
        "coil_order": "n",
        "num_coils": "N",
        
        # Fourier continuation
        "fourier_continuation_orders": "FC",
        
        # Quasisymmetry
        "quasisymmetry_average": "avg(QS)",
        
        # Fast Particle Tracing
        "loss_fraction": "LF",
        
        # Score (keep for sorting but don't display)
        "score_primary": "score",
    }
    
    return shorthand_map.get(metric_name, metric_name.replace("_", " "))


def _format_date(date_str: str) -> str:
    """
    Format date from ISO format (YYYY-MM-DD) to DD/MM/YY format.
    
    Examples:
    - "2025-12-01" -> "01/12/25"
    - "2026-01-21" -> "21/01/26"
    - "_unknown_" -> "_unknown_"
    """
    if date_str is None:
        return "_unknown_"
    if not date_str or date_str == "_unknown_":
        return date_str
    
    # Handle ISO format with or without time component
    if "T" in date_str:
        date_str = date_str.split("T")[0]
    
    # Check if already in "/" format - could be MM/DD/YY or DD/MM/YY
    if "/" in date_str:
        parts = date_str.split("/")
        if len(parts) == 3:
            first, second, year = parts
            # Pad components
            first = first.zfill(2)
            second = second.zfill(2)
            if len(year) == 4:
                year = year[2:]  # Convert YYYY to YY
            elif len(year) != 2:
                # Invalid format, try to parse as ISO instead
                pass
            else:
                # Detect format: if first part > 12, it must be DD/MM/YY (day can be > 12, month can't)
                # If second part > 12, it must be MM/DD/YY (second part is the day)
                # If both <= 12, we need to check: if first <= 12 and second <= 12, it's ambiguous
                # However, if the date is already in DD/MM/YY format (which is our target),
                # we should preserve it. Since we can't tell, we'll use a heuristic:
                # - If first > 12: definitely DD/MM/YY, keep as-is
                # - If second > 12: definitely MM/DD/YY, swap to DD/MM/YY
                # - If both <= 12: check if it looks like it's already DD/MM/YY by checking
                #   if it matches common DD/MM patterns (days 1-31, months 1-12)
                try:
                    first_int = int(first)
                    second_int = int(second)
                    if first_int > 12:
                        # Definitely DD/MM/YY format (day > 12)
                        day, month = first, second
                    elif second_int > 12:
                        # Definitely MM/DD/YY format (second part > 12 means it's the day)
                        day, month = second, first
                    else:
                        # Ambiguous: both <= 12
                        # Check original unpadded values to see if we can determine format
                        original_first = int(parts[0])  # Original unpadded first part
                        original_second = int(parts[1])  # Original unpadded second part
                        
                        # If original first > 12, it's definitely DD/MM/YY
                        if original_first > 12:
                            day, month = first, second
                        # If original second > 12, it's definitely MM/DD/YY
                        elif original_second > 12:
                            day, month = second, first
                        else:
                            # Both original parts <= 12 - truly ambiguous
                            # Since dates should come from ISO format (YYYY-MM-DD), if we see "/" format,
                            # it's likely from old data. We'll use a heuristic:
                            # - If first part could be a day > 12 (when unpadded), it's DD/MM/YY
                            # - Otherwise, assume MM/DD/YY and convert to DD/MM/YY
                            # 
                            # However, we need to be careful: if the date is already DD/MM/YY,
                            # we don't want to double-convert it. Since we can't tell for sure,
                            # we'll check: if first <= 12 AND second <= 12, and first could be
                            # a valid month (1-12) and second could be a valid day (1-31),
                            # assume MM/DD/YY and convert.
                            # 
                            # Actually, the safest approach: if both are <= 12, assume MM/DD/YY
                            # (since that's the US format more common in legacy data) and convert.
                            # But we already checked original_first > 12 above, so if we're here,
                            # original_first <= 12. So we should convert.
                            day, month = second, first
                    return f"{day}/{month}/{year}"
                except (ValueError, TypeError):
                    # If parsing fails, return as-is
                    return f"{first}/{second}/{year}"
    
    try:
        # Parse YYYY-MM-DD format
        parts = date_str.split("-")
        if len(parts) == 3:
            year = parts[0]
            month = parts[1]
            day = parts[2]
            # Convert to DD/MM/YY with zero-padding
            day = day.zfill(2)
            month = month.zfill(2)
            year = year[2:] if len(year) == 4 else year
            return f"{day}/{month}/{year}"
    except (IndexError, AttributeError):
        pass
    
    # Return as-is if parsing fails
    return date_str


def _shorthand_to_math(shorthand: str) -> str:
    r"""
    Convert metric shorthand to RST math mode format.
    
    Examples:
    - "d_cc" -> ":math:`d_{cc}`"
    - "F_max" -> r":math:`F_\text{max}`"
    - "B̄_n" -> r":math:`\bar{B}_n`"
    - "f_B" -> ":math:`f_B`"
    - "κ̄" -> r":math:`\bar{\kappa}`"
    - "n" -> ":math:`n`"
    """
    import re
    
    # If it's already a simple variable or Greek letter, wrap it
    if shorthand in ["n", "N", "L", "t"]:
        return f":math:`{shorthand}`"
    
    # Handle special Unicode characters and new formats
    unicode_map = {
        "κ̄": r":math:`\bar{\kappa}`",
        "F̄": r":math:`\bar{F}`",
        "τ̄": r":math:`\bar{\tau}`",
        "B̄_n": r":math:`\bar{B}_n`",
        "avg(B_n)": r":math:`\text{avg}(B_n)`",
        "max(B_n)": r":math:`\max(B_n)`",
        "Var(l_i)": r":math:`\mathrm{Var}(l_i)`",
        "FC": r":math:`\text{FC}`",  # Fourier continuation
        "F_max": r":math:`F_\text{max}`",
        "τ_max": r":math:`\tau_\text{max}`",
        "κ_max": r":math:`\kappa_\text{max}`",
    }
    if shorthand in unicode_map:
        return unicode_map[shorthand]
    
    # Handle function calls like "max(κ)", "max(B_n)" (d_cc, d_cs, F_max, τ_max, κ_max are now direct variables)
    func_match = re.match(r'(\w+)\(([^)]+)\)', shorthand)
    if func_match:
        func_name = func_match.group(1)
        arg = func_match.group(2)
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
        elif arg == "B_n":
            arg_math = r"B_n"
        else:
            # Default: convert underscores to subscripts
            parts = arg.split("_")
            if len(parts) == 2:
                arg_math = f"{parts[0]}_{{{parts[1]}}}"
            else:
                # Multiple underscores - convert all to subscripts properly
                result = parts[0]
                for part in parts[1:]:
                    result += f"_{{{part}}}"
                arg_math = result
        
        # Use LaTeX operators for min/max, \text{} for other functions
        if func_name == "min":
            func_math = "\\min"
        elif func_name == "max":
            func_math = "\\max"
        elif func_name == "avg":
            func_math = "\\text{avg}"
        else:
            func_math = func_name
        # Format the math expression - func_math already contains proper escaping
        return f":math:`{func_math}({arg_math})`"
    
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
    
    # Handle strings with spaces - wrap in \text{} for RST math mode
    if " " in shorthand:
        # Escape spaces by wrapping in \text{}
        escaped = shorthand.replace(" ", r"\ ")
        return f":math:`\\text{{{escaped}}}`"
    
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
        "max_BdotN_over_B": r"Maximum normalized normal field component $\max(B_n)$ where $B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}$ (dimensionless)",
        "avg_BdotN_over_B": r"Average normalized normal field component $\bar{B}_n$ where $B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}$ and $\bar{B}_n = \frac{\int_{S} |\mathbf{B} \cdot \mathbf{n}| dS}{\int_{S} |\mathbf{B}| dS}$ (dimensionless)",
        
        # Curvature
        "final_average_curvature": r"Mean curvature $\bar{\kappa} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i$ over all coils, where $\kappa_i = |\mathbf{r}''(s)|$ ($\text{m}^{-1}$)",
        "final_max_curvature": r"Maximum curvature $\kappa_\text{max}$ across all coils ($\text{m}^{-1}$)",
        "final_mean_squared_curvature": r"Mean squared curvature $\text{MSC} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i^2$ ($\text{m}^{-2}$)",
        
        # Separations (d_cc and d_cs are minimum distances)
        "final_min_cs_separation": r"Minimum coil-to-surface distance $d_{cs}$ ($\text{m}$)",
        "final_min_cc_separation": r"Minimum coil-to-coil distance $d_{cc}$ ($\text{m}$)",
        "final_cs_separation": r"Average coil-to-surface separation $d_{cs}$ ($\text{m}$)",
        "final_cc_separation": r"Average coil-to-coil separation $d_{cc}$ ($\text{m}$)",
        
        # Length
        "final_total_length": r"Total length $L = \sum_{i=1}^{N} \int_{0}^{L_i} ds$ of all coils ($\text{m}$)",
        
        # Forces/Torques
        "final_max_max_coil_force": r"Maximum force magnitude $F_\text{max}$ across all coils ($\text{N}/\text{m}$)",
        "final_avg_max_coil_force": r"Average of maximum force $\bar{F} = \frac{1}{N} \sum_{i=1}^{N} \max(|\mathbf{F}_i|)$ per coil ($\text{N}/\text{m}$)",
        "final_max_max_coil_torque": r"Maximum torque magnitude $\tau_\text{max}$ across all coils ($\text{N}$)",
        "final_avg_max_coil_torque": r"Average of maximum torque $\bar{\tau} = \frac{1}{N} \sum_{i=1}^{N} \max(|\boldsymbol{\tau}_i|)$ per coil ($\text{N}$)",
        
        # Time
        "optimization_time": r"Total optimization time $t$ ($\text{s}$)",
        
        # Linking number
        "final_linking_number": r"Linking number $\text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot (d\mathbf{r}_i \times d\mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^3}$ between coil pairs (dimensionless)",
        
        # Arclength variation
        "final_arclength_variation": r"Variance of incremental arclength $J = \text{Var}(l_i)$ where $l_i$ is the average incremental arclength on interval $I_i$ from a partition $\{I_i\}_{i=1}^L$ of $[0,1]$ ($\text{m}^2$)",
        
        # Coil parameters
        "coil_order": r"Fourier order $n$ of coil representation: $\mathbf{r}(\phi) = \mathbf{a}_0 + \sum_{m=1}^{n} \left[\mathbf{a}_m \cos(m\phi) + \mathbf{b}_m \sin(m\phi)\right]$ (dimensionless)",
        "num_coils": r"Number of base coils $N$ (before applying stellarator symmetry) (dimensionless)",
        
        # Fourier continuation
        "fourier_continuation_orders": r"**Fourier continuation (FC)**: Sequence of Fourier orders used in continuation method. The optimization starts with a low-order representation, converges, then extends the solution to higher orders using the previous solution as initial condition. This helps achieve convergence for complex problems. Format: comma-separated list of orders (e.g., \"4,6,8\" means optimization was performed at orders 4, 6, and 8 sequentially). If not used, the column shows \"—\".",
        
        # Quasisymmetry
        "quasisymmetry_average": r"Average two-term quasisymmetry error $\text{avg}(QS)$ computed from VMEC equilibrium. The two-term quasisymmetry error measures how well the magnetic field strength $|\mathbf{B}|$ is constant on flux surfaces by evaluating the ratio residual $QS = \frac{|\mathbf{B}|_{m,n}}{|\mathbf{B}|}$ where $(m,n)$ is the target helicity. Lower values indicate better quasisymmetry (dimensionless).",
        
        # Fast Particle Tracing (SIMPLE)
        "loss_fraction": r"Final particle loss fraction from SIMPLE fast particle tracing. The loss fraction is computed as $1 - f_c$ where $f_c$ is the confined fraction (sum of confined passing and trapped particles). Lower values indicate better particle confinement (dimensionless).",
    }
    
    return definitions.get(metric_name, metric_name.replace("_", " ").title())


def _metric_detailed_definition(metric_name: str) -> dict | None:
    """
    Get detailed mathematical definition for a metric in structured format matching
    the "Available objectives" page format.
    
    Returns a dict with:
    - 'title': Metric title with symbol (e.g., "Normalized Squared Flux Error (:math:`f_B`)")
    - 'description': Description text
    - 'math_forms': List of mathematical expressions (LaTeX strings for .. math:: blocks)
    - 'units': Units string
    - 'notes': Optional additional notes
    Returns None if metric doesn't have a detailed definition.
    """
    detailed_defs = {
        "final_normalized_squared_flux": {
            "title": "Normalized Squared Flux Error",
            "symbol": r":math:`f_B`",
            "description": "Measures the quality of the magnetic field on the plasma surface by quantifying how well the normal component of the magnetic field vanishes.",
            "math_forms": [r"f_B = \frac{1}{|S|} \int_{S} \left(\frac{\mathbf{B} \cdot \mathbf{n}}{|\mathbf{B}|}\right)^2 ds"],
            "where": r"where :math:`|S|` is the total surface area of the plasma surface :math:`S`.",
            "units": "dimensionless",
            "notes": "Lower values indicate better field quality (closer to zero normal field component)."
        },
        "avg_BdotN_over_B": {
            "title": "Average Normalized Normal Field Component",
            "symbol": r":math:`\bar{B}_n`",
            "description": "Average of the absolute value of the normalized normal field component across the plasma surface.",
            "math_forms": [
                r"B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}",
                r"\bar{B}_n = \frac{\int_{S} |\mathbf{B} \cdot \mathbf{n}| ds}{\int_{S} |\mathbf{B}| ds}"
            ],
            "units": "dimensionless",
            "notes": "Lower values indicate better field quality."
        },
        "max_BdotN_over_B": {
            "title": "Maximum Normalized Normal Field Component",
            "symbol": r":math:`\max(B_n)`",
            "description": "Maximum value of the normalized normal field component across the plasma surface.",
            "math_forms": [
                r"B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}",
                r"\max(B_n) = \max_{\mathbf{s} \in S} B_n(\mathbf{s})"
            ],
            "units": "dimensionless",
            "notes": "Lower values indicate better field quality."
        },
        "coil_order": {
            "title": "Fourier Order",
            "symbol": r":math:`n`",
            "description": "Order of the Fourier series representation used for coil curves.",
            "math_forms": [r"\mathbf{r}(\phi) = \mathbf{a}_0 + \sum_{m=1}^{n} \left[\mathbf{a}_m \cos(m\phi) + \mathbf{b}_m \sin(m\phi)\right]"],
            "where": r"where :math:`\mathbf{a}_0`, :math:`\mathbf{a}_m`, and :math:`\mathbf{b}_m` are Fourier coefficients and :math:`\phi` is the parameterization angle.",
            "units": "dimensionless",
            "notes": "Higher orders allow more complex coil shapes but increase the number of optimization variables."
        },
        "num_coils": {
            "title": "Number of Base Coils",
            "symbol": r":math:`N`",
            "description": "Number of base coils before applying stellarator symmetry.",
            "units": "dimensionless",
            "notes": "Typical values: 4, 6, 8, 12. More coils allow more complex field shaping but increase computational cost."
        },
        "final_total_length": {
            "title": "Total Length",
            "symbol": r":math:`L`",
            "description": "Total length of all coils.",
            "math_forms": [r"L = \sum_{i=1}^{N} \int_{C_i} d\ell_i"],
            "units": r":math:`\text{m}` (meters)",
            "notes": "Shorter coils are generally preferred for reduced material costs and improved manufacturability."
        },
        "final_average_curvature": {
            "title": "Mean Curvature",
            "symbol": r":math:`\bar{\kappa}`",
            "description": "Average curvature across all coils.",
            "math_forms": [
                r"\kappa_i(\ell_i) = \left|\mathbf{r}_i''(\ell_i)\right|",
                r"\bar{\kappa} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{L_i} \int_{C_i} \kappa_i(\ell_i) ~d\ell_i"
            ],
            "where": r"where :math:`\mathbf{r}_i(\ell_i)` is the parameterization of coil curve :math:`C_i` by arclength.",
            "units": r":math:`\text{m}^{-1}` (inverse meters)",
            "notes": "Lower curvature values indicate smoother coils that are easier to manufacture."
        },
        "final_max_curvature": {
            "title": "Maximum Curvature",
            "symbol": r":math:`\kappa_\text{max}`",
            "description": "Maximum curvature value across all coils.",
            "math_forms": [r"\kappa_\text{max} = \max_{i=1,\ldots,N} \max_{\ell_i \in [0,L_i]} \kappa_i(\ell_i)"],
            "units": r":math:`\text{m}^{-1}` (inverse meters)",
            "notes": "Lower values indicate coils without extreme curvature regions."
        },
        "final_mean_squared_curvature": {
            "title": "Mean Squared Curvature",
            "symbol": r":math:`\text{MSC}`",
            "description": "Mean squared curvature per coil, averaged across all coils.",
            "math_forms": [
                r"J = \frac{1}{L_i} \int_{C_i} \kappa_i^2(\ell_i) ~d\ell_i",
                r"\text{MSC} = \frac{1}{N} \sum_{i=1}^{N} J_i"
            ],
            "where": r"where :math:`L_i` is the total length of coil curve :math:`C_i`, :math:`\ell_i` is the arclength along the curve, and :math:`\kappa_i` is the curvature.",
            "units": r":math:`\text{m}^{-2}` (inverse meters squared)",
            "notes": "This provides a smoother penalty than maximum curvature, encouraging overall smoothness rather than just avoiding extreme values."
        },
        "final_arclength_variation": {
            "title": "Arclength Variation",
            "symbol": r":math:`J`",
            "description": "Variance of incremental arclength between coil segments.",
            "math_forms": [r"J = \text{Var}(l_i)"],
            "where": r"where :math:`l_i` is the average incremental arclength on interval :math:`I_i` from a partition :math:`\{I_i\}_{i=1}^L` of :math:`[0,1]`.",
            "units": r":math:`\text{m}^2` (meters squared)",
            "notes": "Lower values indicate more uniform spacing along coils, which is important for manufacturing and field quality."
        },
        "final_min_cc_separation": {
            "title": "Minimum Coil-to-Coil Distance",
            "symbol": r":math:`d_{cc}`",
            "description": "Minimum distance between any two coils.",
            "math_forms": [r"d_{cc} = \min_{i \neq j} \min_{\mathbf{r}_i \in C_i, \mathbf{r}_j \in C_j} \left\| \mathbf{r}_i - \mathbf{r}_j \right\|_2"],
            "units": r":math:`\text{m}` (meters)",
            "notes": "Ensures coils maintain a safe separation distance to prevent collisions."
        },
        "final_min_cs_separation": {
            "title": "Minimum Coil-to-Surface Distance",
            "symbol": r":math:`d_{cs}`",
            "description": "Minimum distance between any coil and the plasma surface.",
            "math_forms": [r"d_{cs} = \min_{i} \min_{\mathbf{r}_i \in C_i, \mathbf{s} \in S} \left\| \mathbf{r}_i - \mathbf{s} \right\|_2"],
            "units": r":math:`\text{m}` (meters)",
            "notes": "Ensures coils maintain a safe distance from the plasma surface."
        },
        "final_avg_max_coil_force": {
            "title": "Average of Maximum Force",
            "symbol": r":math:`\bar{F}`",
            "description": "Average across coils of the maximum force magnitude per coil.",
            "math_forms": [r"\bar{F} = \frac{1}{N} \sum_{i=1}^{N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{F}_i}{d\ell_i}\right|"],
            "where": r"where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length on coil curve :math:`C_i`.",
            "units": r":math:`\text{N}/\text{m}` (Newtons per meter)",
            "notes": "Lower values indicate coils that are easier to support mechanically."
        },
        "final_max_max_coil_force": {
            "title": "Maximum Force Magnitude",
            "symbol": r":math:`F_\text{max}`",
            "description": "Maximum force magnitude across all coils.",
            "math_forms": [r"F_\text{max} = \max_{i=1,\ldots,N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{F}_i}{d\ell_i}\right|"],
            "units": r":math:`\text{N}/\text{m}` (Newtons per meter)",
            "notes": "High forces indicate coils that may be difficult to support mechanically."
        },
        "final_avg_max_coil_torque": {
            "title": "Average of Maximum Torque",
            "symbol": r":math:`\bar{\tau}`",
            "description": "Average across coils of the maximum torque magnitude per coil.",
            "math_forms": [r"\bar{\tau} = \frac{1}{N} \sum_{i=1}^{N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{T}_i}{d\ell_i}\right|"],
            "where": r"where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length on coil curve :math:`C_i`.",
            "units": r":math:`\text{N}` (Newtons)",
            "notes": "Lower values indicate coils with reduced rotational forces that must be resisted by supports."
        },
        "final_max_max_coil_torque": {
            "title": "Maximum Torque Magnitude",
            "symbol": r":math:`\tau_\text{max}`",
            "description": "Maximum torque magnitude across all coils.",
            "math_forms": [r"\tau_\text{max} = \max_{i=1,\ldots,N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{T}_i}{d\ell_i}\right|"],
            "units": r":math:`\text{N}` (Newtons)",
            "notes": "High torques can lead to mechanical instability."
        },
        "final_linking_number": {
            "title": "Linking Number",
            "symbol": r":math:`\text{LN}`",
            "description": "Topological measure of how coils are linked together.",
            "math_forms": [r"\text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{\left(\mathbf{r}_i - \mathbf{r}_j\right) \cdot \left(d\mathbf{r}_i \times d\mathbf{r}_j\right)}{\left|\mathbf{r}_i - \mathbf{r}_j\right|^3}"],
            "units": "dimensionless",
            "notes": "This metric ensures coils maintain their topological structure during optimization."
        },
        "optimization_time": {
            "title": "Total Optimization Time",
            "symbol": r":math:`t`",
            "description": "Total time required to complete the optimization.",
            "units": r":math:`\text{s}` (seconds)",
            "notes": "Lower values indicate more efficient optimization algorithms or faster convergence."
        },
        "quasisymmetry_average": {
            "title": "Average Quasisymmetry Error",
            "symbol": r":math:`\text{avg}(QS)`",
            "description": "Average two-term quasisymmetry error computed from VMEC equilibrium.",
            "math_forms": [r"QS = \frac{|\mathbf{B}|_{m,n}}{|\mathbf{B}|}"],
            "where": r"The two-term quasisymmetry error measures how well the magnetic field strength :math:`|\mathbf{B}|` is constant on flux surfaces by evaluating the ratio residual where :math:`(m,n)` is the target helicity.",
            "units": "dimensionless",
            "notes": "Lower values indicate better quasisymmetry, which is important for particle confinement in stellarators."
        },
        "loss_fraction": {
            "title": "Loss Fraction",
            "symbol": r":math:`\text{LF}`",
            "description": "Final particle loss fraction from SIMPLE fast particle tracing.",
            "math_forms": [r"\text{LF} = 1 - f_c"],
            "where": r"where :math:`f_c` is the confined fraction (sum of confined passing and trapped particles).",
            "units": "dimensionless",
            "notes": "Lower values indicate better particle confinement. A value of 0 means all particles are confined, while a value of 1 means all particles are lost. This metric is computed by the SIMPLE code using Monte Carlo particle tracing."
        },
        "fourier_continuation_orders": {
            "title": "Fourier Continuation (FC)",
            "description": "Sequence of Fourier orders used in continuation method. The optimization starts with a low-order representation, converges, then extends the solution to higher orders using the previous solution as initial condition. This helps achieve convergence for complex problems.",
            "notes": 'Format: comma-separated list of orders (e.g., "4,6,8" means optimization was performed at orders 4, 6, and 8 sequentially). If not used, the column shows "—".'
        }
    }
    
    return detailed_defs.get(metric_name)


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
    import re  # Import at top of function to avoid UnboundLocalError
    
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
        # Current structure: submissions_root/surface_name/user/timestamp/results.json
        # Where surface_name is the plasma surface name without extension
        path_parts = path.parts
        surface = "unknown"
        user = "unknown"
        
        # Try to find "submissions" in path
        if "submissions" in path_parts:
            submissions_idx = path_parts.index("submissions")
            parts_after_submissions = path_parts[submissions_idx + 1:]
            
            # Current structure: submissions/surface_name/user/timestamp/results.json
            if len(parts_after_submissions) >= 3:
                # Structure: surface_name/user/timestamp/file
                surface = parts_after_submissions[0]
                user = parts_after_submissions[1]
            elif len(parts_after_submissions) >= 2:
                # Could be surface_name/user or user/timestamp
                # Check if second part looks like a timestamp
                timestamp_pattern = r'\d{2}-\d{2}-\d{4}[\d_-]*'
                second_part = parts_after_submissions[1] if len(parts_after_submissions) > 1 else ""
                if re.search(timestamp_pattern, second_part):
                    # Structure: user/timestamp (legacy format without surface)
                    user = parts_after_submissions[0]
                else:
                    # Structure: surface_name/user
                    surface = parts_after_submissions[0]
                    user = parts_after_submissions[1]
            elif len(parts_after_submissions) >= 1:
                # Just user (legacy)
                user = parts_after_submissions[0]
        else:
            # For test cases or non-standard paths, extract from relative path structure
            # Path format: submissions_root/surface_name/user/timestamp/results.json
            try:
                rel_path = path.relative_to(submissions_root)
                rel_parts = rel_path.parts
                if len(rel_parts) >= 3:
                    # Structure: surface_name/user/timestamp/file
                    surface = rel_parts[0]
                    user = rel_parts[1]
                elif len(rel_parts) >= 2:
                    # Check if second part is a timestamp
                    timestamp_pattern = r'\d{2}-\d{2}-\d{4}[\d_-]*'
                    second_part = rel_parts[1] if len(rel_parts) > 1 else ""
                    if re.search(timestamp_pattern, second_part):
                        # Legacy: user/timestamp
                        user = rel_parts[0]
                    else:
                        # Structure: surface_name/user
                        surface = rel_parts[0]
                        user = rel_parts[1]
                elif len(rel_parts) >= 1:
                    user = rel_parts[0]
            except ValueError:
                # If relative path calculation fails, try absolute path structure
                timestamp_pattern = r'\d{2}-\d{2}-\d{4}[\d_-]*'
                if len(path_parts) >= 4:
                    # Check if second-to-last part looks like a timestamp
                    second_last = path_parts[-2] if len(path_parts) > 1 else ""
                    if re.search(timestamp_pattern, second_last):
                        # Structure: .../surface_name/user/timestamp/file
                        surface = path_parts[-4]
                        user = path_parts[-3]
                    else:
                        # Legacy: .../user/timestamp/file
                        user = path_parts[-3]
                elif len(path_parts) >= 3:
                    # Structure: .../surface_name/user/file or .../user/timestamp/file
                    # Check if second-to-last is timestamp
                    second_last = path_parts[-2] if len(path_parts) > 1 else ""
                    if re.search(timestamp_pattern, second_last):
                        user = path_parts[-3]
                    else:
                        surface = path_parts[-3]
                        user = path_parts[-2]
        
        # Extract surface name from case.yaml if available
        # Always try to read from case.yaml first (preferred method)
        case_yaml_path = path.parent / "case.yaml"
        if case_yaml_path.exists():
            # Try to read surface from case.yaml in the same directory
            try:
                import yaml
                case_data = yaml.safe_load(case_yaml_path.read_text())
                surface_file = case_data.get("surface_params", {}).get("surface", "")
                if surface_file:
                    surface_name = Path(surface_file).name
                    if surface_name.startswith("input."):
                        surface = surface_name[6:]
                    elif surface_name.startswith("wout."):
                        surface = surface_name[5:]
                    else:
                        surface = surface_name
            except Exception:
                pass
        
        # If still unknown and path is a zip file, try to read case.yaml from zip
        if surface == "unknown" and path.suffix == ".zip":
            import zipfile
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    if "case.yaml" in zf.namelist():
                        import yaml
                        case_content = zf.read("case.yaml").decode('utf-8')
                        case_data = yaml.safe_load(case_content)
                        surface_file = case_data.get("surface_params", {}).get("surface", "")
                        if surface_file:
                            surface_name = Path(surface_file).name
                            if surface_name.startswith("input."):
                                surface = surface_name[6:]
                            elif surface_name.startswith("wout."):
                                surface = surface_name[5:]
                            else:
                                surface = surface_name
            except Exception:
                pass
        
        # Use explicit method_version if present, otherwise fall back to dir name.
        # For zip files, check if it's "all_files.zip" (new structure) or timestamp-based (old structure)
        if path.suffix == ".zip":
            if path.name == "all_files.zip":
                # New structure: use parent directory name (timestamp)
                version = meta.get("method_version") or path.parent.name
            else:
                # Old structure: use zip filename (without .zip extension)
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
                # Current structure: submissions_root/surface/user/timestamp/all_files.zip
                path_parts = zip_path.parts
                surface = "unknown"
                user = "unknown"
                
                # Try to find surface and user from path
                if "submissions" in path_parts:
                    submissions_idx = path_parts.index("submissions")
                    parts_after_submissions = path_parts[submissions_idx + 1:]
                    
                    # Current structure: submissions/surface/user/timestamp/all_files.zip
                    if len(parts_after_submissions) >= 3:
                        # Structure: surface/user/timestamp/all_files.zip
                        surface = parts_after_submissions[0]
                        user = parts_after_submissions[1]
                    elif len(parts_after_submissions) >= 2:
                        # Could be surface/user (current) or user/timestamp (legacy)
                        timestamp_pattern = r'\d{2}-\d{2}-\d{4}[\d_-]*'
                        second_is_timestamp = bool(re.search(timestamp_pattern, parts_after_submissions[1])) if len(parts_after_submissions) > 1 else False
                        if second_is_timestamp:
                            # Legacy: user/timestamp
                            user = parts_after_submissions[0]
                        else:
                            # Current: surface/user
                            surface = parts_after_submissions[0]
                            user = parts_after_submissions[1]
                    elif len(parts_after_submissions) >= 1:
                        # Just user (legacy)
                        user = parts_after_submissions[0]
                else:
                    # Path is relative to submissions_root
                    try:
                        rel_path = zip_path.relative_to(submissions_root)
                        rel_parts = rel_path.parts
                        if len(rel_parts) >= 3:
                            # Current structure: surface/user/timestamp/all_files.zip
                            surface = rel_parts[0]
                            user = rel_parts[1]
                        elif len(rel_parts) >= 2:
                            # Check if second part is timestamp (legacy: user/timestamp)
                            timestamp_pattern = r'\d{2}-\d{2}-\d{4}[\d_-]*'
                            second_is_timestamp = bool(re.search(timestamp_pattern, rel_parts[1])) if len(rel_parts) > 1 else False
                            if second_is_timestamp:
                                # Legacy: user/timestamp
                                user = rel_parts[0]
                            else:
                                # Current: surface/user
                                surface = rel_parts[0]
                                user = rel_parts[1]
                        elif len(rel_parts) >= 1:
                            # Just user (legacy)
                            user = rel_parts[0]
                    except ValueError:
                        # If relative path calculation fails, try to extract from absolute path
                        # Path format: .../surface/user/timestamp/all_files.zip (current)
                        timestamp_pattern = r'\d{2}-\d{2}-\d{4}[\d_-]*'
                        if len(path_parts) >= 4:
                            # Check if second-to-last part looks like a timestamp
                            second_last = path_parts[-2] if len(path_parts) > 1 else ""
                            if re.search(timestamp_pattern, second_last):
                                # Current structure: .../surface/user/timestamp/all_files.zip
                                surface = path_parts[-4] if len(path_parts) >= 4 else "unknown"
                                user = path_parts[-3]
                            else:
                                # Legacy: .../user/timestamp/all_files.zip
                                user = path_parts[-3]
                        elif len(path_parts) >= 3:
                            # Current: .../surface/user/all_files.zip
                            surface = path_parts[-3]
                            user = path_parts[-2]
                
                # Extract surface name from case.yaml in zip if available
                if "case.yaml" in zf.namelist():
                    try:
                        import yaml
                        case_content = zf.read("case.yaml").decode('utf-8')
                        case_data = yaml.safe_load(case_content)
                        surface_file = case_data.get("surface_params", {}).get("surface", "")
                        if surface_file:
                            surface_name = Path(surface_file).name
                            if surface_name.startswith("input."):
                                surface = surface_name[6:]
                            elif surface_name.startswith("wout."):
                                surface = surface_name[5:]
                            else:
                                surface = surface_name
                    except Exception:
                        pass
                
                # Use directory name (parent of zip) as version for new structure, zip stem for old structure
                # For new structure (all_files.zip), parent is the timestamp directory
                # For old structure (timestamp.zip), use zip filename without extension
                if zip_path.name == "all_files.zip":
                    version = meta.get("method_version") or zip_path.parent.name
                else:
                    version = meta.get("method_version") or zip_path.stem
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
        
        # Handle legacy format where metrics are at top level (not in "metrics" key)
        if not metrics and "final_normalized_squared_flux" in data:
            # This is a legacy format - metrics are at top level
            # Extract metrics by excluding metadata fields and internal fields
            metadata_keys = {"metadata", "method_name", "method_version", "contact", "hardware", "notes", "run_date", "output_directory", "lagrange_multipliers"}
            metrics = {k: v for k, v in data.items() if k not in metadata_keys}
            # If metadata is missing, try to extract from top level
            if not meta:
                meta = {k: data.get(k) for k in ["method_name", "contact", "hardware", "notes", "run_date"] if k in data}
            
            # If still no metadata, try to extract from path
            if not meta.get("contact"):
                # Extract username from path: submissions/surface/user/timestamp/file
                path_parts = path.parts
                if "submissions" in path_parts:
                    submissions_idx = path_parts.index("submissions")
                    parts_after = path_parts[submissions_idx + 1:]
                    # Current structure: submissions/surface/user/timestamp/file
                    if len(parts_after) >= 2:
                        meta["contact"] = parts_after[1]  # Username is second part after submissions
                else:
                    # Try relative path
                    try:
                        rel_path = path.relative_to(submissions_root)
                        rel_parts = rel_path.parts
                        if len(rel_parts) >= 2:
                            meta["contact"] = rel_parts[1]  # Username is second part
                    except ValueError:
                        pass
            
            # Extract run_date from path timestamp if missing
            if not meta.get("run_date"):
                path_parts = path.parts
                # Look for timestamp pattern MM-DD-YYYY_HH-MM in path
                import re
                timestamp_pattern = r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2})'
                for part in path_parts:
                    match = re.search(timestamp_pattern, part)
                    if match:
                        timestamp_str = match.group(1)
                        # Convert MM-DD-YYYY_HH-MM to ISO format
                        month, day, year, hour, minute = timestamp_str.replace('_', '-').split('-')
                        meta["run_date"] = f"{year}-{month}-{day}T{hour}:{minute}:00"
                        break

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
            
            # Extract Fourier continuation information
            fourier_continuation = case_yaml_data.get("fourier_continuation", {})
            if fourier_continuation and fourier_continuation.get("enabled", False):
                orders = fourier_continuation.get("orders", [])
                if orders:
                    # Store as a string representation for display
                    # Note: This is intentionally a string, not a float
                    orders_str = ",".join(str(o) for o in orders)
                    metrics_numeric["fourier_continuation_orders"] = orders_str  # type: ignore
        
        # If num_coils or coil_order are still missing, try to extract from coils.json
        if "coil_order" not in metrics_numeric or "num_coils" not in metrics_numeric:
            coils_json_path = path.parent / "coils.json"
            if coils_json_path.exists():
                try:
                    from simsopt import load
                    coils = load(str(coils_json_path))
                    if coils and len(coils) > 0:
                        # Extract coil order from first coil
                        if "coil_order" not in metrics_numeric and hasattr(coils[0], "curve") and hasattr(coils[0].curve, "order"):
                            metrics_numeric["coil_order"] = float(coils[0].curve.order)
                        
                        # Extract number of base coils
                        # Total coils = base_coils * nfp * (stellsym + 1)
                        # We need nfp and stellsym to calculate base_coils
                        if "num_coils" not in metrics_numeric:
                            total_coils = len(coils)
                            nfp = 1  # Default assumption
                            stellsym = True  # Default assumption
                            
                            # Try to get surface info from case.yaml if available
                            surface_file = None
                            if case_yaml_data:
                                surface_file = case_yaml_data.get("surface_params", {}).get("surface", "")
                            
                            # If no case.yaml, try to extract surface name from path
                            # Path format: submissions/surface_name/user/timestamp/file
                            if not surface_file:
                                path_parts = path.parts
                                if "submissions" in path_parts:
                                    submissions_idx = path_parts.index("submissions")
                                    parts_after = path_parts[submissions_idx + 1:]
                                    if len(parts_after) >= 1:
                                        surface_name = parts_after[0]
                                        # Try common surface file patterns
                                        for pattern in [
                                            f"input.{surface_name}",
                                            f"wout.{surface_name}",
                                            surface_name,
                                        ]:
                                            surface_file = pattern
                                            break
                            
                            # Try to load surface file to get nfp and stellsym
                            if surface_file:
                                try:
                                    from simsopt.geo import SurfaceRZFourier
                                    surface_file_path = None
                                    # Try to find surface file
                                    for potential_path in [
                                        Path(surface_file),
                                        Path("plasma_surfaces") / surface_file,
                                        repo_root / "plasma_surfaces" / surface_file,
                                    ]:
                                        if potential_path.exists():
                                            surface_file_path = potential_path
                                            break
                                    
                                    if surface_file_path:
                                        # Load surface with minimal resolution for speed
                                        surface_file_lower = str(surface_file_path).lower()
                                        if "input" in surface_file_lower:
                                            surface = SurfaceRZFourier.from_vmec_input(
                                                str(surface_file_path), nphi=8, ntheta=8
                                            )
                                        elif "wout" in surface_file_lower:
                                            surface = SurfaceRZFourier.from_wout(
                                                str(surface_file_path), nphi=8, ntheta=8
                                            )
                                        else:
                                            surface = None
                                        
                                        if surface:
                                            nfp = surface.nfp
                                            stellsym = surface.stellsym
                                except Exception:
                                    pass  # Use defaults
                            
                            # Calculate base number of coils
                            # Formula: base_coils = total_coils / (nfp * (stellsym + 1))
                            symmetry_factor = nfp * (2 if stellsym else 1)
                            base_coils = total_coils // symmetry_factor
                            if base_coils > 0:
                                metrics_numeric["num_coils"] = float(base_coils)
                except Exception as e:
                    import sys
                    print(f"Warning: Failed to extract coil info from {coils_json_path}: {e}", file=sys.stderr)
        
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

        # Always prefer GitHub username from path over contact field in metadata
        # Path structure: submissions/surface/user/timestamp/file
        github_username = meta.get("contact", "")
        path_parts = path.parts
        if "submissions" in path_parts:
            submissions_idx = path_parts.index("submissions")
            parts_after = path_parts[submissions_idx + 1:]
            # Current structure: submissions/surface/user/timestamp/file
            if len(parts_after) >= 2:
                github_username = parts_after[1]  # Username is second part after submissions
        else:
            # Try relative path
            try:
                rel_path_obj = path.relative_to(submissions_root)
                rel_parts = rel_path_obj.parts
                if len(rel_parts) >= 2:
                    github_username = rel_parts[1]  # Username is second part
            except ValueError:
                pass

        if primary_score is None:
            skipped_no_score += 1
        
        methods[method_key] = {
            "method_name": meta.get("method_name", "UNKNOWN"),
            "method_version": meta.get("method_version", path.stem if path.suffix == ".zip" else path.parent.name),
            "contact": github_username,  # Use GitHub username from path, not metadata
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
    Entries without score_primary are filtered out.
    """
    entries = []

    for method_key, md in methods.items():
        metrics = md.get("metrics", {})
        path = md.get("path", "")
        
        # Check if score_primary exists in the dict (distinguish between missing and None)
        if "score_primary" in md:
            score_primary = md["score_primary"]
            # If explicitly None, filter out (don't use fallback)
            if score_primary is None:
                import sys
                print(f"Warning: Entry {path} has score_primary=None, skipping", file=sys.stderr)
                continue
        else:
            # score_primary key doesn't exist, try fallback
            score_primary = metrics.get("final_normalized_squared_flux")
            if score_primary is None or not isinstance(score_primary, (int, float)):
                # Skip entries with no score_primary or final_normalized_squared_flux
                import sys
                print(f"Warning: Entry {path} has no score_primary or final_normalized_squared_flux (metrics keys: {list(metrics.keys())[:5]}), skipping", file=sys.stderr)
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
    print(f"Leaderboard: {len(entries)} entries included", file=sys.stderr)

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
        "final_order",  # Exclude final_order, keep only fourier_continuation_orders (FC)
        "continuation_step",  # Exclude continuation_step, keep only fourier_continuation_orders (FC)
        "fourier_continuation",  # Exclude fourier_continuation, keep only fourier_continuation_orders (FC)
        "fourier_order",  # Exclude fourier_order, keep only fourier_continuation_orders (FC)
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
            # All other numeric values use ultra-compact scientific notation, wrapped in span for smaller font
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
                # Wrap in span for smaller font
                if "e" in s:
                    parts = s.split("e")
                    if len(parts) == 2:
                        base, exp = parts[0], parts[1]
                        # Remove leading zero from exponent if present
                        if exp.startswith("0") and len(exp) > 1:
                            exp = exp[1:]
                        s = base + "e" + exp
                # Return formatted number (markdown tables can use HTML if needed, but CSS handles styling)
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
        # Fourier continuation orders are stored as comma-separated string
        if metric_key == "fourier_continuation_orders":
            return str(value) if value else "—"
        if isinstance(value, (float, int)):
            # Use scientific notation with 1 significant digit
            # CSS will handle making numbers smaller (no HTML needed)
            return f"{float(value):.1e}"
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
            "final_order",  # Exclude final_order, keep only fourier_continuation_orders (FC)
            "continuation_step",  # Exclude continuation_step, keep only fourier_continuation_orders (FC)
            "fourier_continuation",  # Exclude fourier_continuation, keep only fourier_continuation_orders (FC)
            "fourier_order",  # Exclude fourier_order, keep only fourier_continuation_orders (FC)
        }
        all_keys = set()
        for entry in entries_for_surface:
            metrics = entry.get("metrics", {})
            for key in metrics.keys():
                if key not in exclude_fields:
                    all_keys.add(key)
        
        # Define the desired order: N, n, FC, fB, \bar{B_n}, max(B_n), L, d_cc, d_cs, \bar{kappa}, MSC, \bar{F}, \bar{\tau}, F_max, \tau_max, LN, t, avg(QS), LF
        desired_order = [
            "num_coils",                    # N
            "coil_order",                   # n
            "fourier_continuation_orders",  # FC
            "final_normalized_squared_flux", # fB
            "avg_BdotN_over_B",             # \bar{B_n}
            "max_BdotN_over_B",             # max(B_n)
            "final_total_length",           # L
            "final_arclength_variation",    # Var(l_i)
            "final_min_cc_separation",      # d_cc
            "final_min_cs_separation",      # d_cs
            "final_average_curvature",      # \bar{kappa}
            "final_mean_squared_curvature",  # MSC
            "final_avg_max_coil_force",     # \bar{F}
            "final_avg_max_coil_torque",    # \bar{\tau}
            "final_max_max_coil_force",     # F_max
            "final_max_max_coil_torque",    # \tau_max
            "final_linking_number",         # LN
            "optimization_time",            # t
            "quasisymmetry_average",        # avg(QS)
            "loss_fraction",                # LF
        ]
        
        # Build ordered list: first add metrics in desired order that exist, then add any others
        ordered_keys = []
        # Always include these columns even if no entries have them (show "—" when missing)
        always_include = ["num_coils", "coil_order", "fourier_continuation_orders"]
        for key in desired_order:
            if key in all_keys or key in always_include:
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

    # Create leaderboard subdirectory for nested files
    leaderboard_dir = out_rst.parent / "leaderboard"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Main leaderboard.rst file with toctree
    main_lines = [
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
        ".. toctree::",
        "   :maxdepth: 2",
        "   :caption: Leaderboard Contents",
        "",
        "   leaderboard/metric_definitions",
        "   leaderboard/surface_specific",
        "",
    ]
    
    # Metric definitions file
    metric_def_lines = [
        "Metric Definitions",
        "===================",
        "",
        "The following metrics are used to evaluate coil optimization submissions:",
        "",
        "Notation",
        "--------",
        "",
        "The following notation is used throughout the mathematical definitions:",
        "",
        r"- :math:`C_i` denotes coil curve :math:`i`",
        r"- :math:`S` denotes the plasma surface",
        r"- :math:`\mathbf{r}_i` denotes a point on coil curve :math:`C_i`",
        r"- :math:`\mathbf{s}` denotes a point on the plasma surface :math:`S`",
        r"- :math:`\ell_i` denotes arclength along coil curve :math:`C_i`",
        r"- :math:`L_i` denotes the total length of coil curve :math:`C_i`",
        r"- :math:`\kappa_i` denotes curvature along coil curve :math:`C_i`",
        r"- :math:`\frac{d\vec{F}_i}{d\ell_i}` denotes force per unit length on coil curve :math:`C_i`",
        r"- :math:`\frac{d\vec{T}_i}{d\ell_i}` denotes torque per unit length on coil curve :math:`C_i`",
        r"- :math:`N` denotes the number of coils",
        r"- :math:`d\ell_i` denotes the differential arclength element along coil curve :math:`C_i`",
        r"- :math:`ds` denotes the differential surface area element on the plasma surface :math:`S`",
        r"- :math:`\mathbf{B}` denotes the magnetic field vector",
        r"- :math:`\mathbf{n}` denotes the unit normal vector to the plasma surface",
        "",
    ]
    
    # Metric definitions (shown once at the top)
    if all_metric_keys:
        
        # Group metrics logically
        field_quality = []
        coil_geometry = []
        separations = []
        forces_torques = []
        topology = []
        performance = []
        particle_confinement = []
        config = []
        
        for key in all_metric_keys:
            detailed_def = _metric_detailed_definition(key)
            if detailed_def:
                if "flux" in key.lower() or "BdotN" in key or "B" in key:
                    field_quality.append((key, detailed_def))
                elif "curvature" in key.lower() or "length" in key.lower() or "arclength" in key.lower() or key in ["coil_order", "num_coils", "fourier_continuation_orders"]:
                    coil_geometry.append((key, detailed_def))
                elif "separation" in key.lower() or "distance" in key.lower():
                    separations.append((key, detailed_def))
                elif "force" in key.lower() or "torque" in key.lower():
                    forces_torques.append((key, detailed_def))
                elif "linking" in key.lower():
                    topology.append((key, detailed_def))
                elif "time" in key.lower():
                    performance.append((key, detailed_def))
                elif key in ["loss_fraction", "quasisymmetry_average"]:
                    particle_confinement.append((key, detailed_def))
                else:
                    config.append((key, detailed_def))
        
        def _format_metric_def(key: str, def_dict: dict) -> list[str]:
            """Format a detailed metric definition into RST lines."""
            lines = []
            symbol = def_dict.get("symbol", "")
            title = def_dict.get("title", key.replace("_", " ").title())
            if symbol:
                lines.append(f"**{title}** ({symbol})")
            else:
                lines.append(f"**{title}**")
            lines.append("   " + def_dict.get("description", ""))
            lines.append("   ")
            
            math_forms = def_dict.get("math_forms", [])
            if math_forms:
                lines.append("   Mathematical form:")
                lines.append("   ")
                for math_form in math_forms:
                    lines.append("   .. math::")
                    lines.append(f"      {math_form}")
                    lines.append("   ")
            
            where = def_dict.get("where")
            if where:
                lines.append(f"   {where}")
                lines.append("   ")
            
            units = def_dict.get("units", "")
            if units:
                lines.append(f"   Units: {units}")
                lines.append("   ")
            
            notes = def_dict.get("notes")
            if notes:
                lines.append(f"   {notes}")
            
            return lines
        
        if field_quality:
            metric_def_lines.append("Field Quality Metrics")
            metric_def_lines.append("-" * len("Field Quality Metrics"))
            metric_def_lines.append("")
            for key, detailed_def in field_quality:
                metric_def_lines.extend(_format_metric_def(key, detailed_def))
                metric_def_lines.append("")
        
        if coil_geometry:
            metric_def_lines.append("Coil Geometry Metrics")
            metric_def_lines.append("-" * len("Coil Geometry Metrics"))
            metric_def_lines.append("")
            for key, detailed_def in coil_geometry:
                metric_def_lines.extend(_format_metric_def(key, detailed_def))
                metric_def_lines.append("")
        
        if separations:
            metric_def_lines.append("Separation Metrics")
            metric_def_lines.append("-" * len("Separation Metrics"))
            metric_def_lines.append("")
            for key, detailed_def in separations:
                metric_def_lines.extend(_format_metric_def(key, detailed_def))
                metric_def_lines.append("")
        
        if forces_torques:
            metric_def_lines.append("Force and Torque Metrics")
            metric_def_lines.append("-" * len("Force and Torque Metrics"))
            metric_def_lines.append("")
            for key, detailed_def in forces_torques:
                metric_def_lines.extend(_format_metric_def(key, detailed_def))
                metric_def_lines.append("")
        
        if topology:
            metric_def_lines.append("Topology Metrics")
            metric_def_lines.append("-" * len("Topology Metrics"))
            metric_def_lines.append("")
            for key, detailed_def in topology:
                metric_def_lines.extend(_format_metric_def(key, detailed_def))
                metric_def_lines.append("")
        
        if performance:
            metric_def_lines.append("Performance Metrics")
            metric_def_lines.append("-" * len("Performance Metrics"))
            metric_def_lines.append("")
            for key, detailed_def in performance:
                metric_def_lines.extend(_format_metric_def(key, detailed_def))
                metric_def_lines.append("")
        
        if particle_confinement:
            metric_def_lines.append("Particle Confinement Metrics")
            metric_def_lines.append("-" * len("Particle Confinement Metrics"))
            metric_def_lines.append("")
            for key, detailed_def in particle_confinement:
                metric_def_lines.extend(_format_metric_def(key, detailed_def))
                metric_def_lines.append("")
        
        if config:
            metric_def_lines.append("Configuration Metrics")
            metric_def_lines.append("-" * len("Configuration Metrics"))
            metric_def_lines.append("")
            for key, detailed_def in config:
                metric_def_lines.extend(_format_metric_def(key, detailed_def))
                metric_def_lines.append("")
        
        # Add visualization link definitions
        metric_def_lines.append("Visualization Links")
        metric_def_lines.append("-" * len("Visualization Links"))
        metric_def_lines.append("")
        metric_def_lines.append("- :math:`i`: Link to 3D visualization plot showing :math:`B_N/|B|` error on plasma surface with initial (pre-optimization) coils")
        metric_def_lines.append("- :math:`f`: Link to 3D visualization plot showing :math:`B_N/|B|` error on plasma surface with final (optimized) coils")
        metric_def_lines.append("- **PP**: Link to Poincaré plot showing fieldline trajectories")
        metric_def_lines.append("- **BP**: Link to Boozer surface plot showing flux surfaces")
        metric_def_lines.append("- **QS**: Link to quasisymmetry error profile plot")
        metric_def_lines.append("- **iota**: Link to rotational transform (iota) profile plot")
        metric_def_lines.append("- **FPT**: Link to Fast Particle Tracing (SIMPLE) loss fraction plot")
        metric_def_lines.append("")

    # Surface-specific leaderboards file
    surface_specific_lines = [
        "Surface-Specific Leaderboards",
        "===============================",
        "",
        "Each plasma surface presents unique challenges for coil optimization. The following",
        "tables show detailed results for each surface, allowing for direct comparison",
        "of methods on specific configurations.",
        "",
        "Visualization Links",
        "--------------------",
        "",
        "The leaderboard tables include visualization links in the following columns:",
        "",
        "- :math:`i`: Link to 3D visualization plot showing :math:`B_N/|B|` error on plasma surface with initial (pre-optimization) coils",
        "- :math:`f`: Link to 3D visualization plot showing :math:`B_N/|B|` error on plasma surface with final (optimized) coils",
        "- **PP**: Link to Poincaré plot showing fieldline trajectories",
        "- **BP**: Link to Boozer surface plot showing flux surfaces",
        "- **QS**: Link to quasisymmetry error profile plot",
        "- **iota**: Link to rotational transform (iota) profile plot",
        "- **FPT**: Link to Fast Particle Tracing (SIMPLE) loss fraction plot",
        "",
    ]
    
    # Build surface-specific content
    lines = surface_specific_lines
    
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
                # Convert shorthand to math mode (e.g., "d_cc" -> ":math:`d_{cc}`", "F_max" -> ":math:`F_\text{max}`")
                math_shorthand = _shorthand_to_math(shorthand)
                surface_header_cols.append(math_shorthand)
            # Add Date, User, i, f, and plot links at the end (use math mode with \text{} to avoid bold formatting)
            surface_header_cols.extend([
                r":math:`\text{Date}`",
                r":math:`\text{User}`",
                r":math:`\text{i}`",
                r":math:`\text{f}`",
                r":math:`\text{PP}`",
                r":math:`\text{BP}`",
                r":math:`\text{QS}`",
                r":math:`\text{iota}`",
                r":math:`\text{FPT}`"
            ])

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
                # Normalize entry_path: remove leading slash if present
                if entry_path.startswith("/"):
                    entry_path = entry_path[1:]
                i_link = "—"  # Initial coils link - show dash if PDF doesn't exist
                f_link = rank_num  # Final coils link - show rank number
                poincare_link = "—"  # Poincaré plot link
                boozer_link = "—"  # Boozer plot link
                qs_link = "—"  # Quasisymmetry plot link
                iota_link = "—"  # Iota plot link
                fpt_link = "—"  # Fast Particle Tracing plot link
                
                # Check if this is a Fourier continuation submission
                fourier_orders_str = metrics.get("fourier_continuation_orders")
                is_fourier_continuation = fourier_orders_str and fourier_orders_str != "—"
                
                if entry_path:
                    repo_root = Path(out_rst.parent.parent).resolve()
                    # Use jsdelivr CDN which serves files with proper content-type headers for inline viewing
                    github_base_url = "https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main"
                    
                    # Determine submission directory
                    path_obj = Path(entry_path)
                    submission_dir = None
                    
                    if path_obj.name == "all_files.zip":
                        # New structure: PDFs are in the same directory as the zip file
                        submission_dir = path_obj.parent
                    elif path_obj.suffix == ".zip":
                        # Legacy format: handle old zip files
                        zip_stem = path_obj.stem
                        if zip_stem.count('-') >= 4 and '_' in zip_stem:
                            timestamp = zip_stem
                            path_parts = path_obj.parts
                            if "submissions" in path_parts:
                                submissions_idx = path_parts.index("submissions")
                                if submissions_idx + 2 < len(path_parts):
                                    user = path_parts[submissions_idx + 2]
                                    new_dir = repo_root / "submissions" / user / timestamp
                                    if new_dir.exists():
                                        submission_dir = Path("submissions") / user / timestamp
                                    else:
                                        old_date_dir = path_obj.parent / timestamp
                                        if old_date_dir.exists():
                                            submission_dir = path_obj.parent / timestamp
                                        else:
                                            submission_dir = path_obj.parent
                                else:
                                    submission_dir = path_obj.parent
                            else:
                                submission_dir = path_obj.parent
                        else:
                            submission_dir = path_obj.parent
                    else:
                        # Not a zip file - PDFs should be in the same directory as results.json
                        submission_dir = path_obj.parent
                    
                    if submission_dir:
                        # Ensure submission_dir is relative to repo_root
                        # If it's absolute, convert to relative
                        if submission_dir.is_absolute():
                            try:
                                submission_dir = submission_dir.relative_to(repo_root.resolve())
                            except ValueError:
                                # If can't make relative, try extracting submissions part
                                submission_str = str(submission_dir)
                                if "submissions" in submission_str:
                                    submissions_idx = submission_str.find("submissions")
                                    submission_dir = Path(submission_str[submissions_idx:])
                                else:
                                    # Fallback: use path_obj.parent if it's relative
                                    if not path_obj.is_absolute():
                                        submission_dir = path_obj.parent
                                    else:
                                        # Can't determine relative path, skip this entry's plots
                                        submission_dir = None
                        
                        # Normalize submission_dir to ensure it's a proper relative path without leading slash
                        if submission_dir:
                            # Convert to string and normalize
                            submission_dir_str = str(submission_dir).replace("\\", "/")
                            # Remove leading slash if present (can happen if path was absolute)
                            submission_dir_str = submission_dir_str.lstrip("/")
                            # Remove leading "./" if present
                            if submission_dir_str.startswith("./"):
                                submission_dir_str = submission_dir_str[2:]
                            # Ensure it doesn't start with a slash after normalization
                            submission_dir_str = submission_dir_str.lstrip("/")
                            submission_dir = Path(submission_dir_str)
                        
                        if submission_dir:
                            full_submission_dir = (repo_root / submission_dir).resolve()
                            orders = []  # Initialize orders list
                            
                            if is_fourier_continuation:
                                # Fourier continuation: handle order_X subdirectories
                                # Parse orders from string (e.g., "4,6,8" -> [4, 6, 8])
                                try:
                                    orders = [int(o.strip()) for o in fourier_orders_str.split(",")]
                                except (ValueError, AttributeError):
                                    orders = []
                                
                                if orders:
                                    # Find all order_X directories that exist
                                    order_dirs = []
                                    for order in orders:
                                        order_dir_name = f"order_{order}"
                                        order_dir_path = full_submission_dir / order_dir_name
                                        if order_dir_path.exists() and order_dir_path.is_dir():
                                            order_dirs.append((order, order_dir_name))
                                    
                                    if order_dirs:
                                        # For "i": use initial PDF from first order
                                        first_order, first_order_dir = order_dirs[0]
                                        initial_pdf_path = submission_dir / first_order_dir / "bn_error_3d_plot_initial.pdf"
                                        full_initial_pdf_path = repo_root / initial_pdf_path
                                        if full_initial_pdf_path.exists():
                                            pdf_url_path_initial = str(initial_pdf_path).replace("\\", "/")
                                            pdf_url_initial = f"{github_base_url}/{pdf_url_path_initial}"
                                            i_link = f"`{rank_num} <{pdf_url_initial}>`__"
                                        
                                        # For "f": create multiple links, one for each order
                                        f_links = []
                                        for order, order_dir_name in order_dirs:
                                            final_pdf_path = submission_dir / order_dir_name / "bn_error_3d_plot.pdf"
                                            full_final_pdf_path = repo_root / final_pdf_path
                                            if full_final_pdf_path.exists():
                                                pdf_url_path = str(final_pdf_path).replace("\\", "/")
                                                pdf_url = f"{github_base_url}/{pdf_url_path}"
                                                f_links.append(f"`{order} <{pdf_url}>`__")
                                        
                                        if f_links:
                                            # Join multiple links with spaces
                                            f_link = " ".join(f_links)
                            else:
                                # Standard submission: PDFs in submission directory
                                pdf_path = submission_dir / "bn_error_3d_plot.pdf"
                                pdf_path_initial = submission_dir / "bn_error_3d_plot_initial.pdf"
                                
                                # Check if PDFs exist and create links
                                full_pdf_path = (repo_root / pdf_path).resolve()
                                if full_pdf_path.exists():
                                    pdf_url_path = str(pdf_path).replace("\\", "/")
                                    pdf_url = f"{github_base_url}/{pdf_url_path}"
                                    f_link = f"`{rank_num} <{pdf_url}>`__"
                                
                                full_pdf_path_initial = (repo_root / pdf_path_initial).resolve()
                                if full_pdf_path_initial.exists():
                                    pdf_url_path_initial = str(pdf_path_initial).replace("\\", "/")
                                    pdf_url_initial = f"{github_base_url}/{pdf_url_path_initial}"
                                    i_link = f"`{rank_num} <{pdf_url_initial}>`__"
                            
                            # Find plot files (poincare, boozer, quasisymmetry, iota, simple)
                            # These are typically in the submission directory or post_processing subdirectory
                            plot_files = [
                                ("poincare_plot.png", "poincare"),
                                ("boozer_surface.png", "boozer"),
                                ("quasisymmetry_profile.png", "qs"),
                                ("iota_profile.png", "iota"),
                                ("simple_loss_fraction.png", "fpt"),  # Fast Particle Tracing
                            ]
                            
                            # Determine relative paths to check for plot files
                            # Use submission_dir (already relative) as base, similar to PDF links
                            plot_paths_to_check = []
                            
                            if is_fourier_continuation and orders:
                                # For Fourier continuation, check highest order directory first
                                # (plots are typically generated for the final order)
                                highest_order = max(orders)
                                highest_order_dir_name = f"order_{highest_order}"
                                # Check in order_X directory
                                plot_paths_to_check.append((submission_dir / highest_order_dir_name, False))
                                # Check in order_X/post_processing directory
                                plot_paths_to_check.append((submission_dir / highest_order_dir_name / "post_processing", True))
                            
                            # Always check main submission directory and its post_processing subdirectory
                            plot_paths_to_check.append((submission_dir, False))
                            plot_paths_to_check.append((submission_dir / "post_processing", True))
                            
                            for filename, plot_type in plot_files:
                                for plot_dir_rel, is_post_processing in plot_paths_to_check:
                                    # Check if file exists using absolute path
                                    plot_dir_abs = repo_root / plot_dir_rel
                                    plot_path_abs = plot_dir_abs / filename
                                    if plot_path_abs.exists():
                                        # Use relative path (plot_dir_rel) for URL, same as PDF links
                                        # Construct path and normalize to ensure no leading slash
                                        plot_path_rel = plot_dir_rel / filename
                                        plot_url_path = str(plot_path_rel).replace("\\", "/")
                                        # Remove any leading slashes or "./" prefixes (jsdelivr needs relative paths)
                                        plot_url_path = plot_url_path.lstrip("/")
                                        if plot_url_path.startswith("./"):
                                            plot_url_path = plot_url_path[2:]
                                        # Double-check: ensure no leading slash remains
                                        plot_url_path = plot_url_path.lstrip("/")
                                        # Construct URL - github_base_url already ends without slash
                                        plot_url = f"{github_base_url}/{plot_url_path}"
                                        # Update the appropriate link variable
                                        if plot_type == "poincare":
                                            poincare_link = f"`{rank_num} <{plot_url}>`__"
                                        elif plot_type == "boozer":
                                            boozer_link = f"`{rank_num} <{plot_url}>`__"
                                        elif plot_type == "qs":
                                            qs_link = f"`{rank_num} <{plot_url}>`__"
                                        elif plot_type == "iota":
                                            iota_link = f"`{rank_num} <{plot_url}>`__"
                                        elif plot_type == "fpt":
                                            fpt_link = f"`{rank_num} <{plot_url}>`__"
                                        break
                
                # Build row: metrics first, then Date, User, i, f, and plot links at the end
                row_parts = []
                for key in surface_metric_keys:
                    value = metrics.get(key)
                    formatted = _format_value(value, metric_key=key) if value is not None else "—"
                    row_parts.append(formatted)
                # Add Date, User, i, f, and plot links at the end
                row_parts.extend([
                    run_date,
                    entry.get("contact", entry.get("method_name", "?"))[:15],
                    i_link,
                    f_link,
                    poincare_link,
                    boozer_link,
                    qs_link,
                    iota_link,
                    fpt_link,
                ])
                
                # First column
                lines.append("   * - " + row_parts[0])
                # Remaining columns
                for val in row_parts[1:]:
                    lines.append("     - " + val)
            
            lines.append("")
            lines.append("")

    surface_specific_lines.extend([
        "",
        ".. note::",
        "   Last updated: run ``stellcoilbench update-db`` to refresh locally.",
        "",
    ])

    # Write all three files
    out_rst.parent.mkdir(parents=True, exist_ok=True)
    out_rst.write_text("\n".join(main_lines))
    
    metric_def_file = leaderboard_dir / "metric_definitions.rst"
    metric_def_file.write_text("\n".join(metric_def_lines))
    
    surface_specific_file = leaderboard_dir / "surface_specific.rst"
    surface_specific_file.write_text("\n".join(surface_specific_lines))








def build_surface_leaderboards(
    leaderboard: Dict[str, Any],
    submissions_root: Path,
    plasma_surfaces_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Group entries by plasma surface extracted from case.yaml files.
    
    Submission paths can be:
    - Old: submissions/<surface>/<username>/<timestamp>.zip
    - New: submissions/<username>/<timestamp>/all_files.zip
    
    Returns dict mapping surface_name -> {"entries": [...]}
    """
    entries = leaderboard.get("entries") or []
    surface_leaderboards: Dict[str, Dict[str, Any]] = {}
    
    # Group entries by surface extracted from case.yaml
    for entry in entries:
        path_str = entry.get("path", "")
        if not path_str:
            continue
        
        # path_str is relative to repo_root (e.g., "submissions/surface/user/timestamp/results.json")
        # Resolve it relative to submissions_root or repo_root
        if path_str.startswith("submissions/"):
            # Relative to repo_root
            path_obj = submissions_root.parent / path_str if submissions_root.parent else Path(path_str)
        else:
            path_obj = Path(path_str)
        
        surface_name = "unknown"
        
        # Try to extract surface from case.yaml
        # Handle both zip files and directories
        if path_obj.suffix == ".zip":
            # Zip file - try to read case.yaml from inside
            try:
                import zipfile
                with zipfile.ZipFile(path_obj, 'r') as zf:
                    if "case.yaml" in zf.namelist():
                        import yaml
                        case_content = zf.read("case.yaml").decode('utf-8')
                        case_data = yaml.safe_load(case_content)
                        surface_file = case_data.get("surface_params", {}).get("surface", "")
                        if surface_file:
                            surface_name = Path(surface_file).name
                            if surface_name.startswith("input."):
                                surface_name = surface_name[6:]
                            elif surface_name.startswith("wout."):
                                surface_name = surface_name[5:]
            except Exception:
                pass
            
            # Fallback: try to extract from old structure path
            if surface_name == "unknown":
                path_parts = path_obj.parts
                if "submissions" in path_parts:
                    submissions_idx = path_parts.index("submissions")
                    parts_after = path_parts[submissions_idx + 1:]
                    # Old structure: submissions/<surface>/<user>/<timestamp>.zip
                    if len(parts_after) >= 3 and parts_after[-1].endswith('.zip') and parts_after[-1] != 'all_files.zip':
                        surface_name = parts_after[0]
        else:
            # Directory or results.json file - try to find case.yaml nearby
            if path_obj.name == "results.json":
                case_yaml_path = path_obj.parent / "case.yaml"
            else:
                case_yaml_path = path_obj / "case.yaml"
            
            if case_yaml_path.exists():
                try:
                    import yaml
                    case_data = yaml.safe_load(case_yaml_path.read_text())
                    surface_file = case_data.get("surface_params", {}).get("surface", "")
                    if surface_file:
                        surface_name = Path(surface_file).name
                        if surface_name.startswith("input."):
                            surface_name = surface_name[6:]
                        elif surface_name.startswith("wout."):
                            surface_name = surface_name[5:]
                        # Remove file extension if present (e.g., ".focus")
                        if "." in surface_name:
                            surface_name = surface_name.split(".", 1)[0]
                except Exception:
                    pass
            
            # Fallback: try to extract from path structure
            # New structure: submissions/<surface>/<user>/<timestamp>/results.json or all_files.zip
            if surface_name == "unknown":
                path_parts = path_obj.parts
                if "submissions" in path_parts:
                    submissions_idx = path_parts.index("submissions")
                    parts_after = path_parts[submissions_idx + 1:]
                    # New structure: submissions/<surface>/<user>/<timestamp>/file
                    if len(parts_after) >= 3:
                        surface_name = parts_after[0]
                else:
                    # Try relative path structure
                    try:
                        rel_path = path_obj.relative_to(submissions_root)
                        rel_parts = rel_path.parts
                        if len(rel_parts) >= 3:
                            # Structure: surface/user/timestamp/file
                            surface_name = rel_parts[0]
                    except ValueError:
                        pass
        
        if surface_name == "unknown":
            # Skip entries where we can't determine surface
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
        # All other numeric values use scientific notation with 1 digit
        # CSS will handle making numbers smaller (no HTML needed)
        if isinstance(value, (float, int)):
            return f"{float(value):.1e}"
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

