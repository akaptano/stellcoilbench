# src/coilbench/update_db.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Reactor-scale engineering constraints
# ---------------------------------------------------------------------------
# Submissions violating a *hard* constraint are infeasible (score = 0) and
# excluded from the main leaderboard.  Soft constraints contribute to the
# composite score via exponential margin factors but do not cause exclusion.
#
# Hard constraints:
#   - Coil-surface linking (topological)
#   - Coil-coil linking number (topological)
#   - Max turns per coil ≤ N_TURNS_MODEL
#   - Finite-build coil-coil clearance ≥ 0  (winding packs must not overlap)
#
# Each constraint is a dict with:
#   metric     – key inside reactor_scale_metrics or metrics (for dimensionless)
#   source     – "reactor_scale_metrics" or "metrics"
#   bound      – numeric bound
#   direction  – "max" (value ≤ bound), "min" (value ≥ bound), or "eq"
#   transform  – optional callable applied to the raw value before comparison
#   hard       – if True, violation sets score = 0 (default False → soft)
#   label      – human-readable name for messages / docs
#   units      – unit string for display

# Number of turns assumed per coil for force feasibility assessment.
# With N turns the current per turn is I/N, so the Lorentz force per
# unit length on each turn drops by a factor of N relative to the
# single-turn value reported by simsopt.
N_TURNS_MODEL: int = 500
REACTOR_SCALE_CONSTRAINTS: List[Dict[str, Any]] = [
    # ---- Hard feasibility constraints (infeasible if violated) ----
    {
        "metric": "coils_linked_to_surface",
        "source": "metrics",
        "bound": True,
        "direction": "eq",       # value must equal bound exactly
        "hard": True,            # infeasibility → composite score = 0
        "label": "Coils linked to plasma surface",
        "units": "(boolean)",
    },
    {
        "metric": "final_linking_number",
        "source": "metrics",
        "bound": 0.5,
        "direction": "max",
        "transform": abs,        # linking number can be negative; check |LN| < 0.5
        "hard": True,
        "label": "Coil-coil linking number (\\|LN\\| ≈ 0)",
        "units": "(dimensionless)",
    },
    # ---- Soft engineering constraints (contribute to composite score) ----
    {
        "metric": "avg_BdotN_over_B",
        "source": "metrics",
        "bound": 1e-2,
        "direction": "max",
        "label": "avg ⟨B·n⟩/⟨B⟩",
        "units": "(dimensionless)",
    },
    {
        "metric": "reactor_scale_min_cs_separation",
        "source": "reactor_scale_metrics",
        "bound": 1.3,
        "direction": "min",
        "label": "Minimum coil-surface distance",
        "units": "m",
    },
    {
        "metric": "reactor_scale_min_cc_separation",
        "source": "reactor_scale_metrics",
        "bound": 0.7,
        "direction": "min",
        "label": "Minimum coil-coil distance",
        "units": "m",
    },
    {
        "metric": "reactor_scale_total_length",
        "source": "reactor_scale_metrics",
        "bound": 220.0,
        "direction": "max",
        "label": "Total coil length",
        "units": "m",
    },
    {
        "metric": "reactor_scale_max_curvature",
        "source": "reactor_scale_metrics",
        "bound": 1.0,
        "direction": "max",
        "label": "Max curvature κ",
        "units": "m⁻¹",
    },
    {
        "metric": "reactor_scale_mean_squared_curvature",
        "source": "reactor_scale_metrics",
        "bound": 1.0,
        "direction": "max",
        "transform": math.sqrt,  # compare sqrt(MSC) since MSC is in m⁻²
        "label": "Max √MSC (RMS curvature)",
        "units": "m⁻¹",
    },
    {
        "metric": "reactor_scale_arclength_variation",
        "source": "reactor_scale_metrics",
        "bound": 1.0,
        "direction": "max",
        "transform": math.sqrt,  # compare sqrt(Var) since Var is in m²
        "label": "Arclength variation √Var",
        "units": "m",
    },
    # ---- Hard feasibility constraints (additional) ----
    {
        "metric": "N_turns_per_coil",
        "source": "reactor_scale_metrics",
        "bound": N_TURNS_MODEL,
        "direction": "max",
        "transform": lambda x: max(x) if isinstance(x, list) and x else 0,
        "hard": True,
        "label": f"Max turns per coil (N_turns ≤ {N_TURNS_MODEL})",
        "units": "(turns)",
    },
    {
        "metric": "finite_build_cc_clearance",
        "source": "reactor_scale_metrics",
        "bound": 0.0,
        "direction": "min",       # clearance must be ≥ 0  (non-negative)
        "hard": True,
        "label": "Finite-build coil-coil clearance (d_cc > w_WP)",
        "units": "m",
    },
]


def check_reactor_constraints(
    metrics: Dict[str, Any],
    reactor_scale_metrics: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Check whether a submission meets all reactor-scale engineering constraints.

    Parameters
    ----------
    metrics : dict
        Device-scale metrics dict (from results.json ``"metrics"``).
    reactor_scale_metrics : dict
        Reactor-scale metrics dict (from results.json ``"reactor_scale_metrics"``).

    Returns
    -------
    passes_hard : bool
        True if all **hard** constraints are satisfied.  Soft-constraint
        violations do *not* affect this flag — they only lower the
        composite score.
    violations : list[dict]
        List of *all* violated constraints (hard **and** soft).  Each
        entry has keys ``label``, ``value``, ``bound``, ``direction``,
        ``units``, and ``hard`` (True for infeasibility constraints).
    """
    violations: List[Dict[str, Any]] = []

    for constraint in REACTOR_SCALE_CONSTRAINTS:
        source_dict = (
            reactor_scale_metrics if constraint["source"] == "reactor_scale_metrics"
            else metrics
        )
        raw_value = source_dict.get(constraint["metric"])
        if raw_value is None:
            # Metric not available – skip (don't penalize old submissions)
            continue

        transform = constraint.get("transform")
        value = transform(raw_value) if transform is not None else raw_value

        direction = constraint["direction"]
        bound = constraint["bound"]

        violated = False
        if direction == "max" and value > bound:
            violated = True
        elif direction == "min" and value < bound:
            violated = True
        elif direction == "eq" and value != bound:
            violated = True

        if violated:
            violations.append({
                "label": constraint["label"],
                "metric": constraint["metric"],
                "value": value,
                "bound": bound,
                "direction": direction,
                "units": constraint.get("units", ""),
                "hard": constraint.get("hard", False),
            })

    has_hard_violation = any(v["hard"] for v in violations)
    return (not has_hard_violation), violations


def compute_composite_score(
    metrics: Dict[str, Any],
    reactor_scale_metrics: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Compute a composite feasibility/quality score.

    The score combines all engineering constraints into a single number via
    a geometric mean of exponential margin factors:

    .. math::

        \\text{score} = \\exp\\!\\left(\\frac{1}{n}\\sum_i m_i\\right)
                      = \\left(\\prod_i e^{m_i}\\right)^{1/n}

    where the margin *m_i* for each constraint is:

    * **"max" constraints** (value ≤ bound): ``m = 1 − value / bound``
    * **"min" constraints** (value ≥ bound): ``m = value / bound − 1``

    Interpretation:

    * score = 0  → hard infeasibility (coils delinked, coils interlinked)
    * score < 1  → one or more soft constraints violated on average
    * score = 1  → constraints met exactly on average
    * score > 1  → constraints met with engineering margin

    Parameters
    ----------
    metrics : dict
        Device-scale metrics dict.
    reactor_scale_metrics : dict
        Reactor-scale metrics dict.

    Returns
    -------
    score : float | None
        Composite score: 0.0 for hard infeasibility, > 0 for feasible
        designs, or None when no soft-constraint metrics are available.
    details : dict
        Diagnostic information: per-factor margins, hard-constraint status.
    """
    details: Dict[str, Any] = {"factors": {}, "infeasible": False}

    # ---- Hard feasibility checks (score = 0 if any fail) ----
    for c in REACTOR_SCALE_CONSTRAINTS:
        if not c.get("hard", False):
            continue
        source_dict = (
            reactor_scale_metrics if c["source"] == "reactor_scale_metrics"
            else metrics
        )
        raw_value = source_dict.get(c["metric"])
        if raw_value is None:
            continue  # missing → don't penalize

        transform = c.get("transform")
        value = transform(raw_value) if transform is not None else raw_value
        bound = c["bound"]
        direction = c["direction"]

        hard_fail = False
        if direction == "max" and value > bound:
            hard_fail = True
        elif direction == "min" and value < bound:
            hard_fail = True
        elif direction == "eq" and value != bound:
            hard_fail = True

        if hard_fail:
            details["infeasible"] = True
            details["reason"] = f"{c['label']}: value={value}, bound={bound}"
            return 0.0, details

    # ---- Soft constraint factors (geometric mean) ----
    exponents: List[float] = []
    for c in REACTOR_SCALE_CONSTRAINTS:
        if c.get("hard", False):
            continue  # hard constraints handled above
        bound = c["bound"]
        if bound == 0:
            continue  # can't form ratio with zero bound

        source_dict = (
            reactor_scale_metrics if c["source"] == "reactor_scale_metrics"
            else metrics
        )
        raw_value = source_dict.get(c["metric"])
        if raw_value is None:
            continue

        transform = c.get("transform")
        value = transform(raw_value) if transform is not None else raw_value

        if c["direction"] == "max":
            exponent = 1.0 - value / bound
        else:  # "min"
            exponent = value / bound - 1.0

        exponents.append(exponent)
        details["factors"][c["metric"]] = {
            "value": float(value),
            "bound": float(bound),
            "direction": c["direction"],
            "margin": float(exponent),
            "factor": float(math.exp(exponent)),
        }

    if not exponents:
        details["reason"] = "No metrics available for scoring"
        return None, details  # None = no data (not infeasible, just unknown)

    mean_exponent = sum(exponents) / len(exponents)
    score = math.exp(mean_exponent)
    details["n_factors"] = len(exponents)
    details["mean_margin"] = float(mean_exponent)

    return float(score), details


def _metric_shorthand(metric_name: str) -> str:
    """
    Convert metric names to compact shorthand/acronyms for display in leaderboard.
    
    Uses LaTeX-style notation where appropriate for compactness.
    """
    shorthand_map = {
        # B-field related
        "max_BdotN_over_B": "max(B_n)",
        "avg_BdotN_over_B": "B̄_n",
        "final_squared_flux": "f_B",
        "final_normalized_squared_flux": "f_B",  # Legacy name (backwards compatibility)
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

        # Reactor-scale metrics (displayed in the reactor-scale leaderboard)
        "reactor_scale_min_cs_separation": "d_cs",
        "reactor_scale_min_cc_separation": "d_cc",
        "reactor_scale_total_length": "L",
        "reactor_scale_max_curvature": "κ_max",
        "reactor_scale_average_curvature": "κ̄",
        "reactor_scale_mean_squared_curvature": "MSC",
        "reactor_scale_max_max_coil_force": "F_max",
        "reactor_scale_avg_max_coil_force": "F̄",
        "reactor_scale_max_max_coil_torque": "τ_max",
        "reactor_scale_avg_max_coil_torque": "τ̄",
        "reactor_scale_arclength_variation": "Var(l_i)",
        "reactor_scale_squared_flux": "f_B",
        "total_superconductor_length_km": "L_SC",
        "max_winding_pack_width": "w_WP",
        "per_turn_max_force": "F_turn",
        "per_turn_max_torque": "τ_turn",
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
        "L_SC": r":math:`L_\text{SC}`",
        "w_WP": r":math:`w_\text{WP}`",
        "F_turn": r":math:`F_\text{turn}`",
        "τ_turn": r":math:`\tau_\text{turn}`",
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


# Units for reactor-scale metric columns (LaTeX math fragments)
_RS_UNITS: Dict[str, str] = {
    "reactor_scale_squared_flux": r"\text{T}^2\text{m}^2",
    "reactor_scale_min_cs_separation": r"\text{m}",
    "reactor_scale_min_cc_separation": r"\text{m}",
    "reactor_scale_total_length": r"\text{m}",
    "reactor_scale_max_curvature": r"\text{m}^{-1}",
    "reactor_scale_average_curvature": r"\text{m}^{-1}",
    "reactor_scale_mean_squared_curvature": r"\text{m}^{-2}",
    "reactor_scale_max_max_coil_force": r"\text{MN/m}",
    "reactor_scale_avg_max_coil_force": r"\text{MN/m}",
    "reactor_scale_max_max_coil_torque": r"\text{MN}",
    "reactor_scale_avg_max_coil_torque": r"\text{MN}",
    "per_turn_max_force": r"\text{MN/m}",
    "per_turn_max_torque": r"\text{MN}",
    "total_superconductor_length_km": r"\text{km}",
    "max_winding_pack_width": r"\text{m}",
    "reactor_scale_arclength_variation": r"\text{m}^2",
}


def _metric_definition(metric_name: str) -> str:
    """
    Get detailed mathematical definition for a metric.
    
    Returns a string with LaTeX-style mathematical notation describing the metric.
    Format: symbol = expression - description
    """
    definitions = {
        # B-field related
        "final_squared_flux": r"Squared flux objective $f_B = \int_{S} (\mathbf{B} \cdot \mathbf{n} - B_\text{target})^2 dS$ on plasma surface ($\text{T}^2 \text{m}^2$). When virtual casing is used, $B_\text{target} = B_\text{external}^\text{normal}$; otherwise $B_\text{target} = 0$.",
        "final_normalized_squared_flux": r"Squared flux objective $f_B = \int_{S} (\mathbf{B} \cdot \mathbf{n} - B_\text{target})^2 dS$ on plasma surface ($\text{T}^2 \text{m}^2$). Legacy name for final_squared_flux.",
        "max_BdotN_over_B": r"Maximum normalized normal field component $\max(B_n)$ where $B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}$ (dimensionless)",
        "avg_BdotN_over_B": r"Average normalized normal field component $\bar{B}_n = \frac{\langle |\mathbf{B} \cdot \mathbf{n} - B_\text{target}| \rangle}{\langle |\mathbf{B}| \rangle}$ (dimensionless). When virtual casing is used, $B_\text{target} = B_\text{external}^\text{normal}$; otherwise $B_\text{target} = 0$.",
        
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


def _recompute_coils_linked_to_surface(
    submission_path: Path,
    surface_name: str,
    repo_root: Path,
) -> bool | None:
    """Recompute *coils_linked_to_surface* from stored coil/surface data.

    Uses the per-phi-slice R check (matching the fix in coil_optimization.py)
    to determine whether each coil topologically encircles the plasma at its
    local toroidal position.

    Returns True/False on success, or None if the data cannot be loaded
    (missing files, simsopt not installed, etc.).
    """
    try:
        from simsopt._core import load as simsopt_load  # type: ignore
        from simsopt.geo import SurfaceRZFourier  # type: ignore
        import numpy as np
        import zipfile
        import tempfile
        import os
    except ImportError:
        return None

    # ---- locate BiotSavart JSON inside the submission ----
    bs_json_bytes: bytes | None = None
    try:
        if submission_path.suffix == ".zip":
            with zipfile.ZipFile(submission_path, "r") as zf:
                bs_files = [
                    n for n in zf.namelist()
                    if n.endswith("biot_savart_optimized.json")
                ]
                if not bs_files:
                    return None
                # Prefer highest order (order_16 > order_8 > order_4 > root)
                bs_files.sort(reverse=True)
                bs_json_bytes = zf.read(bs_files[0])
        else:
            # Regular directory – look for biot_savart in parent
            submission_dir = submission_path.parent
            candidates = sorted(
                submission_dir.rglob("biot_savart_optimized.json"),
                key=lambda p: str(p),
                reverse=True,
            )
            if candidates:
                bs_json_bytes = candidates[0].read_bytes()
    except Exception:
        return None

    if bs_json_bytes is None:
        return None

    # ---- load BiotSavart ----
    try:
        fd, tmpfile = tempfile.mkstemp(suffix=".json")
        os.write(fd, bs_json_bytes)
        os.close(fd)
        bs = simsopt_load(tmpfile)
    except Exception:
        try:
            os.unlink(tmpfile)
        except Exception:
            pass
        return None
    finally:
        try:
            os.unlink(tmpfile)
        except Exception:
            pass

    # ---- locate and load the plasma surface ----
    plasma_surfaces_dir = repo_root / "plasma_surfaces"
    surface_file = None
    for prefix in ["input.", ""]:
        candidate = plasma_surfaces_dir / f"{prefix}{surface_name}"
        if candidate.exists():
            surface_file = candidate
            break
    if surface_file is None:
        return None

    try:
        s = SurfaceRZFourier.from_vmec_input(str(surface_file), range="full torus")
    except Exception:
        return None

    # ---- per-phi-slice linking check ----
    try:
        surface_gamma = s.gamma()
        R_surface = np.sqrt(
            surface_gamma[:, :, 0] ** 2 + surface_gamma[:, :, 1] ** 2
        )
        R_min_per_phi = np.min(R_surface, axis=1)
        R_max_per_phi = np.max(R_surface, axis=1)
        phi_surface_slices = np.arctan2(
            surface_gamma[:, 0, 1], surface_gamma[:, 0, 0]
        )

        coils = bs.coils
        for coil in coils:
            gamma = coil.curve.gamma()
            R_coil = np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)
            phi_coil = np.arctan2(gamma[:, 1], gamma[:, 0])
            dphi = phi_coil[:, None] - phi_surface_slices[None, :]
            dphi = np.abs(np.arctan2(np.sin(dphi), np.cos(dphi)))
            nearest = np.argmin(dphi, axis=1)
            local_R_min = R_min_per_phi[nearest]
            local_R_max = R_max_per_phi[nearest]
            if not (np.any(R_coil < local_R_min) and np.any(R_coil > local_R_max)):
                return False
        return True
    except Exception:
        return None


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
    
    skipped_constraints = 0

    for method_key, path, data in _load_submissions(submissions_root):
        loaded_count += 1
        meta = data.get("metadata") or {}
        metrics = data.get("metrics") or {}
        reactor_scale = data.get("reactor_scale_metrics") or {}
        
        # Handle legacy format where metrics are at top level (not in "metrics" key)
        if not metrics and ("final_squared_flux" in data or "final_normalized_squared_flux" in data):
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
            fallback = metrics.get("final_squared_flux")
            if fallback is None:
                fallback = metrics.get("final_flux")
            if fallback is None:
                fallback = metrics.get("final_normalized_squared_flux")  # Legacy name
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

        # ---- Backfill Jc-based N_turns and total SC length for older submissions ----
        if ("N_turns_jc" not in reactor_scale
                and "N_turns_per_coil" in reactor_scale):
            # Older submissions have force-only N_turns.  Re-derive using
            # the REBCO Jc model so that N_turns = max(force, Jc).
            n_turns_force = reactor_scale["N_turns_per_coil"]
            per_coil_forces_dev = metrics.get("final_max_force_per_coil")
            if (isinstance(n_turns_force, list) and n_turns_force
                    and isinstance(per_coil_forces_dev, list)
                    and len(per_coil_forces_dev) == len(n_turns_force)):
                # Estimate L_scale and B_scale from stored scaling_factors
                sf = reactor_scale.get("scaling_factors", {})
                L_scale_est = sf.get("length_scale")
                B_scale_est = sf.get("B_field_scale")
                target_B_est = sf.get("device_target_B", metrics.get("target_B_field"))
                if L_scale_est and B_scale_est and target_B_est:
                    from stellcoilbench.cli import _compute_N_turns_critical_current
                    per_coil_currents = metrics.get("final_current_per_coil")
                    per_coil_lengths = metrics.get("final_length_per_coil")
                    jc_result = _compute_N_turns_critical_current(
                        per_coil_forces=per_coil_forces_dev,
                        per_coil_currents=per_coil_currents,
                        per_coil_lengths=per_coil_lengths,
                        L_scale=L_scale_est,
                        B_scale=B_scale_est,
                        target_B=target_B_est,
                    )
                    n_turns_jc = jc_result["N_turns_jc"]
                    # Element-wise max(force, Jc)
                    new_n_turns = [max(nf, nj)
                                   for nf, nj in zip(n_turns_force, n_turns_jc)]
                    reactor_scale["N_turns_per_coil"] = new_n_turns
                    reactor_scale["N_turns_force"] = list(n_turns_force)
                    reactor_scale["N_turns_jc"] = n_turns_jc

        # ---- Backfill winding_pack_width_per_coil ----
        n_turns_wp = reactor_scale.get("N_turns_per_coil")
        if (isinstance(n_turns_wp, list) and n_turns_wp
                and "max_winding_pack_width" not in reactor_scale):
            import numpy as _np
            from stellcoilbench.cli import STELLARIS_A_TURN
            turn_side = _np.sqrt(STELLARIS_A_TURN)
            wp_widths = [float(_np.sqrt(n) * turn_side) for n in n_turns_wp]
            reactor_scale["winding_pack_width_per_coil"] = wp_widths
            reactor_scale["max_winding_pack_width"] = float(max(wp_widths))

        # ---- Backfill finite_build_cc_clearance ----
        max_wp = reactor_scale.get("max_winding_pack_width")
        d_cc_rs = reactor_scale.get("reactor_scale_min_cc_separation")
        if (max_wp is not None and d_cc_rs is not None
                and "finite_build_cc_clearance" not in reactor_scale):
            reactor_scale["finite_build_cc_clearance"] = float(d_cc_rs - max_wp)

        # ---- Backfill per_turn_max_force and per_turn_max_torque ----
        n_turns_pt = reactor_scale.get("N_turns_per_coil")
        if (isinstance(n_turns_pt, list) and n_turns_pt
                and "per_turn_max_force" not in reactor_scale):
            # Force: divide per-coil reactor-scale force by N_turns
            rs_forces = reactor_scale.get("reactor_scale_force_per_coil_MN_per_m")
            if isinstance(rs_forces, list) and len(rs_forces) == len(n_turns_pt):
                per_turn_f = [f / n for f, n in zip(rs_forces, n_turns_pt)]
                reactor_scale["per_turn_max_force"] = float(max(per_turn_f))
            elif reactor_scale.get("reactor_scale_max_max_coil_force") is not None:
                # Fallback: divide overall max by min N_turns (conservative)
                reactor_scale["per_turn_max_force"] = float(
                    reactor_scale["reactor_scale_max_max_coil_force"] / min(n_turns_pt)
                )
        if (isinstance(n_turns_pt, list) and n_turns_pt
                and "per_turn_max_torque" not in reactor_scale):
            # Torque: try per-coil, then overall max fallback
            rs_torque_max = reactor_scale.get("reactor_scale_max_max_coil_torque")
            if rs_torque_max is not None:
                reactor_scale["per_turn_max_torque"] = float(
                    rs_torque_max / min(n_turns_pt)
                )

        # ---- Backfill total_superconductor_length_km ----
        n_turns = reactor_scale.get("N_turns_per_coil")
        if (isinstance(n_turns, list) and n_turns
                and "total_superconductor_length_km" not in reactor_scale):
            per_coil_len = metrics.get("final_length_per_coil")
            if isinstance(per_coil_len, list) and len(per_coil_len) == len(n_turns):
                rs_total_len = reactor_scale.get("reactor_scale_total_length")
                if rs_total_len is not None:
                    device_total = metrics.get("final_total_length")
                    if device_total and device_total > 0:
                        L_scale_est = rs_total_len / device_total
                    else:
                        L_scale_est = rs_total_len / sum(per_coil_len) if sum(per_coil_len) > 0 else 1.0
                    reactor_lengths = [ln * L_scale_est for ln in per_coil_len]
                    reactor_scale["total_superconductor_length_km"] = float(
                        sum(n * ln for n, ln in zip(n_turns, reactor_lengths)) / 1e3
                    )
            elif "reactor_scale_total_length" in reactor_scale:
                # Fallback: assume uniform coil length
                rs_total = reactor_scale["reactor_scale_total_length"]
                num_coils = len(n_turns)
                avg_len = rs_total / num_coils
                reactor_scale["total_superconductor_length_km"] = float(
                    sum(n * avg_len for n in n_turns) / 1e3
                )

        # ---- Recompute coils_linked_to_surface using per-phi-slice check ----
        # The original computation used a global R_min/R_max check that gives
        # false negatives on strongly-shaped stellarators (e.g. HSX).
        # Recompute from stored BiotSavart and plasma surface data.
        key_parts = method_key.split(":")
        surface_from_key = key_parts[1] if len(key_parts) >= 2 else ""
        if surface_from_key and surface_from_key != "unknown":
            corrected = _recompute_coils_linked_to_surface(
                path, surface_from_key, repo_root
            )
            if corrected is not None:
                old_val = metrics_numeric.get("coils_linked_to_surface")
                new_val = float(corrected)
                metrics_numeric["coils_linked_to_surface"] = new_val
                metrics["coils_linked_to_surface"] = corrected
                if old_val is not None and old_val != new_val:
                    import sys
                    print(
                        f"  Recomputed coils_linked_to_surface: "
                        f"{old_val} → {new_val} for {path}",
                        file=sys.stderr,
                    )

        # Check reactor-scale engineering constraints
        passes_constraints, violations = check_reactor_constraints(metrics, reactor_scale)
        if not passes_constraints:
            skipped_constraints += 1
            import sys
            print(f"Warning: {path} fails reactor-scale constraints:", file=sys.stderr)
            for v in violations:
                op = "≤" if v["direction"] == "max" else "≥"
                if v["direction"] == "eq":
                    op = "=="
                print(f"  {v['label']}: {v['value']} (bound {op} {v['bound']} {v['units']})"
                      f"{' [HARD]' if v.get('hard') else ''}", file=sys.stderr)

        # Compute composite score (0 for infeasible, geometric mean of margins otherwise)
        composite_score, score_details = compute_composite_score(metrics, reactor_scale)

        methods[method_key] = {
            "method_name": meta.get("method_name", "UNKNOWN"),
            "method_version": meta.get("method_version", path.stem if path.suffix == ".zip" else path.parent.name),
            "contact": github_username,  # Use GitHub username from path, not metadata
            "hardware": meta.get("hardware", ""),
            "run_date": meta.get("run_date", ""),
            "path": rel_path,
            "score_primary": primary_score,
            "composite_score": composite_score,
            "score_details": score_details,
            "metrics": metrics_numeric,
            "reactor_scale_metrics": reactor_scale,
            "passes_constraints": passes_constraints,
            "constraint_violations": violations,
        }
    
    # Log summary
    import sys
    total_duplicates = sum(len(paths) - 1 for paths in duplicate_keys.values())  # -1 because first one isn't a duplicate
    print(f"Loaded {loaded_count} submissions, skipped {skipped_no_metrics} (no metrics), {skipped_no_score} will be filtered (no score), {skipped_constraints} fail constraints", file=sys.stderr)
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

    Ranking uses the **composite score** (higher is better).  The composite
    score is a geometric mean of exponential margin factors over all soft
    engineering constraints.  Entries that fail hard feasibility constraints
    receive a composite score of 0 and are excluded from the main leaderboard.

    Entries without any usable score (neither composite_score nor
    score_primary) are also filtered out.
    """
    entries = []
    excluded_entries = []  # entries that fail constraints (kept for documentation)

    for method_key, md in methods.items():
        metrics = md.get("metrics", {})
        path = md.get("path", "")

        # Determine the composite score (preferred) or fall back to score_primary
        composite_score = md.get("composite_score")
        score_primary = md.get("score_primary")

        # If composite_score is missing, try to derive one from score_primary
        if composite_score is None:
            if "score_primary" in md:
                if score_primary is None:
                    import sys
                    print(f"Warning: Entry {path} has score_primary=None, skipping", file=sys.stderr)
                    continue
            else:
                # score_primary key doesn't exist, try fallback
                score_primary = metrics.get("final_squared_flux")
                if score_primary is None:
                    score_primary = metrics.get("final_normalized_squared_flux")
                if score_primary is None or not isinstance(score_primary, (int, float)):
                    import sys
                    print(f"Warning: Entry {path} has no composite_score or score_primary (metrics keys: {list(metrics.keys())[:5]}), skipping", file=sys.stderr)
                    continue

        entry = {
            "method_key": method_key,
            "method_name": md.get("method_name", "UNKNOWN"),
            "method_version": md.get("method_version", ""),
            "composite_score": float(composite_score) if composite_score is not None else None,
            "score_primary": float(score_primary) if score_primary is not None else None,
            "run_date": md.get("run_date", ""),
            "contact": md.get("contact", ""),
            "hardware": md.get("hardware", ""),
            "path": md.get("path", ""),
            "metrics": metrics,
            "reactor_scale_metrics": md.get("reactor_scale_metrics", {}),
        }

        # Exclude entries that fail *hard* reactor-scale constraints
        # (passes_constraints is False only for hard violations).
        # Soft-constraint violations are captured by composite_score < 1
        # but the entry remains in the main leaderboard.
        # composite_score == None means no data (legacy) → keep.
        all_violations = md.get("constraint_violations", [])
        if not md.get("passes_constraints", True):
            entry["constraint_violations"] = all_violations
            excluded_entries.append(entry)
            continue
        if composite_score is not None and composite_score == 0.0:
            entry["constraint_violations"] = all_violations
            excluded_entries.append(entry)
            continue
        # Carry soft violations through so the reactor-scale table can
        # highlight them, but the entry stays in the main leaderboard.
        if all_violations:
            entry["constraint_violations"] = all_violations

        entries.append(entry)

    # Sort by composite_score descending (higher = better engineering margin).
    # Fall back to score_primary ascending for legacy entries without composite_score.
    def _sort_key(e):
        cs = e.get("composite_score")
        if cs is not None:
            return (1, cs)  # group 1: has composite_score, higher is better
        sp = e.get("score_primary")
        if sp is not None:
            return (0, -sp)  # group 0: legacy, lower squared flux is better
        return (-1, 0)

    entries.sort(key=_sort_key, reverse=True)
    for i, e in enumerate(entries, start=1):
        e["rank"] = i

    import sys
    print(f"Leaderboard: {len(entries)} entries included, {len(excluded_entries)} excluded (failed constraints)", file=sys.stderr)

    return {"entries": entries, "excluded_entries": excluded_entries}


# Metrics that should never appear in device-scale leaderboard tables.
# These are either internal bookkeeping, duplicates of other columns, or
# belong exclusively in the reactor-scale leaderboard.
_DEVICE_LEADERBOARD_EXCLUDE: set[str] = {
    # Sorting / scoring keys (shown as dedicated Score column)
    "score_primary",
    "composite_score",
    # B-field bookkeeping (not useful for ranking)
    "initial_B_field",
    "final_B_field",
    "target_B_field",
    # Threshold / configuration parameters
    "flux_threshold",
    "cc_threshold",
    "cs_threshold",
    "msc_threshold",
    "curvature_threshold",
    "force_threshold",
    "torque_threshold",
    "arclength_variation_threshold",
    # Raw post-processing B·n (duplicates avg_BdotN_over_B / max_BdotN_over_B)
    "BdotN",
    "BdotN_over_B",
    # Legacy duplicate of final_squared_flux (both map to f_B shorthand)
    "final_normalized_squared_flux",
    # Internal / non-display fields
    "coils_linked_to_surface",  # boolean, shown via constraint check
    "arclength_variation",  # intermediate, keep only final_arclength_variation
    # Fourier continuation internals (keep only fourier_continuation_orders)
    "final_order",
    "continuation_step",
    "fourier_continuation",
    "fourier_order",
    # Legacy reactor-scale internals (superseded by total_superconductor_length_km)
    "N_turns_required",
}


def _get_all_metrics_from_entries(entries: list[Dict[str, Any]]) -> list[str]:
    """Get all unique metric keys from overall leaderboard entries."""
    exclude_fields = _DEVICE_LEADERBOARD_EXCLUDE
    
    all_keys = set()
    for entry in entries:
        metrics = entry.get("metrics", {})
        for key in metrics.keys():
            if key not in exclude_fields:
                all_keys.add(key)
    
    # Sort with final_squared_flux (or legacy final_normalized_squared_flux) first
    sorted_keys = sorted(all_keys)
    if "final_squared_flux" in sorted_keys:
        sorted_keys.remove("final_squared_flux")
        sorted_keys.insert(0, "final_squared_flux")
    elif "final_normalized_squared_flux" in sorted_keys:
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
        
        # Build header: Rank, Score, User, Date, then all metrics (compact)
        header_cols = ["#", "Score", "User", "Date"]
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
            
            # Build row: Rank, Score, User, Date, then all metrics
            cs = e.get("composite_score")
            score_str = f"{cs:.3f}" if cs is not None else "—"
            row_parts = [
                str(e['rank']),
                score_str,
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
        exclude_fields = _DEVICE_LEADERBOARD_EXCLUDE
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
            "final_squared_flux",           # fB (new name)
            "final_normalized_squared_flux", # fB (legacy name, for backwards compatibility)
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
        return _surface_display_name(surface_name)

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
    # Put final_squared_flux (or legacy name) first
    if "final_squared_flux" in all_metric_keys:
        all_metric_keys.remove("final_squared_flux")
        all_metric_keys.insert(0, "final_squared_flux")
    elif "final_normalized_squared_flux" in all_metric_keys:
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
        "   leaderboard/reactor_scale",
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
        
        # ---- Composite Score section ----
        metric_def_lines.extend([
            "Composite Score",
            "---------------",
            "",
            "The **Score** column in the leaderboard is a composite feasibility/quality",
            "metric that summarizes how well a design satisfies all reactor-scale",
            "engineering constraints.  It is computed as a geometric mean of exponential",
            "margin factors:",
            "",
            ".. math::",
            "",
            r"   \text{Score}"
            r"     = \exp\!\left(\frac{1}{n}\sum_{i=1}^{n} m_i\right)"
            r"     = \left(\prod_{i=1}^{n} e^{m_i}\right)^{\!1/n}",
            "",
            "where the margin :math:`m_i` for each soft constraint is:",
            "",
            '- **"max" constraints** (value :math:`\\leq` bound):',
            r"  :math:`m_i = 1 - \text{value}/\text{bound}`",
            '- **"min" constraints** (value :math:`\\geq` bound):',
            r"  :math:`m_i = \text{value}/\text{bound} - 1`",
            "",
            "Interpretation:",
            "",
            "- **Score = 0** — hard infeasibility (e.g. coils delinked from plasma, coils interlinked)",
            "- **Score < 1** — one or more soft constraints violated on average",
            "- **Score = 1** — all constraints met exactly on average",
            "- **Score > 1** — constraints met with engineering margin (better)",
            "",
            "Entries are sorted by composite score **descending** (higher is better).",
            "",
        ])

        # ---- Reactor-Scale Constraints section ----
        metric_def_lines.extend([
            "Reactor-Scale Constraints",
            "-" * len("Reactor-Scale Constraints"),
            "",
            "All submissions are scaled to the ARIES-CS reference reactor",
            r"(major radius :math:`R_0 = 7.5\,\text{m}`, on-axis field",
            r":math:`B_0 = 5.7\,\text{T}`) before engineering feasibility is assessed.",
            "",
            "**Hard feasibility constraints** — any violation makes the design infeasible",
            "(score = 0, excluded from the main leaderboard):",
            "",
        ])
        # Dynamically build hard constraints table
        metric_def_lines.extend([
            ".. list-table::",
            "   :header-rows: 1",
            "",
            "   * - Constraint",
            "     - Bound",
            "     - Description",
        ])
        for c in REACTOR_SCALE_CONSTRAINTS:
            if not c.get("hard", False):
                continue
            label = c["label"]
            bound = c["bound"]
            units = c.get("units", "")
            direction = c["direction"]
            if direction == "eq":
                bound_str = f"= {bound}"
            elif direction == "max":
                bound_str = f"≤ {bound}"
            elif direction == "min":
                bound_str = f"≥ {bound}"
            else:
                bound_str = str(bound)
            if units and units not in ("(boolean)", "(turns)"):
                bound_str += f" {units}"
            desc = ""
            if "linked to" in label.lower():
                desc = "Every base coil must topologically encircle the plasma."
            elif "linking" in label.lower():
                desc = "Coils must not interlink with one another."
            elif "turn" in label.lower():
                desc = (
                    f"With :math:`N_{{\\text{{turns}}}}` chosen to keep per-turn force "
                    f"≤ 0.5 MN/m, no coil may require more than {N_TURNS_MODEL} turns."
                )
            elif "finite" in label.lower() or "clearance" in label.lower():
                desc = (
                    "Centreline distance :math:`d_{\\text{cc,min}}` must exceed "
                    "the largest winding-pack width :math:`w_{\\text{WP,max}}` "
                    "to prevent physical overlap of finite-build coils."
                )
            metric_def_lines.extend([
                f"   * - {label}",
                f"     - {bound_str}",
                f"     - {desc}",
            ])
        metric_def_lines.extend([""])

        metric_def_lines.extend([
            "**Soft engineering constraints** — contribute to the composite score via",
            "exponential margin factors.  Violations lower the score below 1 but do not",
            "set it to zero:",
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "",
            "   * - Metric",
            "     - Bound",
            "     - Direction",
            "     - Units",
        ])
        for c in REACTOR_SCALE_CONSTRAINTS:
            if c.get("hard", False):
                continue  # already listed above
            label = c["label"]
            bound = c["bound"]
            units = c.get("units", "")
            direction = c["direction"]
            if direction == "max":
                bound_str = f":math:`\\leq {bound}`"
            elif direction == "min":
                bound_str = f":math:`\\geq {bound}`"
            else:
                bound_str = str(bound)
            metric_def_lines.extend([
                f"   * - {label}",
                f"     - {bound_str}",
                f"     - {direction}",
                f"     - {units}",
            ])
        metric_def_lines.extend([
            "",
            "Winding-Pack Turn-Count Model",
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
            "",
            "The *simsopt* optimiser models each coil as a single filamentary turn",
            "carrying the total current :math:`I`.  In a real reactor the winding pack",
            "contains :math:`N_{\\text{turns}}` turns, each carrying :math:`I / N_{\\text{turns}}`.",
            "We estimate the required number of turns from **two independent criteria**",
            "and take the element-wise maximum:",
            "",
            ".. math::",
            "",
            r"   N_{\text{turns},\,i}"
            r"     = \max\!\bigl(N_{\text{turns},\,i}^{(\text{force})},\;"
            r"                    N_{\text{turns},\,i}^{(J_c)}\bigr)",
            "",
            "1. Force-based turns",
            "^^^^^^^^^^^^^^^^^^^^",
            "",
            "With :math:`N` turns the Lorentz force per unit length on each turn is",
            "",
            ".. math::",
            "",
            r"   F_{\text{turn}} = \frac{F_{\text{reactor,single-turn}}}{N_{\text{turns}}}",
            "",
            "For each coil we find the minimum :math:`N` to keep",
            ":math:`F_{\\text{turn}} \\leq 0.5\\,\\text{MN/m}`:",
            "",
            ".. math::",
            "",
            r"   N_{\text{turns},\,i}^{(\text{force})}"
            r"     = \left\lceil \frac{F_{\text{reactor},\,i}}{0.5\;\text{MN/m}} \right\rceil",
            "",
            "2. Critical-current-density-based turns",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            "",
            "This criterion ensures the HTS superconductor operates within its critical",
            "envelope.  The model follows the Stellaris winding-pack design",
            "(Lion *et al.*, *Fusion Engineering and Design* **214**, 2025, 114868,",
            "Table 7\u20138 and Section 2.9).",
            "",
            "**REBCO tape-stack** :math:`J_c` **model.**",
            "A Kim-like parametrisation calibrated to tape-stack data at 20 K",
            "(field-aligned tapes):",
            "",
            ".. math::",
            "",
            r"   J_c(B, T) = \frac{C_0}{1 + (B/B_0)^\alpha}"
            r"     \;\times\;\frac{1 - T/T_c}{1 - T_{\text{ref}}/T_c}",
            "",
            "with fitted constants at :math:`T_{\\text{ref}} = 20\\,\\text{K}`:",
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "",
            "   * - Parameter",
            "     - Value",
            "     - Description",
            "   * - :math:`C_0`",
            "     - :math:`5.0 \\times 10^{9}\\;\\text{A/m}^2`",
            "     - Zero-field engineering :math:`J_c` (\u2248 5000 A/mm\u00b2)",
            "   * - :math:`B_0`",
            "     - 18.14 T",
            "     - Characteristic field",
            "   * - :math:`\\alpha`",
            "     - 0.902",
            "     - Field exponent",
            "   * - :math:`T_c`",
            "     - 92 K",
            "     - REBCO critical temperature",
            "",
            "Validation against Stellaris Table 8:",
            ":math:`B = 20\\,\\text{T} \\rightarrow J_c \\approx 2450\\;\\text{A/mm}^2`,",
            ":math:`B = 25\\,\\text{T} \\rightarrow J_c \\approx 2200\\;\\text{A/mm}^2`.",
            "",
            "**Stellaris winding-pack parameters.**",
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "",
            "   * - Parameter",
            "     - Value",
            "     - Description",
            "   * - :math:`T_{\\text{op}}`",
            "     - 20 K",
            "     - Operating temperature",
            "   * - :math:`\\eta`",
            "     - 0.80",
            "     - Utilisation cap (:math:`J_{\\text{op}} / J_c \\leq \\eta`)",
            "   * - :math:`I_{\\text{lead,max}}`",
            "     - 50 kA",
            "     - Current-lead limit",
            "   * - :math:`A_{\\text{HTS}}`",
            "     - :math:`36\\;\\text{mm}^2` (6 mm \u00d7 6 mm)",
            "     - HTS tape-stack cross-section per turn",
            "   * - :math:`A_{\\text{turn}}`",
            "     - :math:`400\\;\\text{mm}^2` (20 mm \u00d7 20 mm)",
            "     - Total turn cross-section (incl. stabiliser, insulation, structure)",
            "   * - :math:`f_{\\text{WP}}`",
            "     - 1.3",
            "     - Winding-pack self-field enhancement factor",
            "",
            "**Algorithm for each coil** :math:`i`:",
            "",
            "1. **Required ampere-turns** at reactor scale:",
            "",
            "   .. math::",
            "",
            r"      NI_i = I_{\text{device},i} \times B_{\text{scale}} \times L_{\text{scale}}",
            "",
            "   where :math:`I_{\\text{device},i}` is the *simsopt* single-turn current.",
            "   If per-coil currents are unavailable, :math:`I` is estimated from",
            "   the force data: :math:`I \\approx (F/L) / B_{\\text{device}}`.",
            "",
            "2. **Peak conductor field** estimate:",
            "",
            "   .. math::",
            "",
            r"      B_{\text{ext},i} = \frac{(F/L)_{\text{device},i}}{I_{\text{device},i}} \times B_{\text{scale}},"
            r"      \qquad"
            r"      B_{\text{peak},i} = f_{\text{WP}} \times B_{\text{ext},i}",
            "",
            "   The factor :math:`f_{\\text{WP}} = 1.3` accounts for the additional",
            "   self-field produced by the multi-turn winding pack at its inner edge.",
            "",
            "3. **Critical current of the HTS cable**:",
            "",
            "   .. math::",
            "",
            r"      I_{c,\text{cable}} = J_c(B_{\text{peak}},\; T_{\text{op}}) \times A_{\text{HTS}}",
            "",
            "4. **Operating current per turn** (lead- or tape-limited):",
            "",
            "   .. math::",
            "",
            r"      I_{\text{turn}} = \min\!\bigl(I_{\text{lead,max}},\; \eta \times I_{c,\text{cable}}\bigr)",
            "",
            "5. **Number of turns** from :math:`J_c` requirements:",
            "",
            "   .. math::",
            "",
            r"      N_{\text{turns},\,i}^{(J_c)} = \left\lceil \frac{NI_i}{I_{\text{turn},i}} \right\rceil",
            "",
            "**Hard constraint.**",
            "The final :math:`N_{\\text{turns},i}` (element-wise maximum of force and",
            ":math:`J_c` requirements) must satisfy",
            f":math:`\\max_i N_{{\\text{{turns}},\\,i}} \\leq {N_TURNS_MODEL}`.",
            "",
            "3. Finite-build (winding-pack) extent",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            "",
            "With each turn occupying :math:`A_{\\text{turn}} = 20\\;\\text{mm} \\times",
            "20\\;\\text{mm} = 400\\;\\text{mm}^2` (Table 7 of Lion *et al.*: this area",
            "includes the REBCO tape stack, copper stabiliser, solder, steel jacket,",
            "and helium cooling channel), a square winding pack with :math:`N` turns",
            "has side length",
            "",
            ".. math::",
            "",
            r"   w_{\text{WP}} = \sqrt{N_{\text{turns}}} \times 20\;\text{mm}",
            "",
            "Validation against Stellaris Table 8:",
            "",
            "- Coil 0: :math:`N = 324 \\;\\Rightarrow\\; w = 18 \\times 20\\;\\text{mm} = 360\\;\\text{mm} \\;\\checkmark`",
            "- Coil 5: :math:`N = 225 \\;\\Rightarrow\\; w = 15 \\times 20\\;\\text{mm} = 300\\;\\text{mm} \\;\\checkmark`",
            "",
            "The leaderboard reports :math:`w_{\\text{WP}}` \u2014 the **maximum** winding-pack",
            "side length across all coils (in metres).  This gives the finite-build extent",
            "that must be accommodated by the coil-surface and coil-coil separation gaps.",
            "",
            "4. Finite-build coil-coil intersection check",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            "",
            "*simsopt*'s ``CurveCurveDistance`` penalty measures the **centreline-to-centreline**",
            "distance between coil filaments.  Once the winding-pack extent is known, we",
            "can check whether the finite-build coils would physically overlap.",
            "",
            "Each coil's winding pack extends :math:`w_i / 2` from the centreline in every",
            "direction.  For two coils *i* and *j* separated by centreline distance",
            ":math:`d_{ij}`, the clearance between their outer edges is",
            "",
            ".. math::",
            "",
            r"   \text{clearance}_{ij} = d_{ij} - \frac{w_i}{2} - \frac{w_j}{2}",
            "",
            "Because we only store the **global minimum** coil-coil distance",
            ":math:`d_{\\text{cc,min}} = \\min_{i<j} d_{ij}`, the most conservative",
            "check uses the largest winding-pack width for both coils:",
            "",
            ".. math::",
            "",
            r"   \text{clearance} = d_{\text{cc,min}} - w_{\text{WP,max}}",
            "",
            "where :math:`w_{\\text{WP,max}} = \\max_i w_{\\text{WP},i}`.  This is a **hard",
            "constraint**: if the clearance is negative (:math:`d_{\\text{cc,min}} <",
            "w_{\\text{WP,max}}`), the winding packs would intersect and the design is",
            "infeasible (score = 0).",
            "",
            "5. Per-turn force and torque",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            "",
            "Once :math:`N_{\\text{turns},i}` is known for each coil, the engineering-relevant",
            "structural loads are the per-turn quantities:",
            "",
            ".. math::",
            "",
            r"   F_{\text{turn},i} = \frac{F_{\text{reactor},i}}{N_{\text{turns},i}},"
            r"   \qquad"
            r"   \tau_{\text{turn},i} = \frac{\tau_{\text{reactor},i}}{N_{\text{turns},i}}",
            "",
            "The leaderboard reports:",
            "",
            "- :math:`F_{\\text{turn}}` \u2014 :math:`\\max_i F_{\\text{turn},i}` (MN/m), the",
            "  maximum per-turn force across all coils.",
            "- :math:`\\tau_{\\text{turn}}` \u2014 :math:`\\max_i \\tau_{\\text{turn},i}` (MN), the",
            "  maximum per-turn torque across all coils.",
            "",
            "These replace the single-turn :math:`F_{\\max}` and :math:`\\tau_{\\max}` in the",
            "reactor-scale leaderboard, since the single-turn values are not physically",
            "meaningful for a multi-turn winding pack.",
            "",
        ])

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
            # Build header columns: Score, metrics, then Date, User, IC, # at the end
            surface_header_cols = [r":math:`\text{Score}`"]  # Composite score column first
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
                
                # Build row: Score, metrics, then Date, User, i, f, and plot links at the end
                row_parts = []
                cs = entry.get("composite_score")
                row_parts.append(f"{cs:.3f}" if cs is not None else "—")
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
    
    # Sort entries within each surface by composite_score descending (higher = better)
    # Fall back to score_primary ascending for legacy entries
    def _surface_sort_key(e):
        cs = e.get("composite_score")
        if cs is not None:
            return (1, cs)
        sp = e.get("score_primary")
        if sp is not None:
            return (0, -sp)
        return (-1, 0)

    for surface, surf_data in surface_leaderboards.items():
        entries = surf_data["entries"]
        entries.sort(key=_surface_sort_key, reverse=True)
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
        exclude_fields = _DEVICE_LEADERBOARD_EXCLUDE
        
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
            "final_squared_flux",  # Primary metric (new name)
            "final_normalized_squared_flux",  # Primary metric (legacy name)
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
            header_cols = ["#", "Score", "User", "Date"]
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
                
                cs = entry.get("composite_score")
                score_str = f"{cs:.3f}" if cs is not None else "—"
                row_parts = [
                    str(entry.get("rank", "-")),
                    score_str,
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


# Friendly surface display names (used by multiple leaderboard writers).
_SURFACE_DISPLAY_NAMES: Dict[str, str] = {
    "LandremanPaul2021_QA": "Landreman-Paul QA",
    "LandremanPaul2021_QH_reactorScale_lowres": "Landreman-Paul QH",
    "circular_tokamak": "Circular Tokamak",
    "W7-X_without_coil_ripple_beta0p05_d23p4_tm": "W7-X",
    "HSX_QHS_mn1824_ns101": "HSX",
    "cfqs_2b40": "CFQS",
    "rotating_ellipse": "Rotating Ellipse",
    "c09r00_B_axis_half_tesla_NCSX.focus": "0.5 Tesla NCSX Design",
    "c09r00_B_axis_half_tesla_NCSX": "0.5 Tesla NCSX Design",
    "muse.focus": "MUSE",
    "muse": "MUSE",
    "wout_schuetthenneberg_nfp2.nc": "Schuett-Henneberg QA",
    "wout_schuetthenneberg_nfp2": "Schuett-Henneberg QA",
}


def _surface_display_name(surface_name: str) -> str:
    """Return a human-friendly display name for a plasma surface."""
    if surface_name in _SURFACE_DISPLAY_NAMES:
        return _SURFACE_DISPLAY_NAMES[surface_name]
    base = surface_name.replace("input.", "").replace(".focus", "")
    if base in _SURFACE_DISPLAY_NAMES:
        return _SURFACE_DISPLAY_NAMES[base]
    return surface_name.replace("_", " ").title()


# Reactor-scale metrics to display, in order.
_REACTOR_SCALE_DISPLAY_ORDER: list[str] = [
    "avg_BdotN_over_B",
    "reactor_scale_min_cs_separation",
    "reactor_scale_min_cc_separation",
    "reactor_scale_total_length",
    "total_superconductor_length_km",
    "reactor_scale_max_curvature",
    "reactor_scale_average_curvature",
    "reactor_scale_mean_squared_curvature",
    "per_turn_max_force",
    "per_turn_max_torque",
    "max_winding_pack_width",
]

# Internal reactor-scale keys that should NOT be shown as columns
_REACTOR_SCALE_EXCLUDE: set[str] = {
    "reference",
    "scaling_factors",
    "reactor_scale_force_per_coil_MN_per_m",  # list, not a scalar
    "N_turns_per_coil",                        # list, shown as dedicated column
    "N_turns_force",                           # internal detail (force-based turns)
    "N_turns_jc",                              # internal detail (Jc-based turns)
    "jc_model",                                # internal model parameters dict
    "winding_pack_width_per_coil",             # list, shown via max_winding_pack_width
    "finite_build_cc_clearance",               # derived diagnostic (d_cc - w_max)
    "force_limit_MN_per_m",                    # constant, not a result
    "N_turns_required",                        # legacy, superseded
    "reactor_scale_max_max_coil_force",        # single-turn; replaced by per_turn_max_force
    "reactor_scale_max_max_coil_torque",       # single-turn; replaced by per_turn_max_torque
    "reactor_scale_avg_max_coil_force",        # avg not needed
    "reactor_scale_avg_max_coil_torque",       # avg not needed
    "reactor_scale_arclength_variation",       # constraint retained; column removed
    "reactor_scale_squared_flux",              # replaced by avg_BdotN_over_B
    "error",                                   # legacy error message from old backfills
}


def _generate_score_vs_time_plot(
    surface_leaderboards: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> Path | None:
    """Generate a best-score-over-time plot as a PNG.

    For each surface, collects (run_date, composite_score) pairs from all
    entries, sorts by date, computes the running-best score, and plots the
    envelope.  Returns the path to the written PNG, or *None* on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime as _dt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False

    for surface_name, data in sorted(surface_leaderboards.items()):
        entries = data.get("entries", [])
        points: list[tuple[_dt, float]] = []
        for e in entries:
            cs = e.get("composite_score")
            rd = e.get("run_date", "")
            if cs is None or not rd:
                continue
            try:
                # Handle ISO date with or without time
                if "T" in rd:
                    dt = _dt.fromisoformat(rd.replace("Z", "+00:00").split("+")[0])
                else:
                    dt = _dt.strptime(rd.split("T")[0], "%Y-%m-%d")
                points.append((dt, float(cs)))
            except (ValueError, TypeError):
                continue

        if not points:
            continue

        # Sort by date and compute running best
        points.sort(key=lambda p: p[0])
        dates = [p[0] for p in points]
        scores = [p[1] for p in points]
        best_so_far: list[float] = []
        running_best = -float("inf")
        for s in scores:
            running_best = max(running_best, s)
            best_so_far.append(running_best)

        display = _surface_display_name(surface_name)
        ax.plot(mdates.date2num(dates), best_so_far, "o-", markersize=3, label=display, alpha=0.8)
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Run date")
    ax.set_ylabel("Best composite score")
    ax.set_title("Best Reactor-Scale Score Over Time")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "score_vs_time.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def write_reactor_scale_leaderboard(
    leaderboard: Dict[str, Any],
    surface_leaderboards: Dict[str, Dict[str, Any]],
    out_rst: Path,
    repo_root: Path | None = None,
) -> None:
    """Write a reactor-scale leaderboard RST file with per-surface tables.

    Each table shows reactor-scale engineering metrics (MN/m forces,
    curvatures in 1/m, etc.) alongside the composite score and constraint
    status.  These are the values that matter for assessing whether a
    design is viable at the ARIES-CS reference scale.
    """

    def _rs_format(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, (dict, list)):
            return "—"
        if isinstance(value, str):
            return "—"
        try:
            v = float(value)
        except (ValueError, TypeError):
            return "—"
        if abs(v) < 1e-100:
            return "0"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.2e}"

    def _get_rs_keys(entries: list[Dict[str, Any]]) -> list[str]:
        """Collect reactor-scale metric keys present in entries, in display order.

        Keys may live in either ``reactor_scale_metrics`` or device-scale
        ``metrics`` (e.g. ``avg_BdotN_over_B`` is dimensionless and stored
        at device scale only).
        """
        available: set[str] = set()
        for e in entries:
            rs = e.get("reactor_scale_metrics") or {}
            ms = e.get("metrics") or {}
            for k in rs:
                if k not in _REACTOR_SCALE_EXCLUDE:
                    available.add(k)
            # Also include device-scale metrics that are in the display order
            for k in _REACTOR_SCALE_DISPLAY_ORDER:
                if k in ms:
                    available.add(k)
        ordered = [k for k in _REACTOR_SCALE_DISPLAY_ORDER if k in available]
        # Append any remaining keys not in the predefined order
        for k in sorted(available - set(ordered)):
            ordered.append(k)
        return ordered

    lines: list[str] = [
        "Reactor-Scale Leaderboard",
        "=========================",
        "",
        ".. role:: red",
        ".. role:: orange",
        "",
        ".. raw:: html",
        "",
        "   <style>",
        "   .red { color: #dc3545; font-weight: bold; }",
        "   .orange { color: #e67e22; font-weight: bold; }",
        "   </style>",
        "",
        "All values are scaled to the **ARIES-CS reference** "
        "(major radius :math:`R_0 = 7.5` m, on-axis field :math:`B_0 = 5.7` T).",
        "",
        "Entries are ranked by **composite score** (higher = better engineering margin). "
        "See :doc:`metric_definitions` for constraint bounds and the scoring formula.",
        "",
        "How constraints are applied",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "",
        "**Hard constraints** make a design *infeasible*.  Any hard-constraint violation "
        "sets the composite score to **0** and marks the entry **FAIL**.  Hard constraints "
        "test topological validity (coils must encircle the plasma, coils must not "
        "interlink) and engineering limits on the winding-pack turns.",
        "",
        "**Soft constraints** encode engineering preferences.  Each soft constraint "
        "contributes an exponential margin factor to the composite score "
        "(see :doc:`metric_definitions`).  A soft-constraint violation lowers the score "
        "below 1 but does **not** cause FAIL or exclusion.  Violated soft-constraint "
        "cells are highlighted in :orange:`orange`; hard-constraint violations appear "
        "in :red:`red`.",
        "",
    ]

    # ---- Build a summary constraints table from REACTOR_SCALE_CONSTRAINTS ----
    lines.extend([
        "Engineering Constraints",
        "-----------------------",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: auto",
        "",
        "   * - Constraint",
        "     - Bound",
        "     - Type",
    ])
    for c in REACTOR_SCALE_CONSTRAINTS:
        label = c["label"]
        bound = c["bound"]
        units = c.get("units", "")
        direction = c["direction"]
        hard = c.get("hard", False)
        ctype = "hard" if hard else "soft"

        if direction == "eq":
            bound_str = f"= {bound}"
        elif direction == "max":
            bound_str = f"≤ {bound}"
        elif direction == "min":
            bound_str = f"≥ {bound}"
        else:
            bound_str = str(bound)
        if units and units != "(boolean)":
            bound_str += f" {units}"

        lines.append(f"   * - {label}")
        lines.append(f"     - {bound_str}")
        lines.append(f"     - {ctype}")

    lines.extend(["", ""])

    # Iterate over surfaces
    for surface_name, surf_data in sorted(surface_leaderboards.items()):
        entries = surf_data.get("entries", [])
        display_name = _surface_display_name(surface_name)
        lines.extend([
            display_name,
            "-" * len(display_name),
            "",
        ])

        if not entries:
            lines.extend(["No submissions for this surface.", ""])
            continue

        rs_keys = _get_rs_keys(entries)
        if not rs_keys:
            lines.extend(["No reactor-scale data available for this surface.", ""])
            continue

        # Resolve repo_root for visualization links
        resolved_repo_root = repo_root
        if resolved_repo_root is None:
            # out_rst is docs/leaderboard/reactor_scale.rst → go up 3 levels
            resolved_repo_root = Path(out_rst.parent.parent.parent).resolve()

        # Build header — metric symbol + units in a single :math: element
        header_cols = [
            r":math:`\text{Score}`",
            r":math:`N`",
            r":math:`n`",
        ]
        for k in rs_keys:
            shorthand = _metric_shorthand(k)
            math_sh = _shorthand_to_math(shorthand)
            unit_math = _RS_UNITS.get(k)
            if unit_math:
                # Inject unit inside the closing backtick:
                # `:math:`X`` → `:math:`X\ [\text{unit}]``
                math_sh = math_sh[:-1] + r"\ [" + unit_math + r"]`"
            header_cols.append(math_sh)
        header_cols.extend([
            r":math:`\text{LN}`",
            r":math:`\max_i N_{\text{turns}}`",
            r":math:`\text{User}`",
            r":math:`\text{i}`",
            r":math:`\text{f}`",
            r":math:`\text{PP}`",
        ])

        lines.append(f".. list-table:: {display_name} — Reactor Scale")
        lines.append("   :header-rows: 1")
        lines.append("   :widths: auto")
        lines.append("")

        # Header row
        lines.append("   * - " + header_cols[0])
        for col in header_cols[1:]:
            lines.append("     - " + col)

        # Data rows
        github_base_url = "https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main"
        for entry in entries:
            rs = entry.get("reactor_scale_metrics") or {}
            metrics = entry.get("metrics") or {}
            cs = entry.get("composite_score")
            score_str = f"{cs:.3f}" if cs is not None else "—"

            # Constraint violation sets (still used for cell highlighting)
            violations = entry.get("constraint_violations", [])
            hard_violated = [v for v in violations if v.get("hard")]
            soft_violated = [v for v in violations if not v.get("hard")]
            hard_metric_set: set = {v["metric"] for v in hard_violated}
            soft_metric_set: set = {v["metric"] for v in soft_violated}

            # N (num_coils) and n (coil_order) from device-scale metrics
            n_coils_val = metrics.get("num_coils")
            n_coils_str = str(int(round(float(n_coils_val)))) if n_coils_val is not None else "—"
            c_order_val = metrics.get("coil_order")
            c_order_str = str(int(round(float(c_order_val)))) if c_order_val is not None else "—"

            row = [score_str, n_coils_str, c_order_str]
            for k in rs_keys:
                # Look in reactor_scale_metrics first, fall back to metrics
                raw_val = rs.get(k)
                if raw_val is None:
                    raw_val = metrics.get(k)
                val_str = _rs_format(raw_val)
                if k in hard_metric_set:
                    val_str = f":red:`{val_str}`"
                elif k in soft_metric_set:
                    val_str = f":orange:`{val_str}`"
                row.append(val_str)

            # LN column (from device-scale metrics)
            ln_val = metrics.get("final_linking_number")
            if ln_val is not None:
                ln_str = str(int(round(float(ln_val))))
            else:
                ln_str = "—"
            if "final_linking_number" in hard_metric_set:
                ln_str = f":red:`{ln_str}`"
            row.append(ln_str)

            # N_turns_per_coil column — show max only
            n_turns = rs.get("N_turns_per_coil")
            if isinstance(n_turns, list) and n_turns:
                n_turns_str = str(max(n_turns))
            else:
                n_turns_str = "—"
            if "N_turns_per_coil" in hard_metric_set:
                n_turns_str = f":red:`{n_turns_str}`"
            row.append(n_turns_str)

            row.append(entry.get("contact", entry.get("method_name", "?"))[:15])

            # ---- Visualization link columns: i (initial), f (final), PP (Poincaré) ----
            rank_num = str(entry.get("rank", "-"))
            entry_path = entry.get("path", "")
            if entry_path.startswith("/"):
                entry_path = entry_path[1:]
            i_link = "—"
            f_link = "—"
            poincare_link = "—"

            if entry_path:
                path_obj = Path(entry_path)
                submission_dir = None
                if path_obj.name == "all_files.zip":
                    submission_dir = path_obj.parent
                elif path_obj.suffix == ".zip":
                    submission_dir = path_obj.parent
                else:
                    submission_dir = path_obj.parent

                if submission_dir:
                    if submission_dir.is_absolute():
                        try:
                            submission_dir = submission_dir.relative_to(resolved_repo_root.resolve())
                        except ValueError:
                            submission_str = str(submission_dir)
                            if "submissions" in submission_str:
                                idx = submission_str.find("submissions")
                                submission_dir = Path(submission_str[idx:])
                            else:
                                submission_dir = None

                    if submission_dir:
                        sd_str = str(submission_dir).replace("\\", "/").lstrip("/")
                        if sd_str.startswith("./"):
                            sd_str = sd_str[2:]
                        sd_str = sd_str.lstrip("/")
                        submission_dir = Path(sd_str)

                        full_sd = (resolved_repo_root / submission_dir).resolve()

                        # Fourier continuation check
                        fourier_orders_str = metrics.get("fourier_continuation_orders")
                        is_fc = fourier_orders_str and fourier_orders_str != "—"
                        orders: list[int] = []

                        if is_fc and isinstance(fourier_orders_str, str):
                            try:
                                orders = [int(o.strip()) for o in fourier_orders_str.split(",")]
                            except (ValueError, AttributeError):
                                orders = []
                            if orders:
                                order_dirs = []
                                for order in orders:
                                    od = full_sd / f"order_{order}"
                                    if od.exists() and od.is_dir():
                                        order_dirs.append((order, f"order_{order}"))
                                if order_dirs:
                                    first_order, first_od = order_dirs[0]
                                    init_pdf = submission_dir / first_od / "bn_error_3d_plot_initial.pdf"
                                    if (resolved_repo_root / init_pdf).exists():
                                        url = f"{github_base_url}/{str(init_pdf).replace(chr(92), '/')}"
                                        i_link = f"`{rank_num} <{url}>`__"
                                    f_links = []
                                    for order, od_name in order_dirs:
                                        fp = submission_dir / od_name / "bn_error_3d_plot.pdf"
                                        if (resolved_repo_root / fp).exists():
                                            url = f"{github_base_url}/{str(fp).replace(chr(92), '/')}"
                                            f_links.append(f"`{order} <{url}>`__")
                                    if f_links:
                                        f_link = " ".join(f_links)
                        else:
                            pdf_final = submission_dir / "bn_error_3d_plot.pdf"
                            pdf_init = submission_dir / "bn_error_3d_plot_initial.pdf"
                            if (resolved_repo_root / pdf_final).exists():
                                url = f"{github_base_url}/{str(pdf_final).replace(chr(92), '/')}"
                                f_link = f"`{rank_num} <{url}>`__"
                            if (resolved_repo_root / pdf_init).exists():
                                url = f"{github_base_url}/{str(pdf_init).replace(chr(92), '/')}"
                                i_link = f"`{rank_num} <{url}>`__"

                        # Poincaré plot
                        poincare_dirs = []
                        if is_fc and orders:
                            highest = max(orders)
                            poincare_dirs.append(submission_dir / f"order_{highest}")
                            poincare_dirs.append(submission_dir / f"order_{highest}" / "post_processing")
                        poincare_dirs.append(submission_dir)
                        poincare_dirs.append(submission_dir / "post_processing")
                        for pd in poincare_dirs:
                            pp_path = pd / "poincare_plot.png"
                            if (resolved_repo_root / pp_path).exists():
                                url_path = str(pp_path).replace("\\", "/").lstrip("/")
                                if url_path.startswith("./"):
                                    url_path = url_path[2:]
                                url = f"{github_base_url}/{url_path}"
                                poincare_link = f"`{rank_num} <{url}>`__"
                                break

            row.extend([i_link, f_link, poincare_link])

            lines.append("   * - " + row[0])
            for val in row[1:]:
                lines.append("     - " + val)

        lines.extend(["", ""])

    # ---- Score-vs-time plot ----
    plot_path = _generate_score_vs_time_plot(surface_leaderboards, out_rst.parent)
    if plot_path is not None:
        rel_plot = plot_path.name
        lines.extend([
            "Best Score Over Time",
            "--------------------",
            "",
            f".. image:: {rel_plot}",
            "   :width: 100%",
            "   :alt: Best composite score over time per surface",
            "",
        ])

    # Footer
    lines.extend([
        ".. note::",
        "   Last updated: run ``stellcoilbench update-db`` to refresh locally.",
        "",
    ])

    out_rst.parent.mkdir(parents=True, exist_ok=True)
    out_rst.write_text("\n".join(lines))


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

    # Write separate reactor-scale leaderboard.
    # This view should include ALL entries with reactor-scale data, even those
    # excluded from the main leaderboard for constraint violations — it's a
    # diagnostic/engineering view, not a ranking.
    all_entries_leaderboard = {
        "entries": (leaderboard.get("entries") or [])
                   + (leaderboard.get("excluded_entries") or []),
    }
    rs_surface_leaderboards = build_surface_leaderboards(
        all_entries_leaderboard, submissions_root, plasma_surfaces_dir
    )
    write_reactor_scale_leaderboard(
        leaderboard, rs_surface_leaderboards, docs_dir / "leaderboard" / "reactor_scale.rst",
        repo_root=repo_root,
    )
    
    print(f"Generated {len(surface_names)} surface leaderboard files: {sorted(surface_names)}", file=sys.stderr)

