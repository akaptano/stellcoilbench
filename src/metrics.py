from __future__ import annotations

from typing import Dict

import numpy as np

from .config_schema import CaseConfig


def normal_field_error(Bn: np.ndarray, B0: float) -> float:
    """
    Simple χ²-like normal-field error metric.

    Parameters
    ----------
    Bn:
        Array of B·n values on the plasma surface at a set of quadrature points.
    B0:
        Normalization field [Tesla]. Typically something like on-axis field.

    Returns
    -------
    float
        Dimensionless normal-field error, averaged over sample points.
    """
    Bn = np.asarray(Bn, dtype=float)
    if Bn.size == 0:
        raise ValueError("normal_field_error: Bn array is empty")
    return float(np.mean((Bn / B0) ** 2))


def composite_scores(metric_dict: Dict[str, float], case_cfg: CaseConfig) -> Dict[str, float]:
    """
    Combine raw metrics into one or more scalar scores.

    This is deliberately simple as a starting point:

    - If `case_cfg.scoring` is defined, you can implement case-specific logic.
    - Otherwise, define `score_primary` as a monotonically decreasing function
      of chi2_Bn, with a soft cutoff.

    You can/should refine this function as the benchmark spec matures.
    """
    chi2 = metric_dict.get("chi2_Bn")
    if chi2 is None:
        raise KeyError("composite_scores expected 'chi2_Bn' in metric_dict")

    # Example: map chi2 in [1e-6, 1e-2] into score_primary in [0, 1].
    chi2_clamped = float(np.clip(chi2, 1e-6, 1e-2))
    log_min, log_max = np.log10(1e-6), np.log10(1e-2)
    score = 1.0 - (np.log10(chi2_clamped) - log_min) / (log_max - log_min)
    score = float(np.clip(score, 0.0, 1.0))

    scores = {
        "score_primary": score,
        "chi2_Bn": chi2,
    }

    # Later: add penalties for curvature, constraints, etc. using case_cfg.scoring.
    return scores

