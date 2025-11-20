from __future__ import annotations

from typing import Dict

import numpy as np

from .config_scheme import CaseConfig
from simsopt.field import BiotSavart
from simsopt.geo import Surface


def normalized_normal_field_error(B : BiotSavart, s: Surface) -> float:
    """
    Simple χ²-like normal-field error metric.

    Parameters
    ----------
    B: Simsopt.field.BiotSavart
        Array of B·n values on the plasma surface at a set of quadrature points.
    s: Simsopt.geo.Surface
        Surface to evaluate the normal-field error on.

    Returns
    -------
    float
        Dimensionless normal-field error, averaged over sample points.
    """
    B.set_points(s.points)
    Bn = np.mean(np.array(B()).reshape(-1, 3) * np.array(s.gamma()).reshape(-1, 3), axis=1)
    B0 = np.mean(B.modB())
    return (Bn / B0)


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

