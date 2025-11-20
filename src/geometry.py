from __future__ import annotations

from pathlib import Path
from typing import Dict

from simsopt.geo import CurveCurveDistance, CurveSurfaceDistance, LpCurveCurvature, LinkingNumber

def coil_complexity_metrics(case_dir: Path, coils_path: Path) -> Dict[str, float]:
    """
    Compute coil complexity / engineering metrics for a given case + coil set.

    Examples of metrics that belong here:
      - maximum curvature per coil
      - total coil length, possibly normalized
      - minimum coil–coil spacing
      - minimum coil–plasma spacing

    Parameters
    ----------
    case_dir:
        Path to the case directory (in case you need plasma geometry).
    coils_path:
        Path to the coils file produced by your method.

    Returns
    -------
    dict
        Mapping metric_name -> value (floats).

    Notes
    -----
    The exact implementation depends on your coil file format and chosen
    metrics. For initial prototyping, you can return an empty dict, and the
    benchmark will still be functional with just the normal-field error.
    """
    # TODO: implement actual geometry metrics. For now, return an empty dict.
    return {}

