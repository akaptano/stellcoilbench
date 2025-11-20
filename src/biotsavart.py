from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_Bn_on_plasma_surface(case_dir: Path, coils_path: Path) -> np.ndarray:
    """
    Compute B·n on the plasma surface for a given case + coil set.

    This is the main physics-specific hook that you will want to replace with
    your own implementation (e.g. via SIMSOPT, your own Biot–Savart routines,
    or a DESC/SIMSOPT interface).

    Parameters
    ----------
    case_dir:
        Path to the case directory (contains at least case.yaml and plasma geometry).
    coils_path:
        Path to the coils file produced by your method.

    Returns
    -------
    np.ndarray
        1D array of B·n values at your chosen quadrature points on the plasma surface.

    Notes
    -----
    The benchmark *spec* should define a standard set of quadrature points and
    a file layout for plasma and coil geometry. This function is where those
    details are wired into the metric pipeline.
    """
    raise NotImplementedError(
        "compute_Bn_on_plasma_surface is not implemented yet. "
        "Wire this up to your Biot–Savart / equilibrium machinery."
    )

