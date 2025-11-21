from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_coils_config(config_path: Path) -> Dict[str, Any]:
    """
    Load a coils.yaml-style config into a dict.
    """
    import yaml

    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {config_path}, got {type(data)}")
    return data



def optimize_coils(
    case_dir: Path,
    coils_config: Dict[str, Any],
    coils_out_path: Path,
) -> None:
    """
    Run a coil optimization for a given case using parameters from coils_config,
    and write the resulting coils file to coils_out_path.

    This is the main extension point where you hook in SIMSOPT/REGCOIL/etc.

    Parameters
    ----------
    case_dir:
        Path to case directory containing case.yaml and geometry (plasma.h5, coil_region.h5, ...).
    coils_config:
        Parsed coils.yaml dict (see load_coils_config).
    coils_out_path:
        Where to write the coil geometry file (e.g. HDF5).

    Notes
    -----
    - The benchmark repository doesn't need to know the details of your optimizer; it just
      calls this function.
    - You can dispatch on `coils_config["optimizer"]` to different backends.
    """
    from simsopt.geo import SurfaceRZFourier
    from simsopt.util import optimize_coils_simple
    from simsopt import save
    coil_params = coils_config.get("coils_params", {}) or {}
    optimizer_params = coils_config.get("optimizer_params", {}) or {}
    surface_params = case_dir.get("surface_params", {}) or {}
    if surface_params["surface"][:5] == "input":
        surface = SurfaceRZFourier.from_vmec_input(
            filename=surface_params["surface"], nphi=surface_params["nphi"], ntheta=surface_params["ntheta"])
    elif surface_params["surface"][:4] == "wout":
        surface = SurfaceRZFourier.from_wout(
            filename=surface_params["surface"], nphi=surface_params["nphi"], ntheta=surface_params["ntheta"])
    else:
        raise ValueError(f"Unknown surface type: {surface_params['surface']}")
    coils, results_dict = optimize_coils_simple(surface, **coil_params, **optimizer_params)
    save(coils, coils_out_path / "_coils.json")