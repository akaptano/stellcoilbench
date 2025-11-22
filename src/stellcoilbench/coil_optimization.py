from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .config_scheme import CaseConfig


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
    case_path: Path,
    coils_out_path: Path,
    case_cfg: CaseConfig | None = None,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Run a coil optimization for a given case using parameters from case.yaml,
    and write the resulting coils file to coils_out_path.

    This is the main extension point where you hook in SIMSOPT/REGCOIL/etc.

    Parameters
    ----------
    case_path:
        Path to case directory containing case.yaml and geometry files.
    coils_out_path:
        Where to write the coil geometry file (JSON format).
    case_cfg:
        Optional CaseConfig object. If None, loads from case_path / "case.yaml".
    output_dir:
        Optional directory where VTK files and other optimization outputs will be saved.
        If None, uses the directory containing coils_out_path.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing optimization results/metrics from the optimizer.

    Notes
    -----
    - The benchmark repository doesn't need to know the details of your optimizer; it just
      calls this function.
    - You can dispatch on `optimizer_params["algorithm"]` to different backends.
    """
    from simsopt.geo import SurfaceRZFourier
    from simsopt.util import optimize_coils_simple
    from simsopt.field import regularization_circ
    from simsopt import save
    from .evaluate import load_case_config
    
    if case_cfg is None:
        case_cfg = load_case_config(case_path)
    
    coil_params = dict(case_cfg.coils_params)
    optimizer_params = dict(case_cfg.optimizer_params)
    surface_params = dict(case_cfg.surface_params)
    
    # Remove verbose from coil_params if it's in optimizer_params (avoid duplicate)
    if "verbose" in optimizer_params and "verbose" in coil_params:
        coil_params = {k: v for k, v in coil_params.items() if k != "verbose"}
    
    # Handle surface file path - check if it's relative to plasma_surfaces directory
    surface_file = surface_params["surface"]
    if not Path(surface_file).is_absolute():
        # Try relative to case_path first, then plasma_surfaces
        potential_paths = [
            case_path / surface_file,
            Path("plasma_surfaces") / surface_file,
            Path.cwd() / "plasma_surfaces" / surface_file,
        ]
        for path in potential_paths:
            if path.exists():
                surface_file = str(path)
                break
    
    # Load surface based on file type
    if "input" in surface_file:
        surface_func = SurfaceRZFourier.from_vmec_input
    elif "wout" in surface_file:
        surface_func = SurfaceRZFourier.from_wout
    else:
        raise ValueError(f"Unknown surface type: {surface_file}")

    surface = surface_func(
        filename=surface_file, 
        range=surface_params["range"],
        nphi=surface_params["nphi"], 
        ntheta=surface_params["ntheta"])

    coil_radius = float(coil_params.pop("coil_radius", 0.05))
    coil_regularization = regularization_circ(coil_radius)
    
    # Determine output directory for VTK files
    if output_dir is None:
        output_dir = coils_out_path.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to output directory to ensure VTK files are saved there
    # optimize_coils_simple may save files based on current working directory
    import os
    original_cwd = Path.cwd()
    
    # Convert surface_file to absolute path before changing directories
    if not Path(surface_file).is_absolute():
        surface_file = str(Path(surface_file).resolve())
    
    try:
        os.chdir(output_dir)
        
        # Pass output_dir to optimize_coils_simple for VTK file output
        # optimize_coils_simple saves VTK files to output_dir during optimization
        try:
            coils, results_dict = optimize_coils_simple(
                surface, 
                **coil_params, 
                **optimizer_params, 
                regularization=coil_regularization,
                output_dir=str(output_dir)
            )
        except TypeError:
            # Fallback if optimize_coils_simple doesn't accept output_dir parameter
            # Files will be saved to current directory (which is now output_dir)
            coils, results_dict = optimize_coils_simple(
                surface, 
                **coil_params, 
                **optimizer_params, 
                regularization=coil_regularization
            )
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)
    
    # Ensure output path has .json extension for JSON format
    if not str(coils_out_path).endswith('.json'):
        coils_out_path = coils_out_path.with_suffix('.json')
    
    # Save coils to JSON file (use absolute path to ensure correct location)
    abs_coils_path = coils_out_path if coils_out_path.is_absolute() else (output_dir / coils_out_path.name)
    save(coils, abs_coils_path)
    
    return results_dict