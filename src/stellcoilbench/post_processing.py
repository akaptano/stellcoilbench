"""
Post-processing utilities for coil optimization results.

This module provides functions to analyze optimized coil configurations,
including generating Poincaré plots, computing QFM surfaces, and evaluating
quasisymmetry metrics.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from simsopt import load
from simsopt.geo import SurfaceRZFourier
from simsopt.field import BiotSavart
try:
    from simsopt.util.permanent_magnet_helper_functions import make_qfm  # type: ignore
except ImportError:
    # Fallback import path
    try:
        from simsopt.util import make_qfm  # type: ignore
    except ImportError:
        raise ImportError(
            "make_qfm not found. Please ensure simsopt is installed with permanent magnet utilities."
        )
from simsopt.mhd.vmec import Vmec  # type: ignore
from simsopt.mhd import QuasisymmetryRatioResidual  # type: ignore
from simsopt.util.mpi import MpiPartition  # type: ignore
from simsopt.util import proc0_print
try:
    from simsopt.field.tracing import (
        compute_fieldlines,
        plot_poincare_data,
        LevelsetStoppingCriterion,
    )
    from simsopt.field.magneticfieldclasses import InterpolatedField
    from simsopt.geo import SurfaceClassifier
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
import json
import yaml


def load_coils_and_surface(
    coils_json_path: Path,
    case_yaml_path: Optional[Path] = None,
    plasma_surfaces_dir: Optional[Path] = None,
) -> Tuple[BiotSavart, SurfaceRZFourier]:
    """
    Load coils from JSON and plasma surface from case.yaml or VMEC input file.
    
    Parameters
    ----------
    coils_json_path : Path
        Path to coils JSON file (e.g., biot_savart_optimized.json or coils.json).
    case_yaml_path : Path, optional
        Path to case.yaml file. If None, tries to find it relative to coils_json_path.
    plasma_surfaces_dir : Path, optional
        Directory containing plasma surface files. Defaults to "plasma_surfaces".
    
    Returns
    -------
    Tuple[BiotSavart, SurfaceRZFourier]
        BiotSavart object containing coils and plasma surface.
    
    Raises
    ------
    FileNotFoundError
        If coils JSON or surface file cannot be found.
    ValueError
        If surface file type is not recognized.
    """
    # Load coils from JSON
    if not coils_json_path.exists():
        raise FileNotFoundError(f"Coils JSON file not found: {coils_json_path}")
    
    bfield = load(str(coils_json_path))
    
    # If loaded object is BiotSavart, use it directly
    # Otherwise, assume it's coils and create BiotSavart
    if isinstance(bfield, BiotSavart):
        coils = bfield.coils
    else:
        # Assume it's a list of coils
        coils = bfield if isinstance(bfield, list) else [bfield]
        bfield = BiotSavart(coils)
    
    # Find case YAML if not provided
    potential_case_paths: list[Path] = []
    if case_yaml_path is None:
        # Search in various locations, starting from coils JSON directory and going up
        # Start from coils JSON directory and search up the directory tree
        current_dir = coils_json_path.parent
        for _ in range(5):  # Search up to 5 levels up
            potential_case_paths.append(current_dir / "case.yaml")
            if current_dir.parent == current_dir:  # Reached root
                break
            current_dir = current_dir.parent
        
        # Also try cases directory based on JSON filename
        potential_case_paths.append(
            Path("cases") / coils_json_path.stem.replace("coils", "").replace("biot_savart", "").replace("_optimized", "").replace(".json", "") / "case.yaml"
        )
        
        # Search for case.yaml
        for path in potential_case_paths:
            if path.exists():
                case_yaml_path = path
                break
        
        # If still not found, search for case YAML files in the cases directory
        # that might reference the surface (based on path components)
        if case_yaml_path is None:
            cases_dir = None
            # Find cases directory by going up from coils JSON
            current_dir = coils_json_path.parent
            for _ in range(7):  # Search up to 7 levels
                potential_cases_dir = current_dir / "cases"
                if potential_cases_dir.exists():
                    cases_dir = potential_cases_dir
                    break
                if current_dir.parent == current_dir:
                    break
                current_dir = current_dir.parent
            
            if cases_dir is not None:
                # Extract surface name hint from the path (e.g., "LandremanPaul2021_QA" from submissions path)
                path_parts = coils_json_path.parts
                surface_hint = None
                for part in path_parts:
                    # Look for common surface name patterns
                    if any(name in part for name in ["Landreman", "HSX", "CFQS", "MUSE", "NCSX", "W7X", "tokamak", "ellipse"]):
                        surface_hint = part
                        break
                
                # Search for matching case YAML files
                for yaml_file in cases_dir.glob("*.yaml"):
                    try:
                        case_data = yaml.safe_load(yaml_file.read_text())
                        if case_data and isinstance(case_data, dict):
                            surface_in_case = case_data.get("surface_params", {}).get("surface", "")
                            # Check if this case references a surface matching our hint
                            if surface_hint and surface_hint.replace("_", "") in surface_in_case.replace("_", ""):
                                case_yaml_path = yaml_file
                                potential_case_paths.append(yaml_file)
                                break
                            # Also try matching based on common surface name pattern
                            elif surface_hint and surface_hint.replace("2021", "").replace("_", "").lower() in surface_in_case.replace("2021", "").replace("_", "").lower():
                                case_yaml_path = yaml_file
                                potential_case_paths.append(yaml_file)
                                break
                    except Exception:
                        continue
    
    # Load surface from case YAML
    if case_yaml_path is None or not case_yaml_path.exists():
        raise FileNotFoundError(
            f"Could not find case YAML. Searched: {potential_case_paths}"
        )
    
    with open(case_yaml_path, 'r') as f:
        case_data = yaml.safe_load(f)
    
    surface_params = case_data.get("surface_params", {})
    surface_file = surface_params.get("surface", "")
    surface_range = surface_params.get("range", "half period")
    
    if not surface_file:
        raise ValueError("No surface file specified in case.yaml")
    
    # Find surface file - search in multiple locations
    potential_surface_paths = []
    
    # If absolute path, use it directly
    if Path(surface_file).is_absolute():
        potential_surface_paths.append(Path(surface_file))
    else:
        # Search relative to case.yaml location (likely repo root or cases directory)
        if case_yaml_path:
            case_dir = case_yaml_path.parent
            potential_surface_paths.extend([
                case_dir / surface_file,
                case_dir / "plasma_surfaces" / surface_file,
            ])
            # Go up from case.yaml to find repo root
            current_dir = case_dir
            for _ in range(5):  # Search up to 5 levels
                potential_surface_paths.append(current_dir / "plasma_surfaces" / surface_file)
                if current_dir.parent == current_dir:  # Reached root
                    break
                current_dir = current_dir.parent
        
        # Search relative to coils JSON location
        potential_surface_paths.extend([
            coils_json_path.parent / surface_file,
            coils_json_path.parent / "plasma_surfaces" / surface_file,
        ])
        
        # Search from coils JSON up to repo root
        current_dir = coils_json_path.parent
        for _ in range(5):  # Search up to 5 levels
            potential_surface_paths.append(current_dir / "plasma_surfaces" / surface_file)
            if current_dir.parent == current_dir:  # Reached root
                break
            current_dir = current_dir.parent
        
        # Use provided plasma_surfaces_dir or search for it
        if plasma_surfaces_dir is not None:
            # Use the provided directory
            potential_surface_paths.append(plasma_surfaces_dir / surface_file)
        else:
            # Default plasma_surfaces directory (relative to current working directory)
            plasma_surfaces_dir = Path("plasma_surfaces")
            potential_surface_paths.extend([
                Path(surface_file),  # Relative to current working directory
                plasma_surfaces_dir / surface_file,
                Path.cwd() / plasma_surfaces_dir / surface_file,
            ])
        
        # Add case-insensitive variants (for files like MUSE.focus vs muse.focus)
        surface_file_lower = surface_file.lower()
        if surface_file != surface_file_lower:
            for base_path in [Path("plasma_surfaces"), Path.cwd() / "plasma_surfaces"]:
                potential_surface_paths.append(base_path / surface_file_lower)
            # Also try case-insensitive search in repo root plasma_surfaces
            if case_yaml_path:
                current_dir = case_yaml_path.parent
                for _ in range(5):
                    plasma_dir = current_dir / "plasma_surfaces"
                    if plasma_dir.exists():
                        potential_surface_paths.append(plasma_dir / surface_file_lower)
                    if current_dir.parent == current_dir:
                        break
                    current_dir = current_dir.parent
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in potential_surface_paths:
        path_str = str(path.resolve() if path.is_absolute() else path)
        if path_str not in seen:
            seen.add(path_str)
            unique_paths.append(path)
    
    surface_path = None
    for path in unique_paths:
        if path.exists():
            surface_path = path
            break
    
    # If still not found, try case-insensitive directory search
    if surface_path is None:
        # Find plasma_surfaces directory
        plasma_surfaces_dirs = []
        
        # Use provided plasma_surfaces_dir first
        if plasma_surfaces_dir is not None and plasma_surfaces_dir.exists():
            plasma_surfaces_dirs.append(plasma_surfaces_dir)
        
        # Search from case.yaml location
        if case_yaml_path:
            current_dir = case_yaml_path.parent
            for _ in range(5):
                plasma_dir = current_dir / "plasma_surfaces"
                if plasma_dir.exists() and plasma_dir not in plasma_surfaces_dirs:
                    plasma_surfaces_dirs.append(plasma_dir)
                if current_dir.parent == current_dir:
                    break
                current_dir = current_dir.parent
        
        # Also check default locations
        for default_dir in [Path("plasma_surfaces"), Path.cwd() / "plasma_surfaces"]:
            if default_dir.exists() and default_dir not in plasma_surfaces_dirs:
                plasma_surfaces_dirs.append(default_dir)
        
        # Search for case-insensitive match
        surface_file_lower = surface_file.lower()
        for plasma_dir in plasma_surfaces_dirs:
            for file in plasma_dir.iterdir():
                if file.name.lower() == surface_file_lower:
                    surface_path = file
                    break
            if surface_path:
                break
    
    if surface_path is None:
        raise FileNotFoundError(
            f"Surface file not found: {surface_file}. Searched: {unique_paths[:10]}..."  # Show first 10 paths
        )
    
    # Load surface based on file type
    surface_file_lower = str(surface_path).lower()
    if "input" in surface_file_lower:
        surface = SurfaceRZFourier.from_vmec_input(
            str(surface_path),
            range=surface_range,
            nphi=256,
            ntheta=256,
        )
    elif "wout" in surface_file_lower:
        surface = SurfaceRZFourier.from_wout(
            str(surface_path),
            range=surface_range,
            nphi=256,
            ntheta=256,
        )
    elif "focus" in surface_file_lower:
        surface = SurfaceRZFourier.from_focus(
            str(surface_path),
            range=surface_range,
            nphi=256,
            ntheta=256,
        )
    else:
        raise ValueError(f"Unknown surface file type: {surface_path}")
    
    # Set filename attribute so VMEC can find the original input file
    surface.filename = str(surface_path.resolve())
    
    return bfield, surface


def load_surface_with_range(
    surface_path: Path,
    surface_range: str = "full torus",
) -> SurfaceRZFourier:
    """
    Load a surface from file with a specified range.
    
    Parameters
    ----------
    surface_path : Path
        Path to surface file.
    surface_range : str, default="full torus"
        Range for surface loading ("full torus" or "half period").
    
    Returns
    -------
    SurfaceRZFourier
        Loaded surface with specified range.
    """
    surface_file_lower = str(surface_path).lower()
    if "input" in surface_file_lower:
        surface = SurfaceRZFourier.from_vmec_input(
            str(surface_path),
            range=surface_range,
            nphi=256,
            ntheta=256,
        )
    elif "wout" in surface_file_lower:
        surface = SurfaceRZFourier.from_wout(
            str(surface_path),
            range=surface_range,
            nphi=256,
            ntheta=256,
        )
    elif "focus" in surface_file_lower:
        surface = SurfaceRZFourier.from_focus(
            str(surface_path),
            range=surface_range,
            nphi=256,
            ntheta=256,
        )
    else:
        raise ValueError(f"Unknown surface file type: {surface_path}")
    
    return surface


def compute_qfm_surface(
    surface: SurfaceRZFourier,
    bfield: BiotSavart,
) -> SurfaceRZFourier:
    """
    Compute QFM (quasi-flux surface) from plasma surface and magnetic field.
    
    Parameters
    ----------
    surface : SurfaceRZFourier
        Plasma boundary surface.
    bfield : BiotSavart
        Magnetic field from coils.
    
    Returns
    -------
    SurfaceRZFourier
        QFM surface.
    """
    qfm_surf = make_qfm(surface, bfield, n_iters=100)
    return qfm_surf.surface


def run_vmec_equilibrium(
    qfm_surface: SurfaceRZFourier,
    vmec_input_path: Optional[Path] = None,
    mpi: Optional[Any] = None,  # type: ignore
    plasma_surfaces_dir: Optional[Path] = None,  # type: ignore
) -> Any:  # type: ignore
    """
    Run VMEC to compute equilibrium from QFM surface.
    
    Parameters
    ----------
    qfm_surface : SurfaceRZFourier
        QFM surface to use as VMEC boundary.
    vmec_input_path : Path, optional
        Path to VMEC input file. If None or if the file is not a VMEC input file
        (e.g., .focus file), uses a reference VMEC input file as a template.
    mpi : Any, optional
        MPI partition for parallel execution. If None, creates a single-process partition.
    plasma_surfaces_dir : Path, optional
        Directory containing plasma surface files. Used to find a reference VMEC input file.
    
    Returns
    -------
    Vmec
        VMEC equilibrium object.
    """
    if mpi is None:
        mpi = MpiPartition(ngroups=1)  # type: ignore
    
    # Check if we have a valid VMEC input file
    is_vmec_input_file = False
    template_vmec_path = None
    
    if vmec_input_path is not None and vmec_input_path.exists():
        vmec_input_path_str = str(vmec_input_path).lower()
        is_vmec_input_file = "input" in vmec_input_path_str or vmec_input_path_str.endswith(".input")
        if is_vmec_input_file:
            template_vmec_path = vmec_input_path
    
    # If no valid VMEC input file, find a reference VMEC input file to use as template
    if not is_vmec_input_file:
        # Search for a reference VMEC input file
        search_dirs = []
        if plasma_surfaces_dir:
            search_dirs.append(plasma_surfaces_dir)
        search_dirs.extend([
            Path("plasma_surfaces"),
            Path.cwd() / "plasma_surfaces",
        ])
        
        # Look for common VMEC input files
        reference_files = [
            "input.LandremanPaul2021_QA",
            "input.circular_tokamak",
            "input.HSX_QHS_mn1824_ns101",
            "input.cfqs_2b40",
        ]
        
        for search_dir in search_dirs:
            for ref_file in reference_files:
                potential_path = search_dir / ref_file
                if potential_path.exists():
                    template_vmec_path = potential_path
                    break
            if template_vmec_path:
                break
        
        if template_vmec_path is None:
            # Try to find any .input file
            for search_dir in search_dirs:
                if search_dir.exists():
                    for file in search_dir.iterdir():
                        if "input" in file.name.lower() and file.suffix == "":
                            template_vmec_path = file
                            break
                    if template_vmec_path:
                        break
        
        if template_vmec_path is None:
            raise ValueError(
                "Could not find a VMEC input file to use as template. "
                "VMEC requires an input file even when using a custom boundary surface."
            )
    
    # Load VMEC with the template input file
    equil = Vmec(str(template_vmec_path), mpi)
    # Replace the boundary with our QFM surface
    equil.boundary = qfm_surface
    equil.run()
    
    return equil


def compute_quasisymmetry(
    equil: Any,  # type: ignore
    helicity_m: int = 1,
    helicity_n: int = 0,
    ns: int = 50,
) -> Tuple[float, np.ndarray]:
    """
    Compute quasisymmetry metrics from VMEC equilibrium.
    
    Parameters
    ----------
    equil : Vmec
        VMEC equilibrium object.
    helicity_m : int, default=1
        Poloidal mode number for quasisymmetry.
    helicity_n : int, default=0
        Toroidal mode number for quasisymmetry.
    ns : int, default=50
        Number of radial surfaces to evaluate.
    
    Returns
    -------
    Tuple[float, np.ndarray]
        Average quasisymmetry error and radial profile.
    """
    radii = np.arange(0, 1.01, 1.01 / ns)
    qs = QuasisymmetryRatioResidual(
        equil,
        radii,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
    )
    
    qs_profile = qs.profile()
    qs_average = float(np.mean(qs_profile))
    
    return qs_average, qs_profile


def plot_boozer_surface(
    equil: Any,  # type: ignore
    output_path: Path,
    js: Optional[int] = None,
    dpi: int = 300,
) -> None:
    """
    Plot Boozer surface from VMEC equilibrium.
    
    Creates a 2x2 grid showing Boozer surfaces at s = 0, 0.25, 0.5, 1.0.
    
    Parameters
    ----------
    equil : Vmec
        VMEC equilibrium object.
    output_path : Path
        Where to save the plot.
    js : int, optional
        Deprecated. If provided, plots only that surface index (for backward compatibility).
        Otherwise, creates 2x2 grid at s = 0, 0.25, 0.5, 1.0.
    dpi : int, default=300
        Resolution for saved figure.
    """
    try:
        import booz_xform as bx
    except ImportError:
        raise ImportError(
            "booz_xform is required for Boozer surface plots. "
            "Install with: pip install booz-xform"
        )
    
    b2 = bx.Booz_xform()
    b2.read_wout(equil.output_file)
    b2.run()
    
    # If js is explicitly provided, use old behavior for backward compatibility
    if js is not None:
        fig_single = plt.figure(figsize=(10, 8))
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rc("font", size=18)  # Increased base font size
        ax_single = plt.gca()
        bx.surfplot(b2, js=js, fill=False)
        # Increase font sizes for axes labels, tick labels, and title
        ax_single.xaxis.label.set_fontsize(18)
        ax_single.yaxis.label.set_fontsize(18)
        ax_single.tick_params(labelsize=16)
        if ax_single.get_title():
            ax_single.set_title(ax_single.get_title(), fontsize=20)
        # Update colorbar if it exists
        for ax_cbar in fig_single.axes:
            if ax_cbar != ax_single:
                try:
                    ax_cbar.tick_params(labelsize=16)
                    label = ax_cbar.get_ylabel()
                    if label:
                        ax_cbar.set_ylabel(label, fontsize=18)
                    label = ax_cbar.get_xlabel()
                    if label:
                        ax_cbar.set_xlabel(label, fontsize=18)
                except Exception:
                    pass
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        return
    
    # Get maximum valid surface index from booz_xform
    # booz_xform uses 1-indexed surface indices (js ranges from 1 to nsurf)
    # If booz_xform's arrays have size N, max valid js is typically N-1 (conservative)
    vmec_nsurf = len(equil.wout.iotas) - 1  # type: ignore
    
    # Determine max_js from booz_xform's wout data
    max_js = max(1, vmec_nsurf - 1)  # Conservative default
    try:
        if hasattr(b2, 'wout') and hasattr(b2.wout, 'iotas'):
            # Array size tells us the maximum - use size - 1 to be safe
            array_size = len(b2.wout.iotas)
            max_js = max(1, array_size - 1)
    except Exception:
        pass
    
    # Sample 4 evenly spaced surfaces between first (1) and last (max_js)
    if max_js == 1:
        js_indices = [1, 1, 1, 1]
    else:
        js_indices = np.linspace(1, max_js, 4, dtype=int).tolist()
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc("font", size=18)  # Increased base font size
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    
    # Plot each surface with error handling
    for i, js_idx in enumerate(js_indices):
        ax = axes_flat[i]
        plt.sca(ax)
        try:
            bx.surfplot(b2, js=js_idx, fill=False)
        except (IndexError, ValueError) as e:
            # If index error, try progressively smaller indices
            error_str = str(e).lower()
            if "out of bounds" in error_str or "index" in error_str:
                for fallback_js in range(js_idx - 1, 0, -1):
                    try:
                        bx.surfplot(b2, js=fallback_js, fill=False)
                        js_idx = fallback_js
                        break
                    except Exception:
                        continue
                else:
                    bx.surfplot(b2, js=1, fill=False)
                    js_idx = 1
            else:
                raise
        
        # Increase font sizes for axes labels and tick labels
        ax.xaxis.label.set_fontsize(18)
        ax.yaxis.label.set_fontsize(18)
        ax.tick_params(labelsize=16)
        
        # Compute s value for this surface and set title with larger font
        s_val = (js_idx - 1) / max(1, max_js - 1) if max_js > 1 else 0.5
        ax.set_title(f's = {s_val:.2f}', fontsize=20)
    
    # Increase colorbar font sizes after all plots are created
    # Find colorbars by checking axes that aren't in our main subplot axes
    main_axes_set = set(axes_flat)
    for ax_fig in fig.axes:
        if ax_fig not in main_axes_set:
            # This is likely a colorbar axis
            try:
                ax_fig.tick_params(labelsize=16)
                # Update labels if they exist
                label = ax_fig.get_ylabel()
                if label:
                    ax_fig.set_ylabel(label, fontsize=18)
                label = ax_fig.get_xlabel()
                if label:
                    ax_fig.set_xlabel(label, fontsize=18)
            except Exception:
                pass
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_iota_profile(
    equil: Any,  # type: ignore
    output_path: Path,
    sign: int = 1,
    equil_original: Optional[Any] = None,  # type: ignore
    dpi: int = 300,
) -> None:
    """
    Plot rotational transform (iota) profile from VMEC equilibrium.
    
    Parameters
    ----------
    equil : Vmec
        VMEC equilibrium object (self-consistent solution).
    output_path : Path
        Where to save the plot.
    sign : int, default=1
        Sign to apply to iota (1 or -1).
    equil_original : Vmec, optional
        Original VMEC equilibrium object for comparison.
    dpi : int, default=300
        Resolution for saved figure.
    """
    # Access iota profile from VMEC output
    iotas = equil.wout.iotas[1:]  # type: ignore
    psi_s = np.linspace(
        0,
        len(iotas) * equil.ds,
        len(iotas)
    )
    
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc("font", size=15)
    
    # Plot original surface if provided
    if equil_original is not None:
        iotas_orig = equil_original.wout.iotas[1:]  # type: ignore
        psi_s_orig = np.linspace(
            0,
            len(iotas_orig) * equil_original.ds,
            len(iotas_orig)
        )
        plt.plot(psi_s_orig, sign * iotas_orig, 'b-', label='Original surface', linewidth=2)
    
    # Plot self-consistent solution
    plt.plot(psi_s, sign * iotas, 'rx', label='Self-consistent (QFM)', markersize=8)
    
    if equil_original is not None:
        plt.legend()
    
    plt.ylabel(r'rotational transform $\iota$')
    plt.xlabel('Normalized toroidal flux s')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_quasisymmetry_profile(
    qs_profile: np.ndarray,
    radii: np.ndarray,
    output_path: Path,
    qs_profile_original: Optional[np.ndarray] = None,
    radii_original: Optional[np.ndarray] = None,
    dpi: int = 300,
) -> None:
    """
    Plot quasisymmetry error profile.
    
    Parameters
    ----------
    qs_profile : np.ndarray
        Quasisymmetry error at each radius (self-consistent solution).
    radii : np.ndarray
        Normalized toroidal flux radii.
    output_path : Path
        Where to save the plot.
    qs_profile_original : np.ndarray, optional
        Original quasisymmetry error profile for comparison.
    radii_original : np.ndarray, optional
        Normalized toroidal flux radii for original profile.
    dpi : int, default=300
        Resolution for saved figure.
    """
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc("font", size=15)
    
    # Plot original surface if provided
    if qs_profile_original is not None and radii_original is not None:
        plt.semilogy(radii_original, qs_profile_original, 'b-', label='Original surface', linewidth=2)
    
    # Plot self-consistent solution
    plt.semilogy(radii, qs_profile, 'rx', label='Self-consistent (QFM)', markersize=8)
    
    if qs_profile_original is not None:
        plt.legend()
    
    plt.xlabel('Normalized toroidal flux')
    plt.ylabel('Two-term quasisymmetry error')
    plt.grid()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def trace_fieldlines(
    bfield: BiotSavart,
    surface: SurfaceRZFourier,
    output_path: Path,
    nfieldlines: int = 20,
    tmax: float = 50000,
    tol: float = 1e-12,
    n_phi_slices: int = 4,
    use_interpolated_field: bool = True,
    markersize: int = 1,
    comm: Optional[Any] = None,  # type: ignore
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Trace fieldlines and generate Poincaré plots.
    
    This function creates Poincaré plots by tracing magnetic field lines
    starting from points on the magnetic axis outward toward the plasma boundary.
    
    Parameters
    ----------
    bfield : BiotSavart
        Magnetic field from coils.
    surface : SurfaceRZFourier
        Plasma boundary surface.
    output_path : Path
        Where to save the Poincaré plot.
    nfieldlines : int, default=20
        Number of fieldlines to trace.
    tmax : float, default=10000
        Maximum integration time for fieldline tracing.
    tol : float, default=1e-10
        Tolerance for fieldline integration.
    n_phi_slices : int, default=4
        Number of toroidal angles at which to record Poincaré sections.
    use_interpolated_field : bool, default=True
        Whether to use InterpolatedField for faster tracing (recommended).
    comm : Any, optional
        MPI communicator for parallel tracing.
    dpi : int, default=300
        Resolution for saved figure.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'fieldlines_tys': Fieldline trajectories
        - 'fieldlines_phi_hits': Poincaré section data
        - 'phis': Toroidal angles used
    """
    if not TRACING_AVAILABLE:
        raise ImportError(
            "Fieldline tracing requires simsopt.field.tracing. "
            "Please ensure simsopt is installed with tracing capabilities."
        )
    
    # Set up initial fieldline starting points
    # Sample R0 between innermost and outermost point along phi = 0, Z = 0 line
    gamma = surface.gamma()  # Shape: (nphi, ntheta, 3)
    
    # Find phi index closest to 0
    # Surface uses normalized phi in [0, 1] representing [0, 2*pi/nfp]
    phi_normalized_0 = 0.0
    phi_normalized_values = surface.quadpoints_phi
    phi_idx = np.argmin(np.abs(phi_normalized_values - phi_normalized_0))
    
    # Get all points at phi = 0 (or closest to it)
    points_at_phi0 = gamma[phi_idx, :, :]  # Shape: (ntheta, 3)
    
    # Find points where Z ≈ 0 (within tolerance)
    z_tolerance = 0.01  # 1 cm tolerance
    z_near_zero_mask = np.abs(points_at_phi0[:, 2]) < z_tolerance
    
    if np.any(z_near_zero_mask):
        # Compute R = sqrt(X^2 + Y^2) for points where Z ≈ 0
        points_z0 = points_at_phi0[z_near_zero_mask]
        R_values = np.sqrt(points_z0[:, 0]**2 + points_z0[:, 1]**2)
        
        R_min = np.min(R_values)
        R_max = np.max(R_values)
    else:
        # Fallback: find point closest to Z = 0
        z_abs = np.abs(points_at_phi0[:, 2])
        closest_idx = np.argmin(z_abs)
        closest_point = points_at_phi0[closest_idx]
        R_closest = np.sqrt(closest_point[0]**2 + closest_point[1]**2)
        
        # Use a range around this point
        major_radius = surface.get_rc(0, 0)
        minor_radius_component = abs(surface.get_rc(1, 0))
        R_min = max(R_closest - minor_radius_component * 0.5, major_radius * 0.5)
        R_max = R_closest + minor_radius_component * 0.5
    
    # Sample R0 between innermost and outermost points
    # Stay slightly inside boundary to avoid starting on surface
    R_start = R_min * 1.01  # Slightly inside innermost point
    R_end = R_max * 0.99  # Slightly inside outermost point
    
    # Ensure R_start < R_end (safety check)
    if R_start >= R_end:
        # If they're too close, use a small range around the midpoint
        R_mid = (R_min + R_max) / 2.0
        R_range = max(R_max - R_min, R_min * 0.1)  # At least 10% of R_min
        R_start = R_mid - R_range * 0.4
        R_end = R_mid + R_range * 0.4
    
    R0 = np.linspace(R_start, R_end, nfieldlines)
    print(f"R0 values: {R0}")
    Z0 = np.zeros(nfieldlines)
    
    # Toroidal angles for Poincaré sections
    phis = [(i / n_phi_slices) * (2 * np.pi / surface.nfp) for i in range(n_phi_slices)]
    
    # Create surface classifier for stopping criteria
    sc_fieldline = SurfaceClassifier(surface, h=0.02, p=2)
    
    # Use interpolated field for faster tracing if requested
    if use_interpolated_field:
        proc0_print("Creating interpolated field for faster tracing...")
        # Determine bounds for interpolation
        gamma = surface.gamma()
        rs = np.linalg.norm(gamma[:, :, 0:2], axis=2)
        zs = gamma[:, :, 2]
        
        n = 20  # Grid resolution
        rrange = (np.min(rs), np.max(rs), n)
        phirange = (0, 2 * np.pi / surface.nfp, n * 2)
        zrange = (np.min(zs), np.max(zs), n // 2)
        if surface.stellsym:
            zrange = (0.0, np.max(zs), n // 2)
        
        # Skip function to avoid evaluating outside domain
        def skip(rs, phis, zs):
            rphiz = np.asarray([rs, phis, zs]).T.copy()
            dists = sc_fieldline.evaluate_rphiz(rphiz)
            skip_mask = list((dists < -0.05).flatten())
            return skip_mask
        
        # Create interpolated field
        bfield.set_points(surface.gamma().reshape((-1, 3)))
        bfield_interp = InterpolatedField(
            bfield,
            degree=2,
            rrange=rrange,
            phirange=phirange,
            zrange=zrange,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            skip=skip
        )
        bfield_interp.set_points(surface.gamma().reshape((-1, 3)))
        field_to_trace = bfield_interp
    else:
        field_to_trace = bfield
    
    # Compute fieldlines
    proc0_print(f"Tracing {nfieldlines} fieldlines...")
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        field_to_trace,
        R0,
        Z0,
        tmax=tmax,
        tol=tol,
        comm=comm,
        phis=phis,
        stopping_criteria=[
            LevelsetStoppingCriterion(sc_fieldline.dist),
        ],
    )
    
    # Generate Poincaré plot
    proc0_print("Generating Poincaré plot...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_poincare_data(
        fieldlines_phi_hits,
        phis,
        str(output_path),
        dpi=dpi,
        s=markersize,
        # surf=surface,
        aspect='equal',
    )
    
    return {
        'fieldlines_tys': fieldlines_tys,
        'fieldlines_phi_hits': fieldlines_phi_hits,
        'phis': phis,
    }


def run_post_processing(
    coils_json_path: Path,
    output_dir: Path,
    case_yaml_path: Optional[Path] = None,
    plasma_surfaces_dir: Optional[Path] = None,
    run_vmec: bool = True,
    helicity_m: int = 1,
    helicity_n: int = 0,
    ns: int = 50,
    plot_boozer: bool = True,
    plot_poincare: bool = True,
    nfieldlines: int = 20,
    mpi: Optional[Any] = None,  # type: ignore
) -> Dict[str, Any]:
    """
    Run complete post-processing pipeline.
    
    This function:
    1. Loads coils and plasma surface
    2. Generates Poincaré plot (if requested)
    3. Computes QFM surface
    4. Optionally runs VMEC equilibrium
    5. Computes quasisymmetry metrics
    6. Generates VMEC-dependent plots (Boozer, iota, quasisymmetry)
    
    Parameters
    ----------
    coils_json_path : Path
        Path to coils JSON file.
    output_dir : Path
        Directory where output files will be saved.
    case_yaml_path : Path, optional
        Path to case.yaml file.
    plasma_surfaces_dir : Path, optional
        Directory containing plasma surface files.
    run_vmec : bool, default=True
        Whether to run VMEC equilibrium calculation.
    helicity_m : int, default=1
        Poloidal mode number for quasisymmetry.
    helicity_n : int, default=0
        Toroidal mode number for quasisymmetry.
    ns : int, default=50
        Number of radial surfaces for quasisymmetry evaluation.
    plot_boozer : bool, default=True
        Whether to generate Boozer surface plot.
    plot_poincare : bool, default=True
        Whether to generate Poincaré plot.
    nfieldlines : int, default=20
        Number of fieldlines to trace for Poincaré plot.
    mpi : Any, optional
        MPI partition for parallel execution.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing post-processing results:
        - 'qfm_surface': QFM surface object
        - 'quasisymmetry_average': Average quasisymmetry error
        - 'quasisymmetry_profile': Radial quasisymmetry profile
        - 'vmec': VMEC equilibrium object (if run_vmec=True)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Load coils and surface
    proc0_print("Loading coils and plasma surface...")
    bfield, surface = load_coils_and_surface(
        coils_json_path,
        case_yaml_path=case_yaml_path,
        plasma_surfaces_dir=plasma_surfaces_dir,
    )
    
    # Generate Poincaré plot if requested (do this first, before QFM)
    if plot_poincare:
        try:
            proc0_print("Generating Poincaré plot...")
            # Load surface with 'full torus' range for Poincaré plots
            # Get the original surface file path
            if hasattr(surface, 'filename') and surface.filename:
                surface_path = Path(surface.filename)
                if surface_path.exists():
                    proc0_print("Loading surface with 'full torus' range for Poincaré plot...")
                    poincare_surface = load_surface_with_range(surface_path, surface_range="full torus")
                else:
                    # Fallback: use the original surface if path not found
                    proc0_print("Warning: Could not find surface file for full torus loading, using original surface")
                    poincare_surface = surface
            else:
                # Fallback: try to find surface file from case YAML
                if case_yaml_path and case_yaml_path.exists():
                    import yaml
                    try:
                        case_data = yaml.safe_load(case_yaml_path.read_text())
                        surface_file = case_data.get("surface_params", {}).get("surface", "")
                        if surface_file:
                            # Search for surface file
                            search_dirs = []
                            if plasma_surfaces_dir:
                                search_dirs.append(plasma_surfaces_dir)
                            search_dirs.extend([
                                Path("plasma_surfaces"),
                                Path.cwd() / "plasma_surfaces",
                                case_yaml_path.parent / "plasma_surfaces",
                            ])
                            
                            surface_path = None
                            for search_dir in search_dirs:
                                potential_path = search_dir / surface_file
                                if potential_path.exists():
                                    surface_path = potential_path
                                    break
                            
                            if surface_path:
                                proc0_print("Loading surface with 'full torus' range for Poincaré plot...")
                                poincare_surface = load_surface_with_range(surface_path, surface_range="full torus")
                            else:
                                proc0_print("Warning: Could not find surface file, using original surface")
                                poincare_surface = surface
                        else:
                            poincare_surface = surface
                    except Exception:
                        poincare_surface = surface
                else:
                    poincare_surface = surface
            
            poincare_results = trace_fieldlines(
                bfield,
                poincare_surface,
                output_dir / "poincare_plot.pdf",
                nfieldlines=nfieldlines,
                comm=mpi,
            )
            results['poincare_results'] = poincare_results
        except Exception as e:
            proc0_print(f"Warning: Poincaré plot generation failed: {e}")
            proc0_print("Skipping Poincaré plot.")
    
    # Compute QFM surface
    proc0_print("Computing QFM surface...")
    qfm_surface = compute_qfm_surface(surface, bfield)
    results['qfm_surface'] = qfm_surface
    
    # Save QFM surface as VTK file
    proc0_print("Saving QFM surface as VTK file...")
    qfm_vtk_path = output_dir / "qfm_surface"
    try:
        qfm_surface.to_vtk(str(qfm_vtk_path))
        proc0_print(f"Saved QFM surface to {qfm_vtk_path}.vts")
        results['qfm_vtk_path'] = str(qfm_vtk_path)
    except Exception as e:
        proc0_print(f"Warning: Failed to save QFM surface as VTK: {e}")
    
    # Compute B·n on plasma surface
    bfield.set_points(surface.gamma().reshape((-1, 3)))
    B = bfield.B()
    n = surface.unitnormal()
    # Reshape to match surface grid dimensions
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size
    B_reshaped = B.reshape((nphi, ntheta, 3))
    n_reshaped = n.reshape((nphi, ntheta, 3))
    BdotN = np.mean(np.abs(np.sum(B_reshaped * n_reshaped, axis=2)))
    BdotN_over_B = BdotN / np.mean(bfield.AbsB())
    
    proc0_print(f"B·n on plasma surface: {BdotN:.2e}")
    proc0_print(f"B·n/|B|: {BdotN_over_B:.2e}")
    
    results['BdotN'] = float(BdotN)
    results['BdotN_over_B'] = float(BdotN_over_B)
    
    # Run VMEC if requested
    if run_vmec:
        proc0_print("Running VMEC equilibrium...")
        try:
            # Get VMEC input path from original surface file
            # VMEC needs the original plasma surface input file - it will use the QFM boundary we set
            vmec_input_path = None
            
            # First, try using the surface's filename attribute (set during loading)
            if hasattr(surface, 'filename') and surface.filename:
                potential_path = Path(surface.filename)
                if potential_path.exists():
                    vmec_input_path = potential_path
            
            # If not found, find the original surface file from case YAML
            if (vmec_input_path is None or not vmec_input_path.exists()) and case_yaml_path:
                if not isinstance(case_yaml_path, Path):
                    case_yaml_path = Path(case_yaml_path)
                if case_yaml_path.exists():
                    import yaml
                    try:
                        case_data = yaml.safe_load(case_yaml_path.read_text())
                        surface_file = case_data.get("surface_params", {}).get("surface", "")
                        if surface_file:
                            # Search for the surface file in plasma_surfaces directories
                            search_dirs = []
                            if plasma_surfaces_dir:
                                search_dirs.append(plasma_surfaces_dir)
                            search_dirs.extend([
                                Path("plasma_surfaces"),
                                Path.cwd() / "plasma_surfaces",
                                case_yaml_path.parent / "plasma_surfaces",
                            ])
                            
                            # Also search relative to coils_json_path
                            coils_json_dir = coils_json_path.parent
                            for _ in range(5):  # Search up to 5 levels
                                potential_plasma_dir = coils_json_dir / "plasma_surfaces"
                                if potential_plasma_dir.exists() and potential_plasma_dir not in search_dirs:
                                    search_dirs.append(potential_plasma_dir)
                                if coils_json_dir.parent == coils_json_dir:
                                    break
                                coils_json_dir = coils_json_dir.parent
                            
                            for search_dir in search_dirs:
                                potential_path = search_dir / surface_file
                                if potential_path.exists():
                                    vmec_input_path = potential_path
                                    break
                    except Exception:
                        pass
            
            # If still not found, try searching relative to coils_json_path
            if (vmec_input_path is None or not vmec_input_path.exists()):
                # Try to extract surface filename from surface object if available
                if hasattr(surface, 'filename') and surface.filename:
                    potential_path = Path(surface.filename)
                    # Try resolving relative paths
                    if not potential_path.exists():
                        # Try relative to coils_json_path
                        coils_json_dir = coils_json_path.parent
                        for _ in range(5):
                            potential_path_rel = coils_json_dir / potential_path.name
                            if potential_path_rel.exists():
                                vmec_input_path = potential_path_rel
                                break
                            potential_path_rel = coils_json_dir / "plasma_surfaces" / potential_path.name
                            if potential_path_rel.exists():
                                vmec_input_path = potential_path_rel
                                break
                            if coils_json_dir.parent == coils_json_dir:
                                break
                            coils_json_dir = coils_json_dir.parent
                    elif potential_path.exists():
                        vmec_input_path = potential_path
            
            # Check if vmec_input_path is a valid VMEC input file
            # If not, run_vmec_equilibrium will use a template VMEC input file
            is_vmec_input = False
            if vmec_input_path is not None and vmec_input_path.exists():
                vmec_input_str = str(vmec_input_path).lower()
                is_vmec_input = "input" in vmec_input_str or vmec_input_str.endswith(".input")
            
            if not is_vmec_input:
                proc0_print("Note: Original surface file is not a VMEC input file.")
                proc0_print("Using a template VMEC input file and replacing boundary with QFM surface.")
            
            # Run VMEC for original surface first (for comparison plots)
            equil_original = None
            qs_average_original = None
            qs_profile_original = None
            radii_original = None
            
            # Try to run VMEC for original surface (for comparison)
            # Use VMEC input file if available, otherwise run_vmec_equilibrium will find a template
            try:
                proc0_print("Running VMEC for original plasma surface (for comparison)...")
                equil_original = run_vmec_equilibrium(
                    surface,  # Use original surface
                    vmec_input_path=vmec_input_path if is_vmec_input else None,
                    mpi=mpi,
                    plasma_surfaces_dir=plasma_surfaces_dir,
                )
                
                # Compute quasisymmetry for original surface
                qs_average_original, qs_profile_original = compute_quasisymmetry(
                    equil_original,
                    helicity_m=helicity_m,
                    helicity_n=helicity_n,
                    ns=ns,
                )
                radii_original = np.arange(0, 1.01, 1.01 / ns)
                proc0_print(f"Original surface average quasisymmetry error: {qs_average_original:.2e}")
            except Exception as e:
                proc0_print(f"Warning: Failed to compute original surface profiles: {e}")
                proc0_print("Proceeding with QFM surface only.")
            
            # Run VMEC for QFM surface (self-consistent solution)
            equil = run_vmec_equilibrium(
                qfm_surface,
                vmec_input_path=vmec_input_path if is_vmec_input else None,
                mpi=mpi,
                plasma_surfaces_dir=plasma_surfaces_dir,
            )
            results['vmec'] = equil
            
            # Compute quasisymmetry for QFM surface
            proc0_print("Computing quasisymmetry metrics...")
            qs_average, qs_profile = compute_quasisymmetry(
                equil,
                helicity_m=helicity_m,
                helicity_n=helicity_n,
                ns=ns,
            )
            results['quasisymmetry_average'] = float(qs_average)
            results['quasisymmetry_profile'] = qs_profile.tolist()
            
            proc0_print(f"Average quasisymmetry error: {qs_average:.2e}")
            
            # Always generate iota and quasisymmetry plots vs flux coordinate
            proc0_print("Generating iota profile plot vs flux coordinate...")
            # Determine sign based on helicity
            sign = -1 if helicity_n == -1 else 1
            plot_iota_profile(
                equil,
                output_dir / "iota_profile.png",
                sign=sign,
                equil_original=equil_original,
                dpi=300,  # High resolution for publication
            )
            
            proc0_print("Generating quasisymmetry profile plot vs flux coordinate...")
            radii = np.arange(0, 1.01, 1.01 / ns)
            plot_quasisymmetry_profile(
                qs_profile,
                radii,
                output_dir / "quasisymmetry_profile.png",
                qs_profile_original=qs_profile_original,
                radii_original=radii_original,
                dpi=300,  # High resolution for publication
            )
            
            # Generate Boozer plot if requested
            if plot_boozer:
                proc0_print("Generating Boozer surface plot (2x2 grid at s = 0, 0.25, 0.5, 1.0)...")
                plot_boozer_surface(
                    equil,
                    output_dir / "boozer_surface.png",
                    js=None,  # None triggers 2x2 grid at s = 0, 0.25, 0.5, 1.0
                    dpi=300,  # High resolution for publication
                )
        except Exception as e:
            proc0_print(f"Warning: VMEC calculation failed: {e}")
            proc0_print("Skipping VMEC-dependent post-processing.")
    
    # Save results to JSON
    results_json = {
        'BdotN': results.get('BdotN'),
        'BdotN_over_B': results.get('BdotN_over_B'),
        'quasisymmetry_average': results.get('quasisymmetry_average'),
    }
    
    with open(output_dir / "post_processing_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    proc0_print(f"Post-processing complete. Results saved to {output_dir}")
    
    return results
