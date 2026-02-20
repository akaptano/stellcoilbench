from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from typing import Callable
from datetime import datetime
import zipfile
import os
import sys
from contextlib import contextmanager
from .config_scheme import CaseConfig
from .post_processing import timed_section, get_timing_results, suppress_output

from simsopt.geo import SurfaceRZFourier
try:
    from simsopt.field import regularization_circ
except ImportError:  # pragma: no cover - fallback for older simsopt
    regularization_circ = None

# MPI support for parallel post-processing
try:
    from simsopt.util import comm_world, proc0_print
except (ImportError, RuntimeError):
    # ImportError: simsopt not installed
    # RuntimeError: mpi4py installed but MPI library not available
    comm_world = None
    def proc0_print(*args, **kwargs):
        print(*args, **kwargs)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for PDF generation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore

# Virtual casing support (from simsopt.mhd)
try:
    from simsopt.mhd.virtual_casing import VirtualCasing
    VIRTUAL_CASING_AVAILABLE = True
except ImportError:
    VIRTUAL_CASING_AVAILABLE = False
    VirtualCasing = None  # type: ignore
    cm = None  # type: ignore
    Normalize = None  # type: ignore

# ARIES-CS reactor reference (matches post_processing vmec_RZ_scale scaling)
ARIES_CS_MINOR_RADIUS = 1.7  # meters (aspect ratio ~4.5)


class LinearPenalty:
    """
    Linear penalty function that implements max(objective - threshold, 0).
    
    This is used for l1_threshold options where we want a linear penalty
    above the threshold and zero below.
    """
    def __init__(self, objective, threshold: float):
        self.objective = objective
        self.threshold = threshold
        # Add simsopt compatibility attributes
        self._parent = None
        self._children = []
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying objective for simsopt compatibility."""
        # Only delegate if not already defined on this class
        if name in ['objective', 'threshold', '_parent', '_children', 'J', 'dJ', 'x']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.objective, name)
    
    def J(self):
        """Return max(J - threshold, 0)"""
        J_val = self.objective.J()
        return max(J_val - self.threshold, 0.0)
    
    def dJ(self, **kwargs):
        """Return gradient: dJ/dx if J > threshold, else 0"""
        J_val = self.objective.J()
        grad = self.objective.dJ(**kwargs)
        if J_val > self.threshold:
            return grad
        else:
            # Return zero gradient if below threshold
            # Multiply by 0 to preserve type and structure
            if isinstance(grad, np.ndarray):
                return grad * 0.0
            elif hasattr(grad, '__mul__'):
                return grad * 0.0
            else:
                # Fallback: return zeros with same shape as x
                try:
                    x_arr = np.asarray(self.x)
                    return np.zeros_like(x_arr)
                except (AttributeError, TypeError, ValueError):
                    return 0.0
    
    def __add__(self, other):
        """Allow addition with other objectives for sum() compatibility"""
        if type(other) is type(self):
            # Same class (use type identity to survive module reload in tests)
            combined = self.objective + other.objective
            return type(self)(combined, self.threshold)
        elif isinstance(other, (int, float)) and other == 0:
            # Allow sum() to start with 0
            return self
        return NotImplemented
    
    def __radd__(self, other):
        """Allow right addition for sum() compatibility"""
        if isinstance(other, (int, float)) and other == 0:
            return self
        return NotImplemented
    
    def __mul__(self, other):
        """Allow multiplication with Weight for compatibility"""
        from simsopt.objectives import Weight
        if isinstance(other, Weight):
            # Create a weighted version
            # Weight(2.0) * LinearPenalty(obj, thresh) should give:
            # 2.0 * max(obj - thresh, 0) = max(2.0 * obj - 2.0 * thresh, 0)
            # So we scale both the objective and threshold
            weighted_obj = other * self.objective
            # Extract weight value by comparing weighted vs unweighted objective values
            # This works because Weight(w) * obj gives w * obj.J()
            try:
                unweighted_J = self.objective.J()
                weighted_J = weighted_obj.J()
                if abs(unweighted_J) > 1e-10:
                    weight_val = weighted_J / unweighted_J
                else:
                    # If unweighted is zero, weight doesn't matter, use 1.0
                    weight_val = 1.0
                scaled_threshold = weight_val * self.threshold
            except (AttributeError, ZeroDivisionError, TypeError, ValueError):
                # Fallback: don't scale threshold if we can't determine weight
                # This can happen if objectives don't have J() method, division fails,
                # or other issues occur
                scaled_threshold = self.threshold
            return type(self)(weighted_obj, scaled_threshold)
        return NotImplemented
    
    def __rmul__(self, other):
        """Allow right multiplication with Weight"""
        return self.__mul__(other)
    
    def _add_child(self, child):
        """Add a child objective (simsopt compatibility)."""
        if child not in self._children:
            self._children.append(child)
            if hasattr(child, '_parent'):
                child._parent = self
    
    @property
    def x(self):
        """Get optimization variables"""
        return self.objective.x
    
    @x.setter
    def x(self, value):
        """Set optimization variables"""
        self.objective.x = value


def _get_scipy_algorithm_options(algorithm: str) -> Dict[str, list]:
    """
    Get valid options for a given scipy optimization algorithm.
    
    Returns a dictionary mapping option names to their valid types/values.
    Based on scipy.optimize.minimize documentation.

    Parameters
    ----------
    algorithm: str
        The name of the scipy optimization algorithm.

    Returns
    -------
    Dict[str, list]
        A dictionary mapping option names to their valid types/values.
    """
    # Common options for most algorithms
    common_options = {
        'maxiter': [int],
        'disp': [bool],
    }
    
    # Algorithm-specific options
    algorithm_specific = {
        'BFGS': {
            'gtol': [float],
            'norm': [float],
        },
        'L-BFGS-B': {
            'maxfun': [int],
            'ftol': [float],
            'gtol': [float],
            'eps': [float],
            'maxls': [int],
        },
        'SLSQP': {
            'ftol': [float],
            'eps': [float],
        },
        'Nelder-Mead': {
            'xatol': [float],
            'fatol': [float],
            'adaptive': [bool],
        },
        'Powell': {
            'xtol': [float],
            'ftol': [float],
            'maxfev': [int],
        },
        'CG': {
            'gtol': [float],
            'norm': [float],
        },
        'Newton-CG': {
            'xtol': [float],
            'eps': [float],
        },
        'TNC': {
            'maxfun': [int],
            'ftol': [float],
            'gtol': [float],
            'eps': [float],
        },
        'COBYLA': {
            'maxiter': [int],
            'rhobeg': [float],
            'tol': [float],
        },
        'trust-constr': {
            'xtol': [float],
            'gtol': [float],
            'barrier_tol': [float],
            'initial_barrier_parameter': [float],
            'initial_barrier_tolerance': [float],
            'initial_trust_radius': [float],
            'max_trust_radius': [float],
        },
    }
    
    # Combine common and algorithm-specific options
    options = common_options.copy()
    if algorithm in algorithm_specific:
        options.update(algorithm_specific[algorithm])
    
    return options


def _validate_algorithm_options(algorithm: str, options: Dict[str, Any]) -> None:
    """
    Validate that algorithm-specific options are valid for the given algorithm.
    
    Raises ValueError if invalid options are found.

    Parameters
    ----------
    algorithm: str
        The name of the scipy optimization algorithm.
    options: Dict[str, Any]
        A dictionary of algorithm-specific options to validate.

    Raises
    ------
    ValueError: If invalid options are found.
    """
    valid_options = _get_scipy_algorithm_options(algorithm)
    
    invalid_options = []
    for option_name, option_value in options.items():
        if option_name not in valid_options:
            invalid_options.append(option_name)
        else:
            # Check type
            valid_types = valid_options[option_name]
            if not any(isinstance(option_value, t) for t in valid_types):
                invalid_options.append(f"{option_name} (wrong type: {type(option_value).__name__})")
    
    if invalid_options:
        valid_option_names = ', '.join(sorted(valid_options.keys()))
        raise ValueError(
            f"Invalid algorithm options for '{algorithm}': {', '.join(invalid_options)}. "
            f"Valid options are: {valid_option_names}"
        )


def load_coils_config(config_path: Path) -> Dict[str, Any]:
    """
    Load a coils.yaml-style config into a dict.
    
    Note: This function is deprecated. Use CaseConfig.from_dict() instead,
    which includes validation via validate_case_config().

    Parameters
    ----------
    config_path: Path
        Path to the coils.yaml file.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the loaded coils configuration.

    Raises
    ------
    ValueError: If the config file is not a dictionary.
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
    surface_resolution: int = 32,
    skip_post_processing: bool = False,
    run_vmec: bool = False,
    run_simple: bool = False,
    plot_poincare: bool = False,
    plot_finite_build: bool = False,
    finite_build_width: Optional[float] = None,
    finite_build_height: Optional[float] = None,
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
    surface_resolution:
        Resolution of plasma surface (nphi=ntheta) for evaluation (default: 16).
        Lower values speed up optimization but reduce accuracy. Use 8 for faster unit tests.
    skip_post_processing:
        If True, skip post-processing (QFM, VMEC, Poincaré plots, etc.) after optimization.
        Useful for faster testing and debugging of optimization alone (default: False).
    run_vmec:
        If True, run QFM and VMEC equilibrium calculation during post-processing.
        This is expensive and disabled by default (default: False).
    run_simple:
        If True, run SIMPLE fast particle tracing during post-processing.
        Requires run_vmec=True. Disabled by default (default: False).
    plot_poincare:
        If True, generate Poincaré plot during post-processing (default: False).
    plot_finite_build:
        If True, generate finite-build coil VTK during post-processing (default: False).
    finite_build_width:
        Cross-section width [m] for finite-build coils. If None, uses default (5 cm).
    finite_build_height:
        Cross-section height [m] for finite-build coils. If None, uses default (5 cm).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing optimization results/metrics from the optimizer.

    Notes
    -----
    - The benchmark repository doesn't need to know the details of your optimizer; it just
      calls this function.
    - You can dispatch on `optimizer_params["algorithm"]` to different backends.
    - When running with MPI, coil optimization runs only on rank 0, while post-processing
      uses all MPI processes for VMEC and fieldline tracing.
    """
    from simsopt.geo import SurfaceRZFourier
    from simsopt import save
    from .evaluate import load_case_config
    
    # Check MPI rank - coil optimization runs only on rank 0
    is_mpi_parallel = comm_world is not None and hasattr(comm_world, 'size') and comm_world.size > 1
    is_proc0 = comm_world is None or not hasattr(comm_world, 'rank') or comm_world.rank == 0
    
    # Create MPI partition for post-processing (will be used after optimization)
    # Only create if MPI is available and we're using multiple processes
    mpi_partition = None
    if is_mpi_parallel:
        try:
            from simsopt.util.mpi import MpiPartition
            mpi_partition = MpiPartition(ngroups=1)  # Use all processes for VMEC
        except ImportError:
            mpi_partition = None  # MPI not available
        proc0_print(f"Running with MPI: {comm_world.size} processes")
        proc0_print("Coil optimization will run on rank 0 only; post-processing will use all processes")
        # Note: All ranks proceed together; the if is_proc0 block handles which rank does work
        # Barriers are called uniformly by ALL ranks to ensure synchronization
    
    if case_cfg is None:
        case_cfg = load_case_config(case_path)
    
    # Merge post_processing_params from case.yaml with function parameters
    # Case.yaml settings override function defaults, but CLI flags (passed as function args) take precedence
    pp_params = case_cfg.post_processing_params or {}
    # Only use case.yaml values if they were not explicitly set via CLI (check against defaults)
    if not run_vmec and pp_params.get("run_vmec", False):
        run_vmec = True
    if not run_simple and pp_params.get("run_simple", False):
        run_simple = True
    if not plot_poincare and pp_params.get("plot_poincare", False):
        plot_poincare = True
    elif plot_poincare and not pp_params.get("plot_poincare", True):
        plot_poincare = False
    # Additional params that can only come from case.yaml (or CLI via function args)
    plot_boozer = pp_params.get("plot_boozer", True)
    # CLI flags override case.yaml for plot_finite_build
    if not plot_finite_build:
        plot_finite_build = pp_params.get("plot_finite_build", False)
    if finite_build_width is None:
        finite_build_width = pp_params.get("finite_build_width")
    if finite_build_height is None:
        finite_build_height = pp_params.get("finite_build_height")
    
    # Resolve case_path to absolute path before changing directories
    # This ensures post-processing can find it even after os.chdir(output_dir)
    case_path_obj = Path(case_path)
    if case_path_obj.is_file():
        # It's already the YAML file
        case_yaml_path_abs = case_path_obj.resolve()
    elif case_path_obj.is_dir():
        # It's a directory, look for case.yaml inside
        case_yaml_path_abs = (case_path_obj / "case.yaml").resolve()
    else:
        # Try to resolve it (might be relative path)
        case_yaml_path_abs = case_path_obj.resolve() if case_path_obj.exists() else None
    
    coil_params = dict(case_cfg.coils_params)
    optimizer_params = dict(case_cfg.optimizer_params)
    surface_params = dict(case_cfg.surface_params)
    coil_objective_terms = case_cfg.coil_objective_terms
    
    # Extract threshold values from coil_objective_terms if present
    # These will be passed as kwargs to optimize_coils_loop
    threshold_kwargs = {}
    if coil_objective_terms:
        threshold_keys = [
            "length_threshold",
            "cc_threshold",
            "cs_threshold",
            "curvature_threshold",
            "arclength_variation_threshold",
            "msc_threshold",
            "force_threshold",
            "torque_threshold",
            "flux_threshold",
        ]
        for key in threshold_keys:
            if key in coil_objective_terms:
                threshold_kwargs[key] = coil_objective_terms[key]

        # Create a copy of coil_objective_terms without threshold keys
        coil_objective_terms = {
            k: v for k, v in coil_objective_terms.items()
            if k not in threshold_keys
        }

    # Extract dof_perturbation from case_config (top-level key, not in coil_objective_terms)
    # This allows the autopilot to request random perturbation of initial coil DOFs.
    if case_cfg and hasattr(case_cfg, '__dict__'):
        # Check the raw dict if available
        pass
    # dof_perturbation is passed via the case.yaml directly (not a CaseConfig field)
    # We read it from the YAML file if it was written there
    if case_yaml_path_abs and case_yaml_path_abs.exists():
        import yaml as _yaml_loader
        raw_config = _yaml_loader.safe_load(case_yaml_path_abs.read_text())
        if isinstance(raw_config, dict) and 'dof_perturbation' in raw_config:
            threshold_kwargs['dof_perturbation'] = raw_config['dof_perturbation']
    
    # Handle surface file path - check if it's relative to plasma_surfaces directory
    surface_file = surface_params["surface"]
    if not Path(surface_file).is_absolute():
        # Try relative to case_path first, then plasma_surfaces
        # Also try case-insensitive matching for files like MUSE.focus vs muse.focus
        potential_paths = [
            case_path / surface_file,
            Path("plasma_surfaces") / surface_file,
            Path.cwd() / "plasma_surfaces" / surface_file,
        ]
        # Add case-insensitive variants
        surface_file_lower = surface_file.lower()
        if surface_file != surface_file_lower:
            potential_paths.extend([
                Path("plasma_surfaces") / surface_file_lower,
                Path.cwd() / "plasma_surfaces" / surface_file_lower,
            ])
        
        found = False
        for path in potential_paths:
            if path.exists():
                surface_file = str(path)
                found = True
                break
        
        if not found:
            # Try to find any file with matching name (case-insensitive) in plasma_surfaces
            plasma_surfaces_dir = Path("plasma_surfaces")
            if plasma_surfaces_dir.exists():
                for file in plasma_surfaces_dir.iterdir():
                    if file.name.lower() == surface_file.lower():
                        surface_file = str(file)
                        found = True
                        break
    
    # Load surface based on file type
    # MUSE files are VMEC input files, so treat them the same way
    surface_file_lower = surface_file.lower()
    if "input" in surface_file_lower:
        surface_func = SurfaceRZFourier.from_vmec_input
    elif "wout" in surface_file_lower:
        surface_func = SurfaceRZFourier.from_wout
    elif "focus" in surface_file_lower:
        surface_func = SurfaceRZFourier.from_focus
    else:
        raise ValueError(f"Unknown surface type: {surface_file}")

    surface = surface_func(
        filename=surface_file, 
        range=surface_params["range"],
        nphi=surface_resolution, 
        ntheta=surface_resolution)
    
    # Determine output directory for VTK files
    if output_dir is None:
        output_dir = coils_out_path.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to output directory to ensure VTK files are saved there
    # optimize_coils_loop may save files based on current working directory
    import os
    original_cwd = Path.cwd()
    
    # Convert surface_file to absolute path before changing directories
    if not Path(surface_file).is_absolute():
        surface_file = str(Path(surface_file).resolve())

    if 'muse' in surface_file:
        target_B = 0.15 
    elif 'LandremanPaul2021_QA' in surface_file:
        target_B = 1.0
    elif 'LandremanPaul2021_QH_reactorScale_lowres' in surface_file:
        target_B = 5.7  # Reactor-scale QH design
    elif 'circular_tokamak' in surface_file:
        target_B = 1.0
    elif 'rotating_ellipse' in surface_file:
        target_B = 1.0
    elif 'c09r00' in surface_file:
        target_B = 0.5  # Half-tesla target B-field for C09R00 NCSX from PM4Stell design
    elif 'cfqs_2b40' in surface_file:
        target_B = 1.0
    elif 'W7-X' in surface_file:
        target_B = 2.5  # 2.5 T target B-field for W7-X design here
    elif 'HSX' in surface_file:
        target_B = 2.0  # 2 T target B-field for HSX_QH design here
    elif 'schuetthenneberg' in surface_file.lower():
        target_B = 5.7  # 5.7 T ARIES-CS target B-field for Schuetthenneberg QA design
    else:
        raise ValueError(f"Unknown surface file: {surface_file}")
    coil_params['target_B'] = target_B

    # Virtual casing support
    # If enabled, compute B_external_normal from VMEC equilibrium to use as target in SquaredFlux
    # virtual_casing can be True/False boolean or omitted (defaults to False)
    vc_target = None
    vc_target_plot = None  # For plotting on the full surface
    use_virtual_casing = surface_params.get('virtual_casing', False)
    if use_virtual_casing:
        if not VIRTUAL_CASING_AVAILABLE:
            raise ImportError(
                "Virtual casing is enabled but the virtual_casing package is not installed. "
                "Install it with: pip install git+https://github.com/hiddenSymmetries/virtual-casing"
            )
        
        # Find VMEC wout file for virtual casing
        # Virtual casing requires a VMEC wout file (not input or FOCUS file)
        vmec_file = None
        if "wout" in surface_file_lower:
            vmec_file = surface_file
        else:
            # Try to find a corresponding wout file
            # Common pattern: input.* -> wout_*.nc or focus file -> wout_*.nc
            surface_path = Path(surface_file)
            potential_wout_files = [
                surface_path.parent / f"wout_{surface_path.stem}.nc",
                surface_path.parent / f"wout_{surface_path.stem.replace('input.', '')}.nc",
                Path("plasma_surfaces") / f"wout_{surface_path.stem}.nc",
                Path("plasma_surfaces") / f"wout_{surface_path.stem.replace('input.', '')}.nc",
            ]
            for wout_path in potential_wout_files:
                if wout_path.exists():
                    vmec_file = str(wout_path)
                    break
        
        if vmec_file is None:
            raise ValueError(
                f"Virtual casing is enabled but no VMEC wout file found for surface: {surface_file}. "
                "Virtual casing requires a VMEC wout file. Either provide a wout file directly "
                "or ensure a corresponding wout_*.nc file exists."
            )
        
        # Virtual casing resolution must match the surface resolution
        # The target resolution should match nphi and ntheta of the surface
        # surface uses surface_resolution for both nphi and ntheta
        vc_src_nphi = 80
        vc_src_ntheta = 80
        vc_trgt_nphi = surface_resolution
        vc_trgt_ntheta = surface_resolution
        
        print("Running virtual casing calculation...")
        print(f"  VMEC file: {vmec_file}")
        print(f"  Source resolution: {vc_src_nphi} x {vc_src_ntheta}")
        print(f"  Target resolution (matches surface): {vc_trgt_nphi} x {vc_trgt_ntheta}")
        
        vc = VirtualCasing.from_vmec(
            vmec_file,
            src_nphi=vc_src_nphi, src_ntheta=vc_src_ntheta,
            trgt_nphi=vc_trgt_nphi, trgt_ntheta=vc_trgt_ntheta,
        )
        vc_target = vc.B_external_normal.copy()  # Copy to allow cleanup
        print(f"  Virtual casing calculation complete. B_external_normal shape: {vc_target.shape}")
        del vc  # Free memory from virtual casing calculation
        import gc
        gc.collect()
        print(f"  Surface resolution: {surface_resolution} x {surface_resolution}")
        
        # Also compute virtual casing for full surface plotting
        # The plot surface uses 2x upsampling by default, so use 2*surface_resolution
        plot_resolution = 2 * surface_resolution
        print(f"  Computing virtual casing for full surface plotting (resolution: {plot_resolution} x {plot_resolution})...")
        vc_plot = VirtualCasing.from_vmec(
            vmec_file,
            src_nphi=vc_src_nphi, src_ntheta=vc_src_ntheta,
            trgt_nphi=plot_resolution, trgt_ntheta=plot_resolution,
        )
        vc_target_plot = vc_plot.B_external_normal
        print(f"  Virtual casing for plotting complete. B_external_normal shape: {vc_target_plot.shape}")

    # Coil optimization runs only on rank 0 when using MPI
    # Other ranks will skip optimization and wait at the barrier after optimization
    try:
        os.chdir(output_dir)
        
        # Extract algorithm_options from optimizer_params if present
        # This allows users to specify algorithm-specific hyperparameters
        algorithm_options = optimizer_params.pop('algorithm_options', {})
        
        # Only rank 0 runs the actual optimization
        # When using MPI, skip post-processing in the loop (will run after barrier)
        coils = None
        results_dict = {}
        
        # When using MPI, post-processing must run after the barrier so all processes participate
        # So we skip it in the optimization loop and run it separately
        skip_post_processing_in_loop = skip_post_processing or is_mpi_parallel
        
        if is_proc0:
            fourier_continuation = case_cfg.fourier_continuation
            coil_type = coil_params.get("coil_type", "modular")
            if coil_type == "dipole":
                coils, results_dict = optimize_coils_dipole_loop(
                    surface,
                    surface_file=surface_file,
                    surface_params=surface_params,
                    out_dir=str(output_dir),
                    max_iterations=optimizer_params.get("max_iterations", 100),
                    verbose=optimizer_params.get("verbose", False),
                    skip_post_processing=skip_post_processing_in_loop,
                    case_path=case_yaml_path_abs if case_yaml_path_abs and case_yaml_path_abs.exists() else case_path,
                    run_vmec=run_vmec,
                    run_simple=run_simple,
                    plot_poincare=plot_poincare,
                    plot_boozer=pp_params.get("plot_boozer", True),
                    **{k: v for k, v in coil_params.items() if k not in ("coil_type", "ncoils", "order")},
                )
            elif fourier_continuation and fourier_continuation.get('enabled', False):
                # Use Fourier continuation
                fourier_orders = fourier_continuation.get('orders', [coil_params.get('order', 16)])
                if not isinstance(fourier_orders, list) or not all(isinstance(o, int) for o in fourier_orders):
                    raise ValueError("fourier_continuation.orders must be a list of integers")
                
                coils, results_dict = optimize_coils_with_fourier_continuation(
                    surface,
                    fourier_orders=fourier_orders,
                    target_B=coil_params.get('target_B', 5.7),
                    out_dir=str(output_dir),
                    max_iterations=optimizer_params.get('max_iterations', 30),
                    ncoils=coil_params.get('ncoils', 4),
                    verbose=optimizer_params.get('verbose', False),
                    regularization=regularization_circ if regularization_circ is not None else lambda x: None,
                    coil_objective_terms=coil_objective_terms,
                    surface_resolution=surface_resolution,
                    algorithm_options=algorithm_options,
                    case_path=case_yaml_path_abs if case_yaml_path_abs and case_yaml_path_abs.exists() else case_path,  # Pass resolved absolute path
                    vc_target=vc_target,  # Virtual casing B_external_normal target
                    vc_target_plot=vc_target_plot,  # Virtual casing target for plotting
                    skip_post_processing=skip_post_processing_in_loop,  # Skip in loop when using MPI
                    run_vmec=run_vmec,
                    run_simple=run_simple,
                    plot_poincare=plot_poincare,
                    plot_boozer=plot_boozer,
                    plot_finite_build=plot_finite_build,
                    finite_build_width=finite_build_width,
                    finite_build_height=finite_build_height,
                    **{k: v for k, v in optimizer_params.items() if k != 'max_iterations' and k != 'verbose'},
                    **threshold_kwargs
                )
            else:
                # Standard optimization without continuation
                # Pass output_dir to optimize_coils_loop for VTK file output
                # optimize_coils_loop saves VTK files to output_dir during optimization
                try:
                    coils, results_dict = optimize_coils_loop(
                        surface, 
                        **coil_params, 
                        **optimizer_params, 
                        output_dir=str(output_dir),
                        coil_objective_terms=coil_objective_terms,
                        surface_resolution=surface_resolution,
                        algorithm_options=algorithm_options,
                        case_path=case_yaml_path_abs if case_yaml_path_abs and case_yaml_path_abs.exists() else case_path,  # Pass resolved absolute path
                        vc_target=vc_target,  # Virtual casing B_external_normal target
                        vc_target_plot=vc_target_plot,  # Virtual casing target for plotting
                        skip_post_processing=skip_post_processing_in_loop,  # Skip in loop when using MPI
                        run_vmec=run_vmec,
                        run_simple=run_simple,
                        plot_poincare=plot_poincare,
                        plot_boozer=plot_boozer,
                        plot_finite_build=plot_finite_build,
                        finite_build_width=finite_build_width,
                        finite_build_height=finite_build_height,
                        **threshold_kwargs
                    )
                except TypeError:
                    # Fallback if optimize_coils_loop doesn't accept output_dir parameter
                    # Files will be saved to current directory (which is now output_dir)
                    coils, results_dict = optimize_coils_loop(
                        surface, 
                        **coil_params, 
                        **optimizer_params, 
                        coil_objective_terms=coil_objective_terms,
                        algorithm_options=algorithm_options,
                        surface_resolution=surface_resolution,
                        case_path=case_yaml_path_abs if case_yaml_path_abs and case_yaml_path_abs.exists() else case_path,  # Pass resolved absolute path
                        vc_target=vc_target,  # Virtual casing B_external_normal target
                        vc_target_plot=vc_target_plot,  # Virtual casing target for plotting
                        run_vmec=run_vmec,
                        run_simple=run_simple,
                        plot_poincare=plot_poincare,
                        plot_boozer=plot_boozer,
                        plot_finite_build=plot_finite_build,
                        finite_build_width=finite_build_width,
                        finite_build_height=finite_build_height,
                        **threshold_kwargs
                    )
        
        # Barrier: wait for rank 0 to finish optimization before proceeding
        if is_mpi_parallel:
            comm_world.Barrier()  # All processes wait here until rank 0 finishes optimization
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)
    
    # Ensure output path has .json extension for JSON format
    if not str(coils_out_path).endswith('.json'):
        coils_out_path = coils_out_path.with_suffix('.json')
    
    # Save coils to JSON file (only rank 0 writes, but all ranks need to wait)
    # Use absolute path to ensure correct location
    abs_coils_path = coils_out_path if coils_out_path.is_absolute() else (output_dir / coils_out_path.name)
    if is_proc0:
        if coils is None:
            raise RuntimeError("Coil optimization failed: no coils were produced")
        save(coils, abs_coils_path)
    
    # Barrier: ensure coils file is written before any process proceeds
    if is_mpi_parallel:
        comm_world.Barrier()  # All processes wait until coils file is written
    
    # Run post-processing after barrier (so all MPI processes can participate)
    # This only runs if we skipped it in the loop (i.e., when using MPI)
    if not skip_post_processing and is_mpi_parallel:
        try:
            from .post_processing import run_post_processing
            
            # Find coils JSON file (all processes need to do this)
            coils_json_path = abs_coils_path
            if not coils_json_path.exists():
                # Try alternative names
                coils_json_path = output_dir / "biot_savart_optimized.json"
                if not coils_json_path.exists():
                    coils_json_path = output_dir / "coils.json"
            
            if coils_json_path.exists():
                proc0_print("\nRunning post-processing (QFM, Poincaré plots, profiles)...")
                
                # Determine helicity_n based on surface type (QA=0, QH=-1)
                # Only rank 0 needs to read the file, but all processes need the value
                helicity_n = 0
                if is_proc0 and case_yaml_path_abs and case_yaml_path_abs.exists():
                    import yaml
                    try:
                        case_data = yaml.safe_load(case_yaml_path_abs.read_text())
                        surface_name = case_data.get("surface_params", {}).get("surface", "").lower()
                        if "qh" in surface_name or "qash" in surface_name:
                            helicity_n = -1
                    except Exception:
                        pass
                
                # Broadcast helicity_n to all processes (simple approach: all processes read)
                # Actually, run_post_processing will handle this, so we can just use default
                # But let's read it on all processes for simplicity
                if not is_proc0 and case_yaml_path_abs and case_yaml_path_abs.exists():
                    import yaml
                    try:
                        case_data = yaml.safe_load(case_yaml_path_abs.read_text())
                        surface_name = case_data.get("surface_params", {}).get("surface", "").lower()
                        if "qh" in surface_name or "qash" in surface_name:
                            helicity_n = -1
                    except Exception:
                        pass
                
                # Find plasma_surfaces_dir (all processes need this)
                plasma_surfaces_dir = None
                current_dir = Path(output_dir)
                for _ in range(5):
                    potential_plasma_dir = current_dir / "plasma_surfaces"
                    if potential_plasma_dir.exists():
                        plasma_surfaces_dir = potential_plasma_dir
                        break
                    if current_dir.parent == current_dir:
                        break
                    current_dir = current_dir.parent
                
                # Run post-processing (ALL MPI processes participate - function handles MPI internally)
                post_processing_results = run_post_processing(
                    coils_json_path=coils_json_path,
                    output_dir=output_dir,
                    case_yaml_path=case_yaml_path_abs if case_yaml_path_abs and case_yaml_path_abs.exists() else None,
                    plasma_surfaces_dir=plasma_surfaces_dir,
                    run_vmec=run_vmec,
                    helicity_m=1,
                    helicity_n=helicity_n,
                    ns=50,
                    plot_boozer=plot_boozer,
                    plot_poincare=plot_poincare,
                    nfieldlines=20,
                    run_simple=run_simple,
                    mpi=mpi_partition,  # Pass MPI partition explicitly
                    plot_finite_build=plot_finite_build,
                    finite_build_width=finite_build_width,
                    finite_build_height=finite_build_height,
                )
                proc0_print("Post-processing complete!")
                if is_proc0 and 'quasisymmetry_average' in post_processing_results:
                    proc0_print(f"  Average quasisymmetry error: {post_processing_results['quasisymmetry_average']:.2e}")
                
                # Add post-processing results to results_dict (only rank 0 returns this)
                if is_proc0:
                    results_dict['post_processing'] = post_processing_results
            else:
                proc0_print(f"Warning: Skipping post-processing (coils_json not found: {coils_json_path})")
        except Exception as e:
            proc0_print(f"Warning: Post-processing failed: {e}")
            if is_proc0:
                import traceback
                traceback.print_exc()
    
    return results_dict


def initialize_coils_loop(
    s : SurfaceRZFourier, out_dir: Path | str = '', 
    target_B: float = 5.7, ncoils: int = 4, order: int = 16, coil_width : float = 0.4,
    regularization: Callable | None = regularization_circ):
    """
    Initializes coils with order=16 and total current set to produce 
    a target B-field on-axis. Uses an adaptive strategy to determine R0 and R1
    parameters to ensure coils:
    - Don't intersect with the plasma surface
    - Interlink the plasma (go around it) by being positioned outside the surface
    - Maintain safe distance from surface
    - Don't interlink with each other (linking number ~0, maintain separation)
    
    The function iteratively adjusts R0 and R1 until all constraints are satisfied,
    then iteratively adjusts the total current until the field strength along the 
    major radius averages to the target value.

    Args:
        s: plasma boundary surface.
        out_dir: Path or string for the output directory for saved files.
        target_B: Target magnetic field strength in Tesla (default: 5.7).
        ncoils: Number of coils to create (default: 4).
        order: Fourier order for coil curves (default: 16).
        coil_width: Width of the coil in meters (default: 0.4).
        regularization: Regularization function (default: regularization_circ).
    Returns:
        coils: List of Coil class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, coils_via_symmetries, BiotSavart
    from simsopt.util.coil_optimization_helper_functions import calculate_modB_on_major_radius
    import numpy as np

    out_dir = Path(out_dir)

    if regularization is not None:
        regularizations = [regularization(coil_width) for _ in range(ncoils)]
    else:
        regularizations = None
    
    # Adaptive R0 and R1 initialization
    # Start with conservative initial values
    major_radius = s.major_radius()
    minor_radius_component = s.minor_radius()
    
    # Minimum distances we want to maintain
    min_cs_distance = 0.1 * major_radius  # Minimum coil-to-surface distance (15% of major radius)
    min_cc_distance = 0.1 * major_radius  # Minimum coil-to-coil distance (15% of major radius)
    
    # Initial R0 and R1 scaling factors
    R0_scale = 1.0  # Start with 1.0x major radius
    R1_scale = 2.5  # Start with 2.5x minor radius component
    
    # Maximum iterations for adaptive R0/R1 adjustment
    max_adaptive_iterations = 50
    adaptive_tolerance = 0.1  # 10% tolerance for distance checks
    
    # Maximum scaling factors to prevent coils from going too far
    max_R0_scale = 3.0  # Don't let R0 exceed 3x major radius
    max_R1_scale = 5.0  # Don't let R1 exceed 5x minor radius
    
    # Initial guess for total current (using QH configuration as reference)
    total_current = 5e7  # 50 MA initial guess is not bad for reactor-scale
    
    R0 = major_radius * R0_scale
    R1 = minor_radius_component * R1_scale
    
    # Adaptive loop to find suitable R0 and R1
    from simsopt.geo import CurveSurfaceDistance, CurveCurveDistance, LinkingNumber
    
    # Track previous values to detect oscillation
    prev_R0_scale = None
    prev_R1_scale = None
    oscillation_count = 0
    
    for adaptive_iter in range(max_adaptive_iterations):
        # Check for oscillation (values repeating)
        if prev_R0_scale is not None and prev_R1_scale is not None:
            if abs(R0_scale - prev_R0_scale) < 0.01 and abs(R1_scale - prev_R1_scale) < 0.01:
                oscillation_count += 1
                if oscillation_count >= 3:
                    break
            else:
                oscillation_count = 0
        
        prev_R0_scale = R0_scale
        prev_R1_scale = R1_scale
        # Create equally spaced curves with current R0 and R1
        base_curves = create_equally_spaced_curves(
            ncoils, s.nfp, stellsym=s.stellsym,
            R0=R0, R1=R1, order=order, numquadpoints=256)
        
        # Create temporary coils to check distances
        base_currents_temp = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
        total_current_obj_temp = Current(total_current)
        total_current_obj_temp.fix_all()
        base_currents_temp += [total_current_obj_temp - sum(base_currents_temp)]
        
        try:
            coils_temp = coils_via_symmetries(
                base_curves,
                base_currents_temp,
                s.nfp,
                s.stellsym,
                regularizations=regularizations,
            )
        except TypeError:
            coils_temp = coils_via_symmetries(base_curves, base_currents_temp, s.nfp, s.stellsym)
        
        # Get all curves (including symmetric ones)
        curves_temp = [c.curve for c in coils_temp]
        
        # Check coil-to-surface distance
        cs_dist = CurveSurfaceDistance(curves_temp, s, 0.0)
        min_cs_sep = cs_dist.shortest_distance()
        
        # Check coil-to-coil distance (only between base coils)
        cc_dist = CurveCurveDistance(curves_temp, 0.0, num_basecurves=ncoils)
        min_cc_sep = cc_dist.shortest_distance()
        
        # Check that coils don't interlink with each other (coil-coil interlinking)
        # Linking number should be close to zero - coils should not interlink each other
        link_num = LinkingNumber(curves_temp, downsample=2)
        linking_number = link_num.J()
        
        # Check if constraints are satisfied
        cs_ok = min_cs_sep >= min_cs_distance * (1 - adaptive_tolerance)
        cc_ok = min_cc_sep >= min_cc_distance * (1 - adaptive_tolerance)
        # Coils should NOT interlink with each other (linking number should be small/zero)
        # For equally spaced coils around a torus, linking number should be 0 or very small
        no_coil_interlink = abs(linking_number) < 0.1  # Coils should not interlink each other
        
        # For coils to interlink the plasma, they need to pass through the torus hole.
        # A coil interlinks the plasma if it has points both:
        # 1. Inside the torus hole (R < R_min of plasma surface)
        # 2. Outside the plasma (R > R_max of plasma surface)
        # This geometric check works for any surface geometry.
        
        # Find the R range of the plasma surface
        gamma = s.gamma()
        rs = np.sqrt(gamma[:, :, 0]**2 + gamma[:, :, 1]**2)
        R_min_surface = np.min(rs)  # Inner edge of plasma
        R_max_surface = np.max(rs)  # Outer edge of plasma
        
        # Check if coils interlink the plasma by sampling points on coils
        # and verifying they have both inside-hole and outside-plasma points
        coil_interlinks_plasma = False
        points_inside_hole_count = 0  # R < R_min (inside torus hole)
        points_outside_plasma_count = 0  # R > R_max (outside plasma)
        points_in_plasma_count = 0  # R_min <= R <= R_max (in plasma volume)
        
        # Sample all base curves to get better statistics
        for curve in base_curves:
            points = curve.gamma()
            # Calculate radial distance from origin for each point
            radial_distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            
            # Classify points based on radial position
            inside_hole_mask = radial_distances < R_min_surface * 0.98  # Inside torus hole (with small margin)
            outside_plasma_mask = radial_distances > R_max_surface * 1.02  # Outside plasma (with small margin)
            in_plasma_mask = (radial_distances >= R_min_surface * 0.98) & (radial_distances <= R_max_surface * 1.02)
            
            points_inside_hole_count += np.sum(inside_hole_mask)
            points_outside_plasma_count += np.sum(outside_plasma_mask)
            points_in_plasma_count += np.sum(in_plasma_mask)
            
            # A coil interlinks if it has both inside-hole and outside-plasma points
            if np.any(inside_hole_mask) and np.any(outside_plasma_mask):
                coil_interlinks_plasma = True
                # Don't break - continue to count all points for better diagnostics
        
        # Coils interlink plasma if:
        # 1. They maintain safe distance from surface (cs_ok)
        # 2. They have points both inside and outside the plasma volume
        plasma_interlink_ok = cs_ok and coil_interlinks_plasma
        
        if cs_ok and cc_ok and no_coil_interlink and plasma_interlink_ok:
            # Constraints satisfied, break out of adaptive loop
            break
        
        # Check if we've exceeded maximum scales
        if R0_scale > max_R0_scale or R1_scale > max_R1_scale:
            # Cap the scales
            R0_scale = min(R0_scale, max_R0_scale)
            R1_scale = min(R1_scale, max_R1_scale)
            R0 = major_radius * R0_scale
            R1 = minor_radius_component * R1_scale
            break
        
        # Adjust R0 and R1 based on constraint violations
        # Use priority-based approach with elif to fix only ONE constraint per iteration
        # This prevents oscillation by not adjusting multiple constraints simultaneously
        # Priority: plasma_interlink > cs_ok > cc_ok > no_coil_interlink
        # (Interlinking is most important - coils must pass through torus hole)
        if not plasma_interlink_ok:
            # Priority 1: Coils must interlink plasma (most important constraint)
            # If coils don't interlink, we need to adjust R0 and R1 so coils pass through the torus hole
            # R0 is the mean radius, R1 is the amplitude (coils extend from R0-R1 to R0+R1)
            if points_inside_hole_count == 0:
                # No points in torus hole - coils don't extend inward enough, need to extend inward
                # Increase R1 to make coil extend more inward, decrease R0 if safe
                R1_scale *= 1.2  # More aggressive
                R1 = minor_radius_component * R1_scale
                if min_cs_sep > min_cs_distance * 1.1:  # Small safety margin
                    R0_scale *= 0.95
                    R0 = major_radius * R0_scale
                else:
                    R0 = major_radius * R0_scale
            elif points_outside_plasma_count == 0:
                # No points outside plasma - coils don't extend outward enough, need to extend outward
                # Increase R1 primarily to extend outward (R0+R1 increases)
                R1_scale *= 1.2  # More aggressive increase for R1
                R1 = minor_radius_component * R1_scale
                # Optionally decrease R0 slightly to help extend outward while keeping inner edge similar
                if min_cs_sep > min_cs_distance * 1.5:  # Large safety margin
                    R0_scale *= 0.98
                    R0 = major_radius * R0_scale
                else:
                    R0 = major_radius * R0_scale
            else:
                # Some points in hole and outside but not both - try increasing R1 to extend more
                R1_scale *= 1.15
                R1 = minor_radius_component * R1_scale
        elif not cs_ok:
            # Priority 2: Coils must not intersect surface (only if interlinking is OK)
            # Move coils outward to increase distance from surface
            R0_scale *= 1.1
            R0 = major_radius * R0_scale
        elif not cc_ok:
            # Priority 2: Coils must not be too close to each other
            # Increase R0 to move coils further from center (increases toroidal separation)
            R0_scale *= 1.1
            R0 = major_radius * R0_scale
        elif not no_coil_interlink:
            # Priority 4: Coils should not interlink each other
            R0_scale *= 1.1
            R0 = major_radius * R0_scale
    
    # Final coil creation with determined R0 and R1
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=s.stellsym,
        R0=R0, R1=R1, order=order, numquadpoints=200)
    base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
    total_current_obj = Current(total_current)
    total_current_obj.fix_all()
    base_currents += [total_current_obj - sum(base_currents)]
    try:
        coils = coils_via_symmetries(
            base_curves,
            base_currents,
            s.nfp,
            s.stellsym,
            regularizations=regularizations,
        )
    except TypeError:
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
    
    # Iterative current adjustment to achieve the target B-field
    max_iterations = 30
    tolerance = 1e-3
    for _ in range(max_iterations):
        
        # Distribute current among coils
        base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
        total_current_obj = Current(total_current)
        total_current_obj.fix_all()
        base_currents += [total_current_obj - sum(base_currents)]
        
        # Create coils using symmetries
        try:
            coils = coils_via_symmetries(
                base_curves,
                base_currents,
                s.nfp,
                s.stellsym,
                regularizations=regularizations,
            )
        except TypeError:
            coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
        
        # Create BiotSavart object to evaluate field
        bs = BiotSavart(coils)
        
        # Calculate field strength along major radius (suppress simsopt Bmag prints)
        with suppress_output():
            B_avg = calculate_modB_on_major_radius(bs, s)
        
        # Check convergence
        if abs(B_avg - target_B) / target_B < tolerance:
            break
        
        # Adjust current based on field difference
        # Use simple linear scaling: new_current = current * (target_B / achieved_B)
        current_scale_factor = target_B / B_avg
        total_current *= current_scale_factor
    
    return coils


def initialize_coils_dipole(
    s: SurfaceRZFourier,
    surface_file: str,
    surface_params: Dict[str, Any],
    tf_configuration: str = "LandremanPaulQA",
    poff: float = 1.5,
    coff: float = 3.0,
    Nx: int = 4,
    Ny: int | None = None,
    Nz: int | None = None,
    dipole_order: int = 2,
    dipole_coil_size: float = 0.1,
    tf_coil_size: float = 0.2,
    remove_inboard_eps: float = -0.4,
    out_dir: Path | str = "",
) -> tuple:
    """
    Initialize dipole coils plus TF coils, modeled after dipole_array_tutorial_advanced.py.

    Creates planar dipole coils between two toroidal surfaces (inner and outer, extended
    from the plasma boundary), removes inboard dipoles, removes interlinking dipoles,
    and aligns dipole normals with the plasma surface. TF coils are initialized via
    simsopt.util.initialize_coils for the given configuration.

    Args:
        s: Plasma boundary surface.
        surface_file: Path to surface file (for creating s_inner/s_outer).
        surface_params: Dict with 'range' and other surface params.
        tf_configuration: TF coil config name ('LandremanPaulQA', 'LandremanPaulQH',
            'SchuettHennebergQAnfp2').
        poff: Inner surface extension distance [m].
        coff: Additional outer extension beyond inner [m].
        Nx, Ny, Nz: Grid dimensions for dipole placement (Ny, Nz default to Nx).
        dipole_order: Fourier order for planar dipole curves.
        dipole_coil_size: Dipole wire cross-section [m] (e.g. 0.1 for 10 cm).
        tf_coil_size: TF wire cross-section [m] (e.g. 0.2 for 20 cm).
        remove_inboard_eps: Eps for remove_inboard_dipoles.
        out_dir: Output directory for saved files.

    Returns:
        Tuple (coils, base_curves_dipole, base_curves_TF, ncoils_dipole, ncoils_TF)
        where coils = dipole_coils + TF_coils (all coils for BiotSavart).
    """
    from simsopt.geo import SurfaceRZFourier, create_planar_curves_between_two_toroidal_surfaces
    from simsopt.field import Current, coils_via_symmetries, BiotSavart
    from simsopt.util import (
        initialize_coils,
        remove_inboard_dipoles,
        remove_interlinking_dipoles_and_TFs,
        align_dipoles_with_plasma,
        calculate_modB_on_major_radius,
    )

    try:
        from simsopt.field import regularization_rect
    except ImportError:
        raise ImportError(
            "Dipole coil initialization requires simsopt with regularization_rect "
            "(auglag_coils branch or simsopt >= 0.14)."
        )

    out_dir = Path(out_dir)
    Ny = Ny if Ny is not None else Nx
    Nz = Nz if Nz is not None else Nx

    nphi = getattr(s, "quadpoints_phi", None)
    ntheta = getattr(s, "quadpoints_theta", None)
    nphi = len(nphi) if nphi is not None else 32
    ntheta = len(ntheta) if ntheta is not None else 32
    range_param = surface_params.get("range", "half period")

    surface_file_lower = surface_file.lower()
    if "input" in surface_file_lower:
        surface_func = SurfaceRZFourier.from_vmec_input
    elif "wout" in surface_file_lower:
        surface_func = SurfaceRZFourier.from_wout
    elif "focus" in surface_file_lower:
        surface_func = SurfaceRZFourier.from_focus
    else:
        raise ValueError(f"Unknown surface type: {surface_file}")

    s_inner = surface_func(
        filename=surface_file,
        range=range_param,
        nphi=nphi * 4,
        ntheta=ntheta * 4,
    )
    s_outer = surface_func(
        filename=surface_file,
        range=range_param,
        nphi=nphi * 4,
        ntheta=ntheta * 4,
    )
    s_inner.extend_via_normal(poff)
    s_outer.extend_via_normal(poff + coff)

    regularization_TF = regularization_rect(tf_coil_size, tf_coil_size)
    base_curves_TF, curves_TF, coils_TF, _ = initialize_coils(
        s, tf_configuration, regularization_TF
    )
    num_TF_unique = len(base_curves_TF)

    base_curves, _ = create_planar_curves_between_two_toroidal_surfaces(
        s, s_inner, s_outer, Nx, Ny, Nz, order=dipole_order
    )
    base_curves = remove_inboard_dipoles(s, base_curves, eps=remove_inboard_eps)
    base_curves = remove_interlinking_dipoles_and_TFs(base_curves, base_curves_TF)
    alphas, deltas = align_dipoles_with_plasma(s, base_curves)

    for i in range(len(base_curves)):
        alpha2 = alphas[i] / 2.0
        delta2 = deltas[i] / 2.0
        base_curves[i].set("q0", np.cos(alpha2) * np.cos(delta2))
        base_curves[i].set("qi", np.sin(alpha2) * np.cos(delta2))
        base_curves[i].set("qj", np.cos(alpha2) * np.sin(delta2))
        base_curves[i].set("qk", -np.sin(alpha2) * np.sin(delta2))

    ncoils_dipole = len(base_curves)
    regularization_dipole = regularization_rect(dipole_coil_size, dipole_coil_size)
    base_currents = [Current(1.0) * 1e7 for _ in range(ncoils_dipole)]
    regularizations = [regularization_dipole for _ in range(ncoils_dipole)]
    coils_dipole = coils_via_symmetries(
        base_curves, base_currents, s.nfp, s.stellsym, regularizations=regularizations
    )
    coils = coils_dipole + coils_TF

    bs_TF = BiotSavart(coils_TF)
    bs_dipole = BiotSavart(coils_dipole)
    btot = bs_dipole + bs_TF
    with suppress_output():
        calculate_modB_on_major_radius(btot, s)

    return coils, base_curves, base_curves_TF, ncoils_dipole, num_TF_unique


def _zip_output_files(out_dir: Path) -> Path:
    """
    Zip all output files in the output directory with a date stamp.
    
    Parameters
    ----------
    out_dir: Path
        Directory containing output files to zip.
    
    Returns
    -------
    Path
        Path to the created zip file.
    """
    out_dir = Path(out_dir)
    
    # Create date-stamped zip filename: YYYY-MM-DD_HH-MM-SS.zip
    now = datetime.now()
    zip_filename = now.strftime("%Y-%m-%d_%H-%M-%S.zip")
    zip_path = out_dir / zip_filename
    
    # Find all files to zip (VTK files, JSON files, etc.)
    # Only zip VTK files for compression - keep JSON files (coils.json, results.json) unzipped
    files_to_zip = []
    for pattern in ["*.vtu", "*.vts"]:
        files_to_zip.extend(out_dir.glob(pattern))
    
    # Only create zip if there are files to zip
    if files_to_zip:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_zip:
                # Add file to zip with relative path (just filename)
                zipf.write(file_path, arcname=file_path.name)
        
        # Remove original VTK files after zipping for compression
        for file_path in files_to_zip:
            file_path.unlink()
    
    return zip_path


def _plot_bn_error_3d(
    surface,
    bs,
    coils,
    out_dir: Path,
    filename: str = "bn_error_3d_plot.pdf",
    title: str = "B_N/|B| Error on Plasma Surface with Optimized Coils",
    plot_upsample: int = 2,
    vc_target: np.ndarray | None = None,
) -> None:
    """
    Generate a 3D plot showing B_N/|B| error on the plasma surface with optimized coils.
    
    Parameters
    ----------
    surface: SurfaceRZFourier
        The plasma surface for plotting (should be full torus).
    bs: BiotSavart
        BiotSavart object containing the magnetic field from coils.
    coils: list
        List of coil objects to plot.
    out_dir: Path
        Directory where the PDF plot will be saved.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping 3D plot generation")
        return
    
    # Upsample surface for smoother plotting when possible
    plot_surface = surface
    if isinstance(surface, SurfaceRZFourier) and plot_upsample > 1:
        try:
            qphi = max(16, int(len(surface.quadpoints_phi) * plot_upsample))
            qtheta = max(16, int(len(surface.quadpoints_theta) * plot_upsample))
            quadpoints_phi = np.linspace(0, 1, qphi)
            quadpoints_theta = np.linspace(0, 1, qtheta)
            plot_surface = SurfaceRZFourier(
                nfp=surface.nfp,
                stellsym=surface.stellsym,
                mpol=surface.mpol,
                ntor=surface.ntor,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )
            for m in range(surface.mpol + 1):
                for n in range(-surface.ntor, surface.ntor + 1):
                    rc_val = surface.get_rc(m, n)
                    zs_val = surface.get_zs(m, n)
                    if rc_val != 0:
                        plot_surface.set_rc(m, n, rc_val)
                    if zs_val != 0:
                        plot_surface.set_zs(m, n, zs_val)
        except Exception:
            plot_surface = surface
    
    # Get surface points - grid should be square (nphi == ntheta)
    surface_points = plot_surface.gamma().reshape(-1, 3)
    npoints = surface_points.shape[0]
    nphi_plot = int(np.sqrt(npoints))
    ntheta_plot = nphi_plot
    
    # Reshape surface points to grid
    x_surf = surface_points[:, 0].reshape((nphi_plot, ntheta_plot))
    y_surf = surface_points[:, 1].reshape((nphi_plot, ntheta_plot))
    z_surf = surface_points[:, 2].reshape((nphi_plot, ntheta_plot))
    
    # Calculate B_N/|B| on surface
    bs.set_points(surface_points)
    B_field = bs.B().reshape((nphi_plot, ntheta_plot, 3))
    unit_normal = plot_surface.unitnormal().reshape((nphi_plot, ntheta_plot, 3))
    BdotN_coils = np.sum(B_field * unit_normal, axis=2)
    abs_B = bs.AbsB().reshape((nphi_plot, ntheta_plot))
    
    # If virtual casing target is provided, subtract it from the coil B_N
    if vc_target is not None:
        BdotN_error = np.abs(BdotN_coils - vc_target)
    else:
        BdotN_error = np.abs(BdotN_coils)
    
    # Avoid division by zero
    abs_B = np.where(abs_B > 1e-10, abs_B, 1e-10)
    bn_over_b = BdotN_error / abs_B
    
    # Create figure with 3D subplot
    fig = plt.figure(figsize=(12, 9), dpi=200)  # type: ignore
    ax = fig.add_subplot(111, projection='3d')  # type: ignore
    
    # Plot surface with B_N/|B| as colormap (opaque to avoid artifacts)
    norm = Normalize(vmin=0, vmax=bn_over_b.max() if bn_over_b.max() > 0 else 1)  # type: ignore
    facecolors = cm.viridis(norm(bn_over_b))  # type: ignore[attr-defined]
    ax.plot_surface(  # type: ignore[attr-defined]
        x_surf, y_surf, z_surf,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=True,
        rstride=1,
        cstride=1,
        zorder=1
    )
    
    # Plot coils colored by current magnitude with simple front/back layering
    currents = [abs(c.current.get_value()) for c in coils]
    current_norm = Normalize(  # type: ignore[call-overload]
        vmin=min(currents) if currents else 0.0,
        vmax=max(currents) if currents else 1.0,
    )
    # Use 'plasma' colormap (dark purple->pink->yellow) for coil currents
    # This provides good contrast with viridis (blue-green-yellow) used for B_N errors
    # and has dark colors at the bottom that are visible on white background
    current_cmap = cm.plasma  # type: ignore
    
    def _segments_from_mask(points: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
        segments: list[np.ndarray] = []
        start = 0
        for i in range(1, len(points)):
            if mask[i] != mask[i - 1]:
                if mask[i - 1]:
                    segments.append(points[start:i])
                start = i
        if mask[-1]:
            segments.append(points[start:])
        return segments
    
    center = np.array([x_surf.mean(), y_surf.mean(), z_surf.mean()])
    azim = np.deg2rad(ax.azim)  # type: ignore[attr-defined]
    elev = np.deg2rad(ax.elev)  # type: ignore[attr-defined]
    view_vec = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ])
    
    front_segments: list[tuple[np.ndarray, tuple[float, float, float]]] = []
    
    for coil in coils:
        coil_points = coil.curve.gamma()
        current_val = abs(coil.current.get_value())
        color_rgba = current_cmap(current_norm(current_val))
        # Convert RGBA to RGB (remove alpha channel) to ensure fully opaque coils
        if len(color_rgba) == 4:
            color = tuple(color_rgba[:3])  # Take only RGB, drop alpha
        else:
            color = color_rgba
        closed = np.vstack([coil_points, coil_points[0]])
        depth = (closed - center) @ view_vec
        front_mask = depth >= 0
        back_mask = ~front_mask
        
        for seg in _segments_from_mask(closed, back_mask):
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                seg[:, 2],
                color=color,
                linewidth=2.2,
                solid_capstyle="round",
                zorder=0,
            )
        
        for seg in _segments_from_mask(closed, front_mask):
            front_segments.append((seg, color))
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)  # type: ignore
    ax.set_ylabel('Y (m)', fontsize=12)  # type: ignore
    ax.set_zlabel('Z (m)', fontsize=12)  # type: ignore
    ax.set_title(title, fontsize=13, pad=16)  # type: ignore
    
    # Add surface colorbar
    mappable = cm.ScalarMappable(cmap=cm.viridis, norm=norm)  # type: ignore
    mappable.set_array(bn_over_b)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)  # type: ignore
    cbar.set_label('|B_N|/|B|', fontsize=12, rotation=270, labelpad=20)
    
    # Add coil current colorbar on the left side
    coil_mappable = cm.ScalarMappable(cmap=current_cmap, norm=current_norm)  # type: ignore
    coil_mappable.set_array(currents)
    coil_cbar = plt.colorbar(  # type: ignore
        coil_mappable,
        ax=ax,
        shrink=0.6,
        aspect=20,
        pad=0.08,
        location="left",
    )
    coil_cbar.set_label('|I| (A)', fontsize=12, rotation=90, labelpad=18)
    
    # Draw front coil segments after the surface for better depth cues
    for seg, color in front_segments:
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            seg[:, 2],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=3,
        )
    
    # Set equal aspect ratio
    max_range = np.array([
        x_surf.max() - x_surf.min(),
        y_surf.max() - y_surf.min(),
        z_surf.max() - z_surf.min()
    ]).max() / 2.0
    mid_x = (x_surf.max() + x_surf.min()) * 0.5
    mid_y = (y_surf.max() + y_surf.min()) * 0.5
    mid_z = (z_surf.max() + z_surf.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)  # type: ignore
    ax.set_ylim(mid_y - max_range, mid_y + max_range)  # type: ignore
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # type: ignore
    
    # Clean up axes for a sleeker look
    ax.grid(True)  # type: ignore
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):  # type: ignore
        axis.pane.fill = False  # type: ignore
        axis.pane.set_edgecolor("w")  # type: ignore
    
    # Save as PDF
    pdf_path = out_dir / filename
    plt.savefig(pdf_path, format='pdf', dpi=150, bbox_inches='tight')  # type: ignore
    plt.close(fig)  # type: ignore


def _extend_coils_to_higher_order(
    coils: list, new_order: int, s: SurfaceRZFourier, ncoils: int,
    regularization: Callable | None = None, coil_width: float = 0.4
) -> list:
    """
    Extend coils from a lower Fourier order to a higher order.
    
    This function takes coils optimized at a lower order and extends them to
    a higher order by copying existing Fourier coefficients and padding new
    modes with zeros.
    
    Parameters
    ----------
    coils: list
        List of Coil objects from previous optimization (lower order).
    new_order: int
        Target Fourier order for the extended coils.
    s: SurfaceRZFourier
        Plasma surface (needed for creating new curves).
    ncoils: int
        Number of base coils.
    regularization: Callable, optional
        Regularization function for new coils.
    coil_width: float
        Coil width parameter.
    
    Returns
    -------
    list
        New list of Coil objects with extended Fourier order.
    """
    from simsopt.geo import create_equally_spaced_curves, CurveXYZFourier
    from simsopt.field import coils_via_symmetries
    
    # Get the old order from the first base curve
    old_curves = [coil.curve for coil in coils[:ncoils]]
    old_order = old_curves[0].order if hasattr(old_curves[0], 'order') else len(old_curves[0].dofs) // 3
    
    if new_order <= old_order:
        # No extension needed, return coils as-is
        return coils
    
    # Get major radius for creating new curves
    R0 = s.major_radius()
    R1 = s.get_rc(1, 0) * 3.5
    
    # Create new base curves with higher order
    new_base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=s.stellsym,
        R0=R0, R1=R1, order=new_order, numquadpoints=256
    )
    
    # Copy Fourier coefficients from old curves to new curves
    for old_curve, new_curve in zip(old_curves, new_base_curves):
        if isinstance(old_curve, CurveXYZFourier) and isinstance(new_curve, CurveXYZFourier):
            # Get the dofs (Fourier coefficients) from old curve
            old_dofs = old_curve.get_dofs()
            
            # Get the dofs structure from new curve (initialize to zeros)
            new_dofs = new_curve.get_dofs().copy()
            
            # Structure: For order N, each component has (2*N + 1) dofs:
            # - (N+1) cosine modes: indices 0 to N
            # - N sine modes: indices N+1 to 2*N
            # Components are stored as: [x_dofs, y_dofs, z_dofs]
            old_dofs_per_comp = 2 * old_order + 1
            new_dofs_per_comp = 2 * new_order + 1
            
            # Copy coefficients component by component (x, y, z)
            for comp_idx in range(3):
                old_start = comp_idx * old_dofs_per_comp
                new_start = comp_idx * new_dofs_per_comp
                
                # Copy all matching dofs (cosine + sine modes up to old_order)
                for i in range(old_dofs_per_comp):
                    if old_start + i < len(old_dofs) and new_start + i < len(new_dofs):
                        new_dofs[new_start + i] = old_dofs[old_start + i]
            
            # Set the extended dofs to the new curve
            new_curve.set_dofs(new_dofs)
        else:
            # Fallback: try to copy dofs directly if curves support it
            try:
                old_dofs = old_curve.get_dofs()
                new_dofs = new_curve.get_dofs()
                # Pad with zeros if needed
                if len(old_dofs) < len(new_dofs):
                    padded_dofs = np.zeros_like(new_dofs)
                    padded_dofs[:len(old_dofs)] = old_dofs
                    new_curve.set_dofs(padded_dofs)
                else:
                    new_curve.set_dofs(old_dofs[:len(new_dofs)])
            except (AttributeError, TypeError):
                # If we can't extend, just use the new curve as-is
                pass
    
    # Extract currents from old coils
    base_currents = [coil.current for coil in coils[:ncoils]]
    
    # Create new coils with extended curves
    if regularization is not None:
        regularizations = [regularization(coil_width) for _ in range(ncoils)]
    else:
        regularizations = None
    
    try:
        new_coils = coils_via_symmetries(
            new_base_curves,
            base_currents,
            s.nfp,
            s.stellsym,
            regularizations=regularizations,
        )
    except TypeError:
        new_coils = coils_via_symmetries(new_base_curves, base_currents, s.nfp, s.stellsym)
    
    return new_coils


def optimize_coils_with_fourier_continuation(
    s: SurfaceRZFourier,
    fourier_orders: list[int],
    target_B: float = 5.7,
    out_dir: Path | str = '',
    max_iterations: int = 30,
    ncoils: int = 4,
    verbose: bool = False,
    regularization: Callable | None = regularization_circ,
    coil_objective_terms: Dict[str, Any] | None = None,
    surface_resolution: int = 32,
    case_path: Path | None = None,
    skip_post_processing: bool = False,
    run_vmec: bool = False,
    run_simple: bool = False,
    plot_poincare: bool = False,
    plot_boozer: bool = True,
    plot_finite_build: bool = False,
    finite_build_width: Optional[float] = None,
    finite_build_height: Optional[float] = None,
    **kwargs
) -> tuple[list, Dict[str, Any]]:
    """
    Perform coil optimization with Fourier continuation.
    
    This function solves a sequence of coil optimizations, starting with a low
    number of Fourier modes, converging that problem, and using the solution
    as an initial condition for the next optimization with more Fourier modes.
    
    Parameters
    ----------
    s: SurfaceRZFourier
        Plasma boundary surface.
    fourier_orders: list[int]
        Sequence of Fourier orders to use (e.g., [4, 6, 8]).
        Must be in ascending order.
    target_B: float
        Target magnetic field strength in Tesla (default: 5.7).
    out_dir: Path | str
        Output directory for saved files.
    case_path: Path, optional
        Path to case directory containing case.yaml. Used for post-processing.
    max_iterations: int
        Maximum number of optimization iterations per order (default: 30).
    ncoils: int
        Number of base coils to create (default: 4).
    verbose: bool
        Print out progress and results (default: False).
    regularization: Callable
        Regularization function (default: regularization_circ).
    coil_objective_terms: Dict[str, Any] | None
        Dictionary specifying which objective terms to include.
    surface_resolution: int
        Resolution of plasma surface (nphi=ntheta) for evaluation (default: 32).
        Lower values speed up optimization but reduce accuracy. Use 8 for faster unit tests.
    skip_post_processing: bool
        If True, skip post-processing (QFM, VMEC, Poincaré plots, etc.) after optimization.
        Useful for faster testing and debugging of optimization alone (default: False).
    **kwargs: Additional keyword arguments
        Same as optimize_coils_loop (thresholds, algorithm options, etc.).
        plot_upsample_factor: Factor for upsampling plotting surface (default: 4).
    
    Returns
    -------
    tuple[list, Dict[str, Any]]
        Final optimized coils and combined results dictionary.
    """
    if not fourier_orders:
        raise ValueError("fourier_orders must be a non-empty list")
    
    if not all(isinstance(o, int) and o > 0 for o in fourier_orders):
        raise ValueError("All fourier_orders must be positive integers")
    
    if fourier_orders != sorted(fourier_orders):
        raise ValueError("fourier_orders must be in ascending order")
    
    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    coils: list | None = None
    coil_width = kwargs.get('coil_width', 0.4)
    cached_thresholds: Dict[str, Any] = {}  # Initialize cache for thresholds
    
    print(f"Starting Fourier continuation with orders: {fourier_orders}")
    
    for i, order in enumerate(fourier_orders):
        print(f"\n{'='*60}")
        print(f"Fourier continuation step {i+1}/{len(fourier_orders)}: order={order}")
        print(f"{'='*60}")
        
        # Create subdirectory for this order
        order_dir = out_dir_path / f"order_{order}"
        order_dir.mkdir(exist_ok=True)
        
        if i == 0:
            # First iteration: use standard initialization
            print(f"Initializing coils with order={order}...")
            coils, results = optimize_coils_loop(
                s=s,
                target_B=target_B,
                out_dir=str(order_dir),
                max_iterations=max_iterations,
                ncoils=ncoils,
                order=order,
                verbose=verbose,
                regularization=regularization,
                coil_objective_terms=coil_objective_terms,
                surface_resolution=surface_resolution,
                skip_post_processing=True,  # Skip post-processing for intermediate orders
                **kwargs
            )
            # Extract cached thresholds from first step for reuse in continuation
            cached_thresholds = results.get('_cached_thresholds', {})
        else:
            # Subsequent iterations: extend previous solution
            if coils is None:
                raise RuntimeError("Cannot extend coils: previous step produced None coils")
            print(f"Extending coils from order={fourier_orders[i-1]} to order={order}...")
            coils = _extend_coils_to_higher_order(
                coils, order, s, ncoils, regularization, coil_width
            )
            
            # Optimize with extended coils as initial condition
            # Pass cached thresholds to avoid recalculating them
            continuation_kwargs = kwargs.copy()
            if cached_thresholds:
                continuation_kwargs['_cached_thresholds'] = cached_thresholds
            
            print(f"Optimizing with extended coils (order={order})...")
            coils, results = optimize_coils_loop(
                s=s,
                target_B=target_B,
                out_dir=str(order_dir),
                max_iterations=max_iterations,
                ncoils=ncoils,
                order=order,
                verbose=verbose,
                regularization=regularization,
                coil_objective_terms=coil_objective_terms,
                initial_coils=coils,  # Pass extended coils as initial condition
                surface_resolution=surface_resolution,
                skip_post_processing=True,  # Skip post-processing for intermediate orders
                **continuation_kwargs
            )
        
        # Store results for this order
        results['fourier_order'] = order
        results['continuation_step'] = i + 1
        all_results.append(results)
        
        # Note: optimize_coils_loop already saves VTK files (coils_optimized) and 
        # Bn error PDF (bn_error_3d_plot.pdf) in order_dir, so they're automatically
        # saved after each continuation step. We just need to ensure they're accessible.
        # The files are saved to: order_dir/coils_optimized.* and order_dir/bn_error_3d_plot.pdf
    
    # Combine results from all continuation steps
    combined_results = {
        'fourier_continuation': True,
        'fourier_orders': fourier_orders,
        'final_order': fourier_orders[-1],
        'continuation_results': all_results,
        **all_results[-1]  # Include final step results at top level
    }
    
    print(f"\n{'='*60}")
    print("Fourier continuation completed!")
    print(f"Final order: {fourier_orders[-1]}")
    print(f"{'='*60}\n")
    
    if coils is None:
        raise RuntimeError("Fourier continuation failed: no coils were produced")
    
    # Run post-processing on final optimized coils
    # Use the final order's BiotSavart object if available
    if not skip_post_processing:
        try:
            from .post_processing import run_post_processing
            import yaml as yaml_module
            
            # Find case YAML file - try case_path first if provided
            case_yaml_path = None
            if case_path is not None:
                case_path_obj = Path(case_path)
            if case_path_obj.is_file():
                # It's already the YAML file
                case_yaml_path = case_path_obj.resolve()
            elif case_path_obj.is_dir():
                # It's a directory, look for case.yaml inside
                case_yaml_path = (case_path_obj / "case.yaml").resolve()
                if not case_yaml_path.exists():
                    case_yaml_path = None
            else:
                # Try to resolve it (might be relative path)
                if case_path_obj.exists():
                    case_yaml_path = case_path_obj.resolve()
                else:
                    # Try as directory with case.yaml
                    case_yaml_path = (case_path_obj / "case.yaml").resolve()
                    if not case_yaml_path.exists():
                        case_yaml_path = None
            
            # Try in out_dir if not found yet
            if case_yaml_path is None or not case_yaml_path.exists():
                case_yaml_path = out_dir_path / "case.yaml"
            if not case_yaml_path.exists():
                case_yaml_path = out_dir_path.parent / "case.yaml"
            if not case_yaml_path.exists() and hasattr(s, 'filename') and s.filename:
                # Try to find case YAML relative to the surface file
                surface_dir = Path(s.filename).parent
                surface_stem = Path(s.filename).stem.replace("input.", "").replace(".focus", "")
                potential_case_paths = [
                    surface_dir / "case.yaml",
                    surface_dir.parent / "case.yaml",
                    Path("cases") / surface_stem / "case.yaml",
                ]
                for path in potential_case_paths:
                    if path.exists():
                        case_yaml_path = path
                        break
            
            # If still not found, search for case YAML files that reference this surface
            if case_yaml_path is None or not case_yaml_path.exists():
                cases_dir = Path("cases")
                if cases_dir.exists():
                    surface_filename = Path(s.filename).name if hasattr(s, 'filename') and s.filename else ""
                    for yaml_file in cases_dir.glob("*.yaml"):
                        try:
                            case_data = yaml_module.safe_load(yaml_file.read_text())
                            if case_data and isinstance(case_data, dict):
                                surface_in_case = case_data.get("surface_params", {}).get("surface", "")
                                # Check if this case references the same surface file
                                if surface_filename and surface_filename in surface_in_case:
                                    case_yaml_path = yaml_file
                                    break
                                elif surface_in_case in surface_filename:
                                    case_yaml_path = yaml_file
                                    break
                        except Exception:
                            continue
            
            # Coils JSON path - should be in the final order directory
            # For Fourier continuation, the biot_savart_optimized.json is saved in the final order_dir
            final_order_dir = out_dir_path / f"order_{fourier_orders[-1]}"
            coils_json_path = final_order_dir / "biot_savart_optimized.json"
            if not coils_json_path.exists():
                # Fallback: try main out_dir
                coils_json_path = out_dir_path / "biot_savart_optimized.json"
            if not coils_json_path.exists():
                # Also check for coils.json (used by submit-case CLI)
                coils_json_path = out_dir_path / "coils.json"
            
            if coils_json_path.exists():
                print("\nRunning post-processing on final optimized coils (QFM, Poincaré plots, profiles)...")
                
                # Determine helicity_n based on surface type (QA=0, QH=-1)
                helicity_n = 0
                if case_yaml_path.exists():
                    import yaml
                    try:
                        case_data = yaml.safe_load(case_yaml_path.read_text())
                        surface_name = case_data.get("surface_params", {}).get("surface", "").lower()
                        if "qh" in surface_name or "qash" in surface_name:
                            helicity_n = -1
                    except Exception:
                        pass
                
                # Determine plasma_surfaces_dir - go up from output directory to find repo root
                plasma_surfaces_dir = None
                current_dir = out_dir_path
                for _ in range(5):  # Search up to 5 levels
                    potential_plasma_dir = current_dir / "plasma_surfaces"
                    if potential_plasma_dir.exists():
                        plasma_surfaces_dir = potential_plasma_dir
                        break
                    if current_dir.parent == current_dir:  # Reached root
                        break
                    current_dir = current_dir.parent
                
                # Save post-processing outputs to main output directory (same level as order subdirectories)
                # This ensures QFM surface, Poincaré plots, etc. are easily accessible
                post_processing_results = run_post_processing(
                    coils_json_path=coils_json_path,
                    output_dir=out_dir_path,  # Save plots in main output directory
                    case_yaml_path=case_yaml_path if case_yaml_path.exists() else None,
                    plasma_surfaces_dir=plasma_surfaces_dir,  # Pass repo root plasma_surfaces directory
                    run_vmec=run_vmec,
                    helicity_m=1,
                    helicity_n=helicity_n,
                    ns=50,
                    plot_boozer=plot_boozer,
                    plot_poincare=plot_poincare,
                    nfieldlines=20,
                    run_simple=run_simple,
                    plot_finite_build=plot_finite_build,
                    finite_build_width=finite_build_width,
                    finite_build_height=finite_build_height,
                )
                print("Post-processing complete!")
                if 'quasisymmetry_average' in post_processing_results:
                    print(f"  Average quasisymmetry error: {post_processing_results['quasisymmetry_average']:.2e}")
            else:
                print(f"Warning: Skipping post-processing (coils_json not found: {coils_json_path})")
                post_processing_results = {}  # Initialize empty dict if post-processing skipped
        except Exception as e:
            print(f"Warning: Post-processing failed: {e}")
            import traceback
            traceback.print_exc()
            post_processing_results = {}  # Initialize empty dict if post-processing failed
    else:
        post_processing_results = {}  # Skip post-processing if flag is set
    
    # Merge post-processing results into combined_results
    if post_processing_results:
        # Only include numeric/metric values, not objects like 'vmec' or 'qfm_surface'
        for key, value in post_processing_results.items():
            if key in ['quasisymmetry_average', 'loss_fraction', 'BdotN', 'BdotN_over_B']:
                if isinstance(value, (int, float)):
                    combined_results[key] = float(value)
    
    return coils, combined_results


def _is_ci_running() -> bool:
    """
    Check if the code is running in a CI environment.
    
    Returns:
        True if running in CI (GitHub Actions, GitLab CI, Jenkins, etc.), False otherwise.
    """
    ci_env_vars = ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS_URL', 
                   'TRAVIS', 'CIRCLECI', 'APPVEYOR', 'BUILDKITE']
    return any(os.getenv(var) for var in ci_env_vars)


@contextmanager
def _nullcontext():
    """Null context manager that does nothing."""
    yield


@contextmanager
def _redirect_verbose_to_file(output_file: Path):
    """
    Context manager to redirect stdout to a file while preserving stderr.
    
    Args:
        output_file: Path to the file where stdout should be written.
    """
    original_stdout = sys.stdout
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            yield
    finally:
        sys.stdout = original_stdout


def optimize_coils_dipole_loop(
    s: SurfaceRZFourier,
    surface_file: str,
    surface_params: Dict[str, Any],
    out_dir: Path | str = "",
    max_iterations: int = 100,
    verbose: bool = False,
    skip_post_processing: bool = False,
    case_path: Path | None = None,
    run_vmec: bool = False,
    run_simple: bool = False,
    plot_poincare: bool = False,
    plot_boozer: bool = True,
    tf_configuration: str = "LandremanPaulQA",
    Nx: int = 4,
    dipole_order: int = 2,
    **kwargs,
) -> tuple:
    """
    Optimize dipole coils plus TF coils, modeled after dipole_array_tutorial_advanced.py.

    Uses initialize_coils_dipole and dipole_array_optimization_function from simsopt.
    """
    from scipy.optimize import minimize
    from simsopt.geo import (
        CurveLength,
        CurveCurveDistance,
        CurveSurfaceDistance,
        LinkingNumber,
        MeanSquaredCurvature,
        LpCurveCurvature,
    )
    from simsopt.field import BiotSavart, coils_to_vtk
    from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

    try:
        from simsopt.util import dipole_array_optimization_function, save_coil_sets
    except ImportError as e:
        raise ImportError(
            "Dipole coil optimization requires simsopt with auglag_coils branch "
            "(dipole_array_helper_functions). Install from: "
            "https://github.com/hiddenSymmetries/simsopt"
        ) from e

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with timed_section("coil_initialization", print_time=False):
        with suppress_output():
            coils, base_curves, base_curves_TF, ncoils_dipole, ncoils_TF = initialize_coils_dipole(
                s,
                surface_file=surface_file,
                surface_params=surface_params,
                tf_configuration=tf_configuration,
                Nx=Nx,
                dipole_order=dipole_order,
                out_dir=out_dir,
                **{k: v for k, v in kwargs.items() if k in ("poff", "coff", "Ny", "Nz", "dipole_coil_size", "tf_coil_size", "remove_inboard_eps")},
            )

    coils_TF = coils[ncoils_dipole:]
    base_coils_TF = coils_TF[:ncoils_TF]
    base_coils = coils[:ncoils_dipole]
    curves = [c.curve for c in coils]
    curves_TF = [c.curve for c in coils_TF]
    all_coils = coils
    all_base_coils = base_coils + base_coils_TF

    bs_TF = BiotSavart(coils_TF)
    bs_dipole = BiotSavart(coils[:ncoils_dipole])
    btot = bs_dipole + bs_TF
    eval_points = s.gamma().reshape(-1, 3)
    btot.set_points(eval_points)

    try:
        coils_to_vtk(all_coils, out_dir / "coils_initial")
    except Exception as e:
        print(f"Warning: Failed to save initial coils to VTK: {e}")

    LENGTH_WEIGHT = Weight(0.01)
    LENGTH_WEIGHT2 = Weight(0.01)
    LENGTH_TARGET = 85
    LINK_WEIGHT = 1e4
    CC_THRESHOLD = 0.8
    CC_WEIGHT = 1e2
    CS_THRESHOLD = 1.3
    CS_WEIGHT = 1e1
    CURVATURE_THRESHOLD = 0.5
    MSC_THRESHOLD = 0.05
    CURVATURE_WEIGHT = 1e-2
    MSC_WEIGHT = 1e-1

    Jf = SquaredFlux(s, btot)
    Jls = [CurveLength(c) for c in base_curves]
    Jls_TF = [CurveLength(c) for c in base_curves_TF]
    Jlength = QuadraticPenalty(sum(Jls_TF), LENGTH_TARGET, "max")
    Jlength2 = QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max")
    Jccdist = CurveCurveDistance(curves + curves_TF, CC_THRESHOLD / 2.0, num_basecurves=len(all_coils))
    Jccdist2 = CurveCurveDistance(curves_TF, CC_THRESHOLD, num_basecurves=len(coils_TF))
    Jcsdist = CurveSurfaceDistance(curves + curves_TF, s, CS_THRESHOLD)
    linkNum = LinkingNumber(curves + curves_TF, downsample=2)
    Jcs = [LpCurveCurvature(c.curve, 2, CURVATURE_THRESHOLD) for c in base_coils_TF]
    Jmscs = [MeanSquaredCurvature(c.curve) for c in base_coils_TF]

    class _ZeroObj:
        def J(self): return 0.0

    try:
        from simsopt.field.force import LpCurveForce, LpCurveTorque
        Jforce = LpCurveForce(base_coils_TF, source_coils_coarse=coils[:ncoils_dipole], source_coils_fine=coils_TF, downsample=2) + LpCurveForce(base_coils, source_coils_coarse=coils[:ncoils_dipole], source_coils_fine=coils_TF, downsample=2)
        Jtorque = LpCurveTorque(base_coils_TF, source_coils_coarse=coils[:ncoils_dipole], source_coils_fine=coils_TF, downsample=2) + LpCurveTorque(base_coils, source_coils_coarse=coils[:ncoils_dipole], source_coils_fine=coils_TF, downsample=2)
        Jforce2 = Jtorque2 = _ZeroObj()
    except (ImportError, TypeError):
        Jforce = Jforce2 = Jtorque = Jtorque2 = _ZeroObj()

    JF = (
        Jf
        + CC_WEIGHT * Jccdist
        + CC_WEIGHT * Jccdist2
        + CS_WEIGHT * Jcsdist
        + CURVATURE_WEIGHT * sum(Jcs)
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
        + LINK_WEIGHT * linkNum
        + LENGTH_WEIGHT * Jlength
        + LENGTH_WEIGHT2 * Jlength2
    )

    obj_dict = {
        "JF": JF,
        "Jf": Jf,
        "Jlength": Jlength,
        "Jlength2": Jlength2,
        "Jls": Jls,
        "Jls_TF": Jls_TF,
        "Jcs": Jcs,
        "Jmscs": Jmscs,
        "Jccdist": Jccdist,
        "Jccdist2": Jccdist2,
        "Jcsdist": Jcsdist,
        "linkNum": linkNum,
        "Jforce": 0,
        "Jforce2": 0,
        "Jtorque": 0,
        "Jtorque2": 0,
        "btot": btot,
        "s": s,
        "base_curves_TF": base_curves_TF,
        "psc_array": None,
    }
    weight_dict = {
        "length_weight": LENGTH_WEIGHT.value,
        "curvature_weight": CURVATURE_WEIGHT,
        "msc_weight": MSC_WEIGHT,
        "msc_threshold": MSC_THRESHOLD,
        "cc_weight": CC_WEIGHT,
        "cs_weight": CS_WEIGHT,
        "link_weight": LINK_WEIGHT,
        "force_weight": 0.0,
        "torque_weight": 0.0,
        "net_force_weight": 0.0,
        "net_torque_weight": 0.0,
    }

    dofs = JF.x
    res = minimize(
        dipole_array_optimization_function,
        dofs,
        args=(obj_dict, weight_dict, None),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "maxcor": 1000},
        tol=1e-20,
    )

    save_coil_sets(btot, str(out_dir) + "/", "_optimized")

    combined_results = {"success": res.success, "fun": float(res.fun), "nit": int(res.nit)}
    return coils, combined_results


def optimize_coils_loop(
    s : SurfaceRZFourier, target_B : float = 5.7, out_dir : Path | str = '', 
    max_iterations : int = 30, 
    ncoils : int = 4, order : int = 16, 
    verbose : bool = False,
    regularization : Callable | None = regularization_circ, 
    coil_objective_terms: Dict[str, Any] | None = None,
    initial_coils: list | None = None,
    surface_resolution: int = 32,
    skip_post_processing: bool = False,
    case_path: Path | None = None,
    run_vmec: bool = False,
    run_simple: bool = False,
    plot_poincare: bool = True,
    plot_boozer: bool = True,
    **kwargs):
    """
    Performs complete coil optimization including initialization and optimization.
    This function initializes coils with the target B-field and then optimizes
    them using the augmented Lagrangian method.

    Args:
        s: plasma boundary surface.
        target_B: Target magnetic field strength in Tesla (default: 5.7).
        out_dir: Path or string for the output directory for saved files.
        max_iterations: Maximum number of optimization iterations (default: 1500).
        ncoils: Number of base coils to create (default: 4).
        order: Fourier order for coil curves (default: 16).
        verbose: Print out progress and results (default: False).
        surface_resolution: Resolution of plasma surface (nphi=ntheta) for evaluation (default: 32).
            Lower values speed up optimization but reduce accuracy. Use 8 for faster unit tests.
        **kwargs: Additional keyword arguments for constraint thresholds.
            max_iter_subopt: Maximum number of suboptimization iterations (default: max_iterations // 50).
            length_threshold: Threshold for the length objective (default: 200.0).
            flux_threshold: Threshold for the flux objective (default: 1e-8).
            cc_threshold: Threshold for the coil-coil distance objective (default: 1.0).
            cs_threshold: Threshold for the coil-surface distance objective (default: 1.3).
            msc_threshold: Threshold for the mean squared curvature objective (default: 1.0).
            curvature_threshold: Threshold for the curvature objective (default: 1.0).
            force_threshold: Threshold for the coil force objective (default: 1.0).
            torque_threshold: Threshold for the coil torque objective (default: 1.0).
    Returns:
        coils: List of optimized Coil class objects.
        results: Dictionary containing optimization results and metrics.
    """
    out_dir = Path(out_dir).resolve()
    
    # If verbose=True and CI is running, redirect output to a file so the job log
    # is not flooded. Set STELLCOILBENCH_CI_VERBOSE_STDOUT=1 to keep verbose on
    # stdout (so the per-case .log file in the workflow gets it for progress display).
    verbose_output_file = None
    if verbose and _is_ci_running() and not os.getenv("STELLCOILBENCH_CI_VERBOSE_STDOUT"):
        verbose_output_file = out_dir / "verbose_output.txt"
    
    # Use context manager to redirect output when needed
    redirect_context = _redirect_verbose_to_file(verbose_output_file) if verbose_output_file else _nullcontext()
    
    with redirect_context:
        return _optimize_coils_loop_impl(
            s, target_B, out_dir, max_iterations, ncoils, order, verbose,
            regularization, coil_objective_terms, initial_coils, surface_resolution,
            skip_post_processing, case_path, run_vmec, run_simple, plot_poincare,
            plot_boozer, **kwargs
        )


def _optimize_coils_loop_impl(
    s : SurfaceRZFourier, target_B : float = 5.7, out_dir : Path | str = '', 
    max_iterations : int = 30, 
    ncoils : int = 4, order : int = 16, 
    verbose : bool = False,
    regularization : Callable | None = regularization_circ, 
    coil_objective_terms: Dict[str, Any] | None = None,
    initial_coils: list | None = None,
    surface_resolution: int = 32,
    skip_post_processing: bool = False,
    case_path: Path | None = None,
    run_vmec: bool = False,
    run_simple: bool = False,
    plot_poincare: bool = False,
    plot_boozer: bool = True,
    **kwargs):
    """
    Internal implementation of optimize_coils_loop.
    This function contains the actual optimization logic.
    """
    import time
    from scipy.optimize import minimize
    from simsopt.geo import SurfaceRZFourier
    from simsopt.geo import LinkingNumber, CurveLength, CurveCurveDistance, ArclengthVariation
    from simsopt.geo import LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature
    from simsopt.objectives import SquaredFlux, QuadraticPenalty, Weight
    from simsopt.field import BiotSavart, coils_to_vtk
    from simsopt.field.force import LpCurveForce, LpCurveTorque
    from simsopt.util import calculate_modB_on_major_radius

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # nturns is constant for all cases (defined here so it's always available)
    nturns = 200 # nturns = 200 to give reasonable upper bound on reactor scale forces
    
    # Check if this is a continuation step (initial_coils provided) to avoid duplicate work
    is_continuation_step = initial_coils is not None
    
    # For continuation steps, reuse pre-computed thresholds/weights from kwargs if available
    # This avoids recalculating thresholds and weights that don't change between continuation steps
    if is_continuation_step and '_cached_thresholds' in kwargs:
        # Use cached thresholds from first step
        cached = kwargs['_cached_thresholds']
        length_threshold = cached['length_threshold']
        flux_threshold = cached['flux_threshold']
        cc_threshold = cached['cc_threshold']
        cs_threshold = cached['cs_threshold']
        msc_threshold = cached['msc_threshold']
        arclength_variation_threshold = cached['arclength_variation_threshold']
        curvature_threshold = cached['curvature_threshold']
        force_threshold = cached['force_threshold']
        torque_threshold = cached['torque_threshold']
        coil_width = cached['coil_width']
        # a0 = minor-radius scale factor (ARIES_CS_MINOR_RADIUS / minor_radius); support legacy "R0" key
        a0 = cached.get('a0', cached.get('R0'))
        major_radius = cached.get('major_radius', s.major_radius())
        minor_radius = cached.get('minor_radius', ARIES_CS_MINOR_RADIUS / a0 if a0 and a0 != 0 else 1.7)
    else:
        # First step or no cache: compute thresholds normally
        # Set default constraint thresholds if not provided
        # Defaults here are for ARIES-CS reactor (minor radius 1.7 m, 5.7 T)
        length_threshold = kwargs.get('length_threshold', 200.0)
        flux_threshold = kwargs.get('flux_threshold', 1e-8)
        cc_threshold = kwargs.get('cc_threshold', 0.8)
        cs_threshold = kwargs.get('cs_threshold', 1.3)
        msc_threshold = kwargs.get('msc_threshold', 1.0)
        arclength_variation_threshold = kwargs.get('arclength_variation_threshold', 0.0)
        curvature_threshold = kwargs.get('curvature_threshold', 1.0)

        coil_width = 0.4  # 0.4 m at reactor-scale is the default coil width
        force_threshold = kwargs.get('force_threshold', 1.0) * nturns
        torque_threshold = kwargs.get('torque_threshold', 1.0) * nturns

        # Rescale thresholds by plasma minor radius (consistent with post_processing vmec_RZ_scale)
        # a0 = ARIES_CS_MINOR_RADIUS / minor_radius scales device-size thresholds to reactor scale
        major_radius = s.major_radius()  # Major radius in meters [L]
        minor_radius = float(s.minor_radius())  # Minor radius in meters [L]
        a0 = ARIES_CS_MINOR_RADIUS / minor_radius  # Minor-radius scale factor (same convention as vmec_RZ_scale)
        if 'length_threshold' not in kwargs:
            length_threshold /= a0
        if 'cc_threshold' not in kwargs:
            cc_threshold /= a0
        if 'cs_threshold' not in kwargs:
            cs_threshold /= a0
        if 'curvature_threshold' not in kwargs:
            curvature_threshold *= a0
        if 'msc_threshold' not in kwargs:
            msc_threshold *= a0
        if 'arclength_variation_threshold' not in kwargs:
            arclength_variation_threshold *= a0 ** 2
        # coil_width is not a threshold parameter, so always scale it
        coil_width /= a0

    # CI autopilot hard cap: clamp max_iterations to 10 000
    _CI_MAX_ITER_CAP = 10_000
    if max_iterations > _CI_MAX_ITER_CAP:
        print(
            f"Warning: max_iterations ({max_iterations}) exceeds CI cap "
            f"({_CI_MAX_ITER_CAP}); clamping."
        )
        max_iterations = _CI_MAX_ITER_CAP

    # If there is a suboptimization, set the max iterations 
    max_iter_subopt = kwargs.get('max_iter_subopt', max_iterations // 50)
    algorithm = kwargs.get('algorithm', 'augmented_lagrangian')
    
    # Normalize algorithm name (handle case variations)
    if isinstance(algorithm, str):
        algorithm_lower = algorithm.lower()
        if algorithm_lower in ['l-bfgs', 'lbfgs', 'l-bfgs-b']:
            algorithm = 'L-BFGS-B'
        elif algorithm_lower == 'augmented_lagrangian':
            algorithm = 'augmented_lagrangian'
        # Keep other algorithm names as-is (they should match scipy method names)
    
    # Extract algorithm-specific options from kwargs
    # These will be passed to scipy.minimize for scipy algorithms
    # First check for nested 'algorithm_options' dict
    algorithm_options = kwargs.get('algorithm_options', {}).copy()
    
    # Also look for algorithm-specific options directly in kwargs (for convenience)
    # This allows users to specify options like 'maxls' directly in optimizer_params
    # instead of requiring a nested algorithm_options dict
    valid_algo_options = _get_scipy_algorithm_options(algorithm)
    for opt_name in valid_algo_options:
        if opt_name in kwargs and opt_name not in algorithm_options:
            algorithm_options[opt_name] = kwargs[opt_name]

    # Step 1: Initialize coils with target B-field
    with timed_section("coil_initialization", print_time=False):
        if initial_coils is None:
            with suppress_output():
                coils = initialize_coils_loop(s, out_dir=out_dir, target_B=target_B, ncoils=ncoils, order=order, coil_width=coil_width, regularization=regularization)
            # Apply random DOF perturbation to break determinism if requested
            dof_perturbation = kwargs.get('dof_perturbation', 0.0)
            if isinstance(dof_perturbation, (int, float)) and dof_perturbation > 0:
                print(f"  Applying DOF perturbation with scale {dof_perturbation}")
                for coil in coils[:ncoils]:
                    x = coil.curve.x
                    noise = np.random.randn(len(x)) * dof_perturbation * np.std(x)
                    coil.curve.x = x + noise
        else:
            coils = initial_coils

    # Calculate total_current (needed for threshold scaling)
    # Sum the unique base coils (coils[:ncoils]) to get total current
    total_current = sum([c.current.get_value() for c in coils[:ncoils]])
    
    # Calculate current_scale_factor for force/torque threshold and weight scaling
    # This makes force/torque thresholds and weights dimensionless relative to reactor scale
    current_scale_factor = 1.0  # Default: no scaling
    total_current_reactor_scale = None  # Will be set if needed for weight scaling
    if not is_continuation_step and ('force_threshold' not in kwargs or 'torque_threshold' not in kwargs):
        with suppress_output():
            coils_backup = initialize_coils_loop(s, out_dir=out_dir, ncoils=ncoils, order=order, coil_width=coil_width, regularization=regularization)
        # Sum the unique base coils to get total current
        total_current_reactor_scale = sum([c.current.get_value() for c in coils_backup[:ncoils]])
        current_scale_factor = (total_current / total_current_reactor_scale) ** 2
        if 'force_threshold' not in kwargs:
            force_threshold *= current_scale_factor
        if 'torque_threshold' not in kwargs:
            torque_threshold *= current_scale_factor

    # Extract base curves and currents from the initialized coils
    base_curves = [coil.curve for coil in coils[:ncoils]]
    
    # Step 2: Create plotting surface for visualization
    # Use surface_resolution for plotting (can be upsampled, but respect the surface_resolution parameter)
    # For tests, use lower upsampling factor to speed things up
    plot_upsample_factor = kwargs.get('plot_upsample_factor', 2)
    # Use surface_resolution directly, don't override with len(s.quadpoints_phi) which may be higher
    base_resolution = surface_resolution
    qphi = plot_upsample_factor * base_resolution
    qtheta = plot_upsample_factor * base_resolution
    quadpoints_phi = np.linspace(0, 1, qphi)
    quadpoints_theta = np.linspace(0, 1, qtheta)
    
    # Create a plotting surface (full torus)
    # Handle case where surface was created manually (no filename)
    if hasattr(s, 'filename') and s.filename is not None:
        s_plot = SurfaceRZFourier.from_vmec_input(
            s.filename,
            range="full torus",
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            nfp=s.nfp,
            stellsym=s.stellsym
        )
    else:
        # Create surface manually with same parameters
        s_plot = SurfaceRZFourier(
            nfp=s.nfp,
            stellsym=s.stellsym,
            mpol=s.mpol,
            ntor=s.ntor,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta
        )
    
    # Copy the surface coefficients
    for m in range(s.mpol + 1):
        for n in range(-s.ntor, s.ntor + 1):
            if s.get_rc(m, n) != 0:
                s_plot.set_rc(m, n, s.get_rc(m, n))
            if s.get_zs(m, n) != 0:
                s_plot.set_zs(m, n, s.get_zs(m, n))

    # Step 3: Create BiotSavart object and save initial state
    with timed_section("biotsavart_setup", print_time=False):
        bs = BiotSavart(coils)
        with suppress_output():
            calculate_modB_on_major_radius(bs, s)
        curves = [c.curve for c in coils]
        
        # Save initial coils
        try:
            coils_to_vtk(coils, out_dir / "coils_initial")
        except Exception as e:
            print(f"Warning: Failed to save initial coils to VTK: {e}")
            print("  Continuing optimization without VTK export...")
        
        # Calculate initial B-field (used for surface data)
        bs.set_points(s_plot.gamma().reshape((-1, 3)))
        with suppress_output():
            B_initial = calculate_modB_on_major_radius(bs, s_plot)
        
        # Save initial surface data
        bs.set_points(s_plot.gamma().reshape((-1, 3)))
        pointData = {
            "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                              s_plot.unitnormal(), axis=2)[:, :, None] / 
                            bs.AbsB().reshape((qphi, qtheta, 1)),
            "modB": bs.AbsB().reshape((qphi, qtheta, 1))
        }
        s_plot.to_vtk(out_dir / "surface_initial", extra_data=pointData)
        
        # Generate 3D visualization plot for initial coils
        try:
            _plot_bn_error_3d(
                s_plot,
                bs,
                coils,
                out_dir,
                filename="bn_error_3d_plot_initial.pdf",
                title="B_N/|B| Error on Plasma Surface with Initial Coils",
            )
        except Exception as e:
            print(f"Warning: Failed to generate initial 3D plot: {e}")

    # Step 4: Define objective function and constraints
    objective_setup_start = time.perf_counter()
    bs.set_points(s.gamma().reshape((-1, 3)))
    
    # Main objective: Squared flux (always included)
    # If virtual casing target is provided, use B_external_normal as the target
    # Otherwise use default (zero normal field)
    vc_target = kwargs.get('vc_target', None)
    if vc_target is not None:
        print(f"Using virtual casing target for SquaredFlux (target shape: {vc_target.shape})")
        Jf = SquaredFlux(s, bs, target=vc_target, threshold=flux_threshold)
    else:
        Jf = SquaredFlux(s, bs, threshold=flux_threshold)
    
    # Build constraint terms based on coil_objective_terms configuration
    # If coil_objective_terms is None or empty, omit all constraint objectives (only flux objective included)
    # Only explicitly specified objectives in coil_objective_terms will be included
    
    # Prepare all constraint objects (create them regardless, but only add to c_list if specified)
    Jls = [CurveLength(c) for c in base_curves]
    
    # Get p values for lp terms (default to 2)
    curvature_p = coil_objective_terms.get("coil_curvature_p", 2) if coil_objective_terms else 2
    force_p = coil_objective_terms.get("coil_coil_force_p", 2) if coil_objective_terms else 2
    torque_p = coil_objective_terms.get("coil_coil_torque_p", 2) if coil_objective_terms else 2
    
    # Determine thresholds for distance and force/torque terms based on options
    # Default to using thresholds (for backward compatibility with default behavior)
    cc_thresh = cc_threshold
    cs_thresh = cs_threshold
    force_thresh = force_threshold
    torque_thresh = torque_threshold
    
    # coil_coil_distance and coil_surface_distance are always included automatically
    # They use CurveCurveDistance and CurveSurfaceDistance which handle thresholding internally
    # No need to check coil_objective_terms for these - they're always enabled
    
    # Check if l1 (no threshold) or l1_threshold is specified for force/torque
    # Only adjust thresholds if the term is explicitly specified in coil_objective_terms
    if coil_objective_terms:
        
        coil_force_option = coil_objective_terms.get("coil_coil_force")
        if coil_force_option and "threshold" in coil_force_option:
            force_thresh = force_threshold
        else:
            force_thresh = 0.0
        
        coil_torque_option = coil_objective_terms.get("coil_coil_torque")
        if coil_torque_option and "threshold" in coil_torque_option:
            torque_thresh = torque_threshold
        else:
            torque_thresh = 0.0
    
    # Create distance and force/torque objects with appropriate thresholds
    Jccdist = CurveCurveDistance(curves, cc_thresh, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, cs_thresh)
    Jalenvar = [ArclengthVariation(c) for c in base_curves]
    Jcs = [LpCurveCurvature(c, 2, curvature_threshold) for c in base_curves]
    Jlink = LinkingNumber(curves, downsample=2)
    Jforce = LpCurveForce(coils[:ncoils], coils, p=force_p, threshold=force_thresh, downsample=2)
    Jtorque = LpCurveTorque(coils[:ncoils], coils, p=torque_p, threshold=torque_thresh, downsample=2)
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    
    # Update curvature with correct p value if specified
    if coil_objective_terms and curvature_p != 2:
        Jcs = [LpCurveCurvature(c, curvature_p, curvature_threshold) for c in base_curves]

    # Print initial constraint values and weights (will be updated after building c_list and weights)
    # This will be printed after weights are determined
    
    # Build constraint list dynamically based on coil_objective_terms
    # coil_coil_distance and coil_surface_distance are always included automatically
    # Other objectives are only included if explicitly specified in coil_objective_terms
    # If coil_objective_terms is None or empty, only flux and distance objectives are included
    c_list = [Jf]  # Always include flux
    
    # Always include coil_coil_distance and coil_surface_distance
    # These use CurveCurveDistance and CurveSurfaceDistance which handle thresholding internally
    cc_distance_idx = len(c_list)
    c_list.append(Jccdist)
    cs_distance_idx = len(c_list)
    c_list.append(Jcsdist)
    
    # Track constraint names and thresholds for printing
    constraint_names_and_thresholds = []
    constraint_names_and_thresholds.append(("CC Distance", cc_threshold))
    constraint_names_and_thresholds.append(("CS Distance", cs_threshold))
    
    # Track index of coil_surface_distance and coil_coil_distance constraints for heavy weighting
    cs_distance_index = cs_distance_idx
    cc_distance_index = cc_distance_idx
    
    # Build constraint list based on coil_objective_terms
    # Map term names to constraint objects and penalty types
    # Note: Thresholds for l1/l1_threshold/lp/lp_threshold are already set during object creation
    # Only l2/l2_threshold options need QuadraticPenalty wrapping
    # Initialize term_map (empty if no coil_objective_terms)
    term_map = {}
    if coil_objective_terms:
        term_map = {
            "total_length": {
                "obj": sum(Jls),
                "threshold": length_threshold,
                "l1": lambda obj, thresh: obj,
                "l1_threshold": lambda obj, thresh: obj,  # max(obj - threshold, 0)
                "l2": lambda obj, thresh: QuadraticPenalty(obj, 0.0, "max"),
                "l2_threshold": lambda obj, thresh: QuadraticPenalty(obj, thresh, "max"),
            },
            "coil_curvature": {
                "obj": sum(Jcs),
                "threshold": curvature_threshold,
                "lp": lambda obj, thresh: obj,  # Threshold already set in object creation
                "lp_threshold": lambda obj, thresh: obj,  # Threshold already set in object creation
            },
            "coil_arclength_variation": {
                "obj": Jalenvar,
                "threshold": arclength_variation_threshold,
                "l2": lambda obj, thresh: sum([QuadraticPenalty(j, 0.0, "max") for j in obj]),
                "l2_threshold": lambda obj, thresh: sum([QuadraticPenalty(j, thresh, "max") for j in obj]),
                "l1": lambda obj, thresh: sum(obj),
                "l1_threshold": lambda obj, thresh: sum(obj)
            },
            "coil_mean_squared_curvature": {
                "obj": Jmscs,
                "threshold": msc_threshold,
                "l2": lambda obj, thresh: sum([QuadraticPenalty(j, 0.0, "max") for j in obj]),
                "l2_threshold": lambda obj, thresh: sum([QuadraticPenalty(j, thresh, "max") for j in obj]),
                "l1": lambda obj, thresh: sum(obj),
                "l1_threshold": lambda obj, thresh: sum(obj)
            },
            "linking_number": {
                "obj": Jlink,
                "threshold": None,
                "": lambda obj, thresh: obj,
            },
            "coil_coil_force": {
                "obj": Jforce,
                "threshold": force_threshold,
                "lp": lambda obj, thresh: obj,  # Threshold already set to 0.0 in object creation
                "lp_threshold": lambda obj, thresh: obj,  # Threshold already set in object creation
            },
            "coil_coil_torque": {
                "obj": Jtorque,
                "threshold": torque_threshold,
                "lp": lambda obj, thresh: obj,  # Threshold already set to 0.0 in object creation
                "lp_threshold": lambda obj, thresh: obj,  # Threshold already set in object creation
            },
        }
    
    # Map constraint indices to their scaling factors for dimensionless weights
    # Use major_radius (with units [L]) for proper dimensional scaling
    constraint_scaling = {}  # Maps constraint index to scaling factor
    constraint_idx_to_term = {}  # Maps constraint index to term name for named weights
    major_radius = s.major_radius()  # Major radius in meters [L]
    
    # Add scaling for always-included distance objectives
    # CurveCurveDistance and CurveSurfaceDistance compute squared penalties, so units are [L^2]
    # Weight scaling = 1 / (major_radius^2) to make weight * constraint dimensionless
    constraint_scaling[cc_distance_idx] = 1.0 / (major_radius ** 2)  # [L^2] -> weight [1/L^2]
    constraint_scaling[cs_distance_idx] = 1.0 / (major_radius ** 2)  # [L^2] -> weight [1/L^2]
    
    if coil_objective_terms:
        for term_name, term_value in coil_objective_terms.items():
            # Skip _p parameters (already handled above)
            if term_name.endswith("_p"):
                continue
            if term_name in term_map:
                term_config = term_map[term_name]
                obj = term_config["obj"]
                thresh = term_config["threshold"]
                
                if term_value in term_config:
                    constraint = term_config[term_value](obj, thresh)
                    constraint_idx = len(c_list)  # Index before appending
                    c_list.append(constraint)
                    
                    # Scaling factors to make weight * constraint dimensionless
                    # 
                    # Base constraint units (from simsopt):
                    # - Length/distance: [L] (m)
                    # - Curvature: [1/L] for l1/l2, [1/L^(p-1)] for lp (LpCurveCurvature)
                    # - Mean squared curvature: [1/L^2]
                    # - Arclength variation: [L^2]
                    # - Force: [F^p / L^(p-1)] where F is force per unit length [F/L] = [N/m]
                    # - Torque: [T^p / L^(p-1)] where T is torque per unit length [T/L] = [N]
                    # 
                    # Penalty type affects final units:
                    # - l1/l1_threshold: same as base constraint
                    # - l2/l2_threshold: base units squared
                    # - lp/lp_threshold: depends on constraint type (see below)
                    
                    # Get p value for lp penalties
                    p_value = 2  # Default p value
                    if term_value in ["lp", "lp_threshold"]:
                        p_key = f"{term_name}_p"
                        p_value = coil_objective_terms.get(p_key, 2)
                    
                    # Base scaling for l1/l1_threshold (linear penalties)
                    # Weight scaling = 1 / (constraint units) to make weight * constraint dimensionless
                    base_scaling = 1.0
                    if term_name == "total_length":
                        base_scaling = 1.0 / major_radius  # [L] -> weight needs [1/L]
                    elif term_name == "coil_coil_distance":
                        # CurveCurveDistance already computes squared penalty, so units are [L^2]
                        base_scaling = 1.0 / (major_radius ** 2)  # [L^2] -> weight needs [1/L^2]
                    elif term_name == "coil_surface_distance":
                        # CurveSurfaceDistance already computes squared penalty, so units are [L^2]
                        base_scaling = 1.0 / (major_radius ** 2)  # [L^2] -> weight needs [1/L^2]
                    elif term_name == "coil_curvature":
                        base_scaling = major_radius  # [1/L] -> weight needs [L]
                    elif term_name == "coil_mean_squared_curvature":
                        base_scaling = major_radius ** 2  # [1/L^2] -> weight needs [L^2]
                    elif term_name == "coil_arclength_variation":
                        base_scaling = 1.0 / (major_radius ** 2)  # [L^2] -> weight needs [1/L^2]
                    elif term_name == "linking_number":
                        base_scaling = 1.0  # Already dimensionless
                    elif term_name in ["coil_coil_force", "coil_coil_torque"]:
                        base_scaling = 1.0  # Handled in lp section (always uses lp/lp_threshold)
                    
                    # Adjust scaling for penalty type
                    if term_value in ["l2", "l2_threshold"]:
                        # Squared penalty: constraint units squared, so weight scaling squared
                        if term_name == "total_length":
                            constraint_scaling[constraint_idx] = base_scaling / major_radius  # [L^2] -> weight [1/L^2]
                        # coil_coil_distance and coil_surface_distance already have squared units, handled above
                        elif term_name == "coil_curvature":
                            constraint_scaling[constraint_idx] = base_scaling * major_radius  # [1/L^2] -> weight [L^2]
                        elif term_name == "coil_mean_squared_curvature":
                            constraint_scaling[constraint_idx] = base_scaling * (major_radius ** 2)  # [1/L^4] -> weight [L^4]
                        elif term_name == "coil_arclength_variation":
                            constraint_scaling[constraint_idx] = base_scaling / (major_radius ** 2)  # [L^4] -> weight [1/L^4]
                        else:
                            constraint_scaling[constraint_idx] = base_scaling
                    elif term_value in ["lp", "lp_threshold"]:
                        # Lp penalty: units depend on constraint type
                        if term_name == "coil_curvature":
                            # LpCurveCurvature: (1/p) ∫ max(κ - κ₀, 0)^p dl has units [1/L^(p-1)]
                            # Weight needs [L^(p-1)]: weight *= major_radius^(p-1)
                            constraint_scaling[constraint_idx] = major_radius ** (p_value - 1)
                        elif term_name in ["coil_coil_force", "coil_coil_torque"]:
                            # LpCurveForce/LpCurveTorque: (1/p) ∫ max(|F| - F₀, 0)^p dℓ
                            # F is force per unit length [F/L] = [N/m], so constraint has units [F^p / L^(p-1)]
                            # Force scales with current^2: F ∝ I^2, so F^p ∝ I^(2p)
                            # Weight needs [L^(p-1) / F^p] = [L^(p-1) / I^(2p)] to make weight * constraint dimensionless
                            # 
                            # To get units [L^(p-1)] (since weight * constraint must be dimensionless):
                            #   weight *= major_radius^(p-1) / total_current^(2p)
                            # This scales the weight inversely with current^(2p) to account for force scaling as I^2
                            constraint_scaling[constraint_idx] = (major_radius ** (p_value - 1)) / (total_current ** (2 * p_value))
                        elif term_name in ["total_length", "coil_coil_distance", "coil_surface_distance"]:
                            constraint_scaling[constraint_idx] = base_scaling / (major_radius ** (p_value - 1))  # [L^p] -> weight [1/L^p]
                        elif term_name == "coil_mean_squared_curvature":
                            constraint_scaling[constraint_idx] = base_scaling * (major_radius ** (2 * p_value - 2))  # [1/L^(2p)] -> weight [L^(2p)]
                        elif term_name == "coil_arclength_variation":
                            constraint_scaling[constraint_idx] = base_scaling / (major_radius ** (2 * p_value - 2))  # [L^(2p)] -> weight [1/L^(2p)]
                        else:
                            constraint_scaling[constraint_idx] = base_scaling
                    elif term_value == "":
                        # Empty string: for coil_coil_distance and coil_surface_distance
                        # These already compute squared penalties internally, so units are [L^2]
                        # Scaling already set correctly above (base_scaling = 1.0 / (major_radius ** 2))
                        constraint_scaling[constraint_idx] = base_scaling
                    else:
                        # For l1/l1_threshold (linear penalties), use base scaling
                        constraint_scaling[constraint_idx] = base_scaling
                    
                    # Track constraint name and threshold for printing
                    name_map = {
                        "total_length": ("Length", length_threshold),
                        "coil_mean_squared_curvature": ("MSC", msc_threshold),
                        "coil_arclength_variation": ("Arclength Var", arclength_variation_threshold),
                        "coil_curvature": ("κ", curvature_threshold),
                        "linking_number": ("Link #", None),
                        "coil_coil_force": ("Force", force_threshold),
                        "coil_coil_torque": ("Torque", torque_threshold),
                    }
                    if term_name in name_map:
                        constraint_names_and_thresholds.append(name_map[term_name])
                    
                    # Track constraint index to term name mapping for named weights
                    constraint_idx_to_term[constraint_idx] = term_name
                else:
                    print(f"Warning: Unknown option '{term_value}' for {term_name}, skipping")
    
    # Record objective setup time
    objective_setup_time = time.perf_counter() - objective_setup_start
    from .post_processing import _timing_results
    _timing_results["objective_setup"] = objective_setup_time
    
    # Step 5: Run optimization
    optimization_start = time.perf_counter()
    start_time = time.time()
    lag_mul = None  # Initialize lag_mul for scipy methods
    iterations_used = 0  # Track total iterations for CI reporting
    opt_result = None  # Scipy/minimize result for metadata (auglag does not provide this)
    
    # Check if weight is specified for coil-surface distance and coil-coil distance constraints
    cs_weight_specified = False
    cc_weight_specified = False
    if cs_distance_index is not None:
        cs_weight_key = f'constraint_weight_{cs_distance_index}'
        cs_weight_specified = cs_weight_key in kwargs
    if cc_distance_index is not None:
        cc_weight_key = f'constraint_weight_{cc_distance_index}'
        cc_weight_specified = cc_weight_key in kwargs
    
    if algorithm == "augmented_lagrangian":
        # Apply weight to coil-surface distance and coil-coil distance for augmented_lagrangian
        # Use specified weight or default to 1e3, then apply scaling
        if cs_distance_index is not None:
            # If weight is specified, use it; otherwise default to 1e3
            if cs_weight_specified:
                cs_weight = kwargs[f'constraint_weight_{cs_distance_index}']
            else:
                cs_weight = kwargs.get(f'constraint_weight_{cs_distance_index}', 1e3)
            # Apply scaling to make weight dimensionless (always apply scaling for distance objectives)
            if cs_distance_index in constraint_scaling:
                cs_weight *= constraint_scaling[cs_distance_index]
            c_list[cs_distance_index] = Weight(cs_weight) * c_list[cs_distance_index]
        if cc_distance_index is not None:
            # If weight is specified, use it; otherwise default to 1e3
            if cc_weight_specified:
                cc_weight = kwargs[f'constraint_weight_{cc_distance_index}']
            else:
                cc_weight = kwargs.get(f'constraint_weight_{cc_distance_index}', 1e3)
            # Apply scaling to make weight dimensionless (always apply scaling for distance objectives)
            if cc_distance_index in constraint_scaling:
                cc_weight *= constraint_scaling[cc_distance_index]
            c_list[cc_distance_index] = Weight(cc_weight) * c_list[cc_distance_index]
        
        # auglag_coils: simsopt.solve.augmented_lagrangian; some versions export via solve.__init__
        try:
            from simsopt.solve import augmented_lagrangian_method
        except ImportError:
            from simsopt.solve.augmented_lagrangian import augmented_lagrangian_method
        import inspect
        _alm_sig = inspect.signature(augmented_lagrangian_method)
        _alm_params = set(_alm_sig.parameters.keys())
        augmented_lagrangian_options = {
            "MAXITER": max_iterations,
            "MAXITER_lag": max_iter_subopt,
            "verbose": verbose,
        }
        if "mu_init" in kwargs.keys():
            augmented_lagrangian_options["mu_init"] = kwargs["mu_init"]
        if "tau" in kwargs.keys():
            augmented_lagrangian_options["tau"] = kwargs["tau"]
        if "minimize_method" in kwargs.keys():
            augmented_lagrangian_options["minimize_method"] = kwargs["minimize_method"]
        # Filter to only params the function accepts
        augmented_lagrangian_options = {k: v for k, v in augmented_lagrangian_options.items() if k in _alm_params}
        _, _, lag_mul = augmented_lagrangian_method(
            f=None,
            equality_constraints=c_list,
            **augmented_lagrangian_options,
        )
        # augmented_lagrangian_method doesn't return nit; estimate from settings
        iterations_used = max_iterations
    elif algorithm in ['BFGS', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead', 'Powell', 'CG', 'Newton-CG', 'TNC', 'COBYLA', 'trust-constr']:
        # Build weighted objective function from constraints
        # c_list includes flux first, then other constraints
        # Default weight is 1.0 for all constraints
        weights = []
        
        # Mapping from term names to their weight parameter names in coil_objective_terms
        term_to_weight_key = {
            "total_length": "length_weight",
            "coil_coil_distance": "cc_weight",
            "coil_surface_distance": "cs_weight",
            "coil_curvature": "curvature_weight",
            "coil_arclength_variation": "arclength_variation_weight",
            "coil_mean_squared_curvature": "msc_weight",
            "coil_coil_force": "force_weight",
            "coil_coil_torque": "torque_weight",
            "linking_number": "linking_weight",
        }
        
        for i, constraint in enumerate(c_list):
            # Map constraint index to weight name (for backward compatibility)
            # Flux (index 0) always has weight 1.0 (dimensionless)
            if i == 0:
                # Check for flux_weight in coil_objective_terms
                if coil_objective_terms and "flux_weight" in coil_objective_terms:
                    weights.append(float(coil_objective_terms["flux_weight"]))
                else:
                    weights.append(1.0)  # Flux weight default
            else:
                # For other constraints, try to get specific weight or default to 1.0
                weight_key = f'constraint_weight_{i}'
                weight_specified = weight_key in kwargs
                weight = kwargs.get(weight_key, 1.0)
                
                # Check for named weight in coil_objective_terms (takes precedence)
                term_name = constraint_idx_to_term.get(i)
                if term_name and coil_objective_terms:
                    weight_param = term_to_weight_key.get(term_name)
                    if weight_param and weight_param in coil_objective_terms:
                        weight = float(coil_objective_terms[weight_param])  # Convert to float (handles string "1e3")
                        weight_specified = True
                
                # Apply weight to coil-surface distance and coil-coil distance constraints
                # Use specified weight or default to 1e3 for distance constraints
                if cs_distance_index is not None and i == cs_distance_index:
                    # Check named weight first
                    if coil_objective_terms and "cs_weight" in coil_objective_terms:
                        weight = float(coil_objective_terms["cs_weight"])
                        weight_specified = True
                    elif cs_weight_specified:
                        weight = kwargs[f'constraint_weight_{i}']
                    else:
                        weight = kwargs.get(f'constraint_weight_{i}', 1e3)
                elif cc_distance_index is not None and i == cc_distance_index:
                    # Check named weight first
                    if coil_objective_terms and "cc_weight" in coil_objective_terms:
                        weight = float(coil_objective_terms["cc_weight"])
                        weight_specified = True
                    elif cc_weight_specified:
                        weight = kwargs[f'constraint_weight_{i}']
                    else:
                        weight = kwargs.get(f'constraint_weight_{i}', 1e3)
                
                # Rescale weight to be dimensionless
                # Always apply scaling for distance objectives (they have squared units)
                # For other constraints, only apply if weight not explicitly specified
                if i in constraint_scaling:
                    if i in [cc_distance_idx, cs_distance_idx]:
                        # Always apply scaling for distance objectives
                        weight *= constraint_scaling[i]
                    elif not weight_specified:
                        # For other constraints, only if weight not explicitly specified
                        weight *= constraint_scaling[i]
                
                weights.append(weight)
        
        # Create weighted sum of constraints
        JF = sum([Weight(w) * c for c, w in zip(c_list, weights)])
        
        # Track iteration number for objective function
        iteration_count = [0]  # Use list to allow modification in nested function

        # Define the objective function and gradient
        def objective(x: np.ndarray) -> float:
            JF.x = x  # type: ignore[attr-defined]
            J = JF.J()  # type: ignore[attr-defined]
            iteration_count[0] += 1
            if verbose and (iteration_count[0] == 1 or iteration_count[0] % 100 == 0):
                grad = JF.dJ()  # type: ignore[attr-defined]
                outstr = f"[{iteration_count[0]}]"
                outstr += f" L={sum(J.J() for J in Jls):.2f}"
                outstr += f", d_cc={Jccdist.shortest_distance():.2f}, d_cs={Jcsdist.shortest_distance():.2f}"
                kappa_values = [c.kappa().max() for c in base_curves]
                msc_values = [MeanSquaredCurvature(c).J() for c in base_curves]
                kappa_str = ",".join([f"{k:.1f}" for k in kappa_values])
                msc_str = ",".join([f"{m:.1f}" for m in msc_values])
                outstr += f", κ=[{kappa_str}]"  # type: ignore[attr-defined]
                outstr += f", MSC=[{msc_str}]"
                outstr += f", LN={int(round(Jlink.J()))}"
                outstr += f", F={Jforce.J():.2e}"
                outstr += f", τ={Jtorque.J():.2e}"
                outstr += f", ‖∇J‖={np.linalg.norm(grad):.1e}"
                print(outstr)
                
                # Print weighted contributions of each objective term
                contrib_parts = []
                name_short = {"Flux": "J_f", "CC Distance": "d_cc", "CS Distance": "d_cs",
                              "Length": "L", "MSC": "MSC", "Arclength Var": "Var",
                              "κ": "κ", "Link #": "LN", "Force": "F", "Torque": "τ"}
                # Flux contribution (index 0)
                flux_contrib = weights[0] * c_list[0].J()
                contrib_parts.append(f"{name_short.get('Flux', 'Flux')}={flux_contrib:.1e}")
                # Other constraint contributions
                for idx, (name, _) in enumerate(constraint_names_and_thresholds, start=1):
                    if idx < len(c_list) and idx < len(weights):
                        constraint_contrib = weights[idx] * c_list[idx].J()
                        short = name_short.get(name, name)
                        contrib_parts.append(f"{short}={constraint_contrib:.1e}")
                contrib_str = "Objs: " + ", ".join(contrib_parts)
                contrib_str += f", Total={J:.1e}"
                print(contrib_str)
            return J
        
        def gradient(x: np.ndarray) -> np.ndarray:
            JF.x = x  # type: ignore[attr-defined]
            return JF.dJ()  # type: ignore[attr-defined]
        
        # Taylor test to verify gradient computation
        # Check that f(x + εh) ≈ f(x) + ε * ∇f(x) · h for small ε
        # The error should decrease by at least a factor of 0.6 as ε decreases
        x0 = JF.x.copy()  # type: ignore[attr-defined]
        J0 = objective(x0)
        grad0 = gradient(x0)
        
        # Generate random direction h (normalized)
        np.random.seed(42)  # For reproducibility
        h = np.random.randn(len(x0))
        h = h / np.linalg.norm(h)
        
        # Test with small perturbation (decreasing epsilon)
        epsilons = [1e-6, 1e-7, 1e-8]
        errors = []
        for eps in epsilons:
            x_perturbed = x0 + eps * h
            J_perturbed = objective(x_perturbed)
            J_predicted = J0 + eps * np.dot(grad0, h)
            error = abs(J_perturbed - J_predicted) / (abs(J0) + 1e-12)
            errors.append(error)
            
            # if verbose:
            #     print(f"Taylor test ε={eps:.1e}: error={error:.2e}")
        
        # Check that error decreases by at least a factor of 0.6 as epsilon decreases
        # (epsilon decreases by factor of 10, so error should decrease by at least 0.6)
        taylor_test_passed = True
        for i in range(len(errors) - 1):
            if errors[i] > 0:
                error_ratio = errors[i + 1] / errors[i]
                # Error should decrease, so ratio should be < 1.0
                # We require it to decrease by at least factor of 0.6
                if error_ratio > 0.6:
                    print(f"WARNING: Taylor test failed: error ratio {error_ratio:.3f} > 0.6 "
                          f"(ε={epsilons[i]:.1e} -> {epsilons[i+1]:.1e}, "
                          f"error={errors[i]:.2e} -> {errors[i+1]:.2e})", file=sys.stderr)
                    taylor_test_passed = False
        
        if not taylor_test_passed:
            print("Gradient computation may be incorrect!", file=sys.stderr)
        elif verbose:
            print("Taylor test passed: error decreases as expected")
        
        # Restore original state
        JF.x = x0  # type: ignore[attr-defined, assignment]
        
        # Build options dictionary, starting with defaults
        options = {'maxiter': max_iterations}
        # Set algorithm-specific tolerance defaults
        if algorithm == 'L-BFGS-B':
            # L-BFGS-B uses ftol and gtol, not tol
            # Defaults: ftol=2.220446049250313e-09, gtol=1e-05
            # Use scipy defaults to avoid premature convergence
            # Note: Very strict tolerances (like 1e-12) can cause early convergence
            # if the gradient norm drops below gtol quickly
            options.setdefault('ftol', 1e-12)  # scipy default
            options.setdefault('gtol', 1e-12)  # scipy default
        elif algorithm == 'TNC':
            options.setdefault('ftol', 1e-6)  # Reasonable default for TNC
            options.setdefault('gtol', 1e-05)  # scipy default
        elif algorithm in ['COBYLA']:
            options.setdefault('tol', 1e-12)  # COBYLA uses tol
        if algorithm in ['L-BFGS-B', 'TNC']:
            if 'maxfun' not in options:
                options['maxfun'] = max_iterations * 15000
            if 'max_iter_subopt' in options:
                options['maxfun'] = max_iter_subopt * 15000
            # If user explicitly set maxfun in algorithm_options, it will override via options.update() below
        
        # Add user-specified algorithm-specific options
        # Validate them first to catch errors early
        if algorithm_options:
            _validate_algorithm_options(algorithm, algorithm_options)
            # Merge user options, allowing them to override defaults
            options.update(algorithm_options)
        
        result = minimize(
            fun=objective,
            x0=JF.x,  # type: ignore[attr-defined]
            method=algorithm,
            jac=gradient,
            options=options,
        )
        
        # Record iterations and metadata from scipy result
        iterations_used = getattr(result, 'nit', 0)
        opt_result = result
    
    end_time = time.time()
    optimization_time = time.perf_counter() - optimization_start
    _timing_results["coil_optimization"] = optimization_time
    
    # Start timing for save and metrics section
    save_metrics_start = time.perf_counter()
    
    # Calculate final total current (sum of unique base coils)
    total_current_final = sum([c.current.get_value() for c in coils[:ncoils]])
    
    # Save optimized coils
    try:
        coils_to_vtk(coils, out_dir / "coils_optimized")
    except Exception as e:
        print(f"Warning: Failed to save optimized coils to VTK: {e}")
        print("  Continuing without VTK export...")
    bs.save(out_dir / "biot_savart_optimized.json")
    
    # Calculate final B-field (suppress simsopt Bmag output)
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    with suppress_output():
        B_final = calculate_modB_on_major_radius(bs, s_plot)
    
    # Save final surface data
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {
        "B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                     s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                         s_plot.unitnormal(), axis=2)[:, :, None] /
                    bs.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs.AbsB().reshape((qphi, qtheta, 1))
    }
    s_plot.to_vtk(out_dir / "surface_optimized", extra_data=pointData)
    
    # Calculate final forces
    # Try new coil.force() and coil.torque() API (pedro_simsopt), fall back to old API
    if hasattr(coils[0], 'force') and hasattr(coils[0], 'torque'):
        # New API: coil.force(coils) and coil.torque(coils)
        max_force = [np.max(np.linalg.norm(c.force(coils), axis=1)) for c in coils[:ncoils]]
        max_torque = [np.max(np.linalg.norm(c.torque(coils), axis=1)) for c in coils[:ncoils]]
    else:
        # Old API: coil_force(coil, coils) and coil_torque(coil, coils)
        try:
            from simsopt.field.force import coil_force, coil_torque
            max_force = [np.max(np.linalg.norm(coil_force(c, coils), axis=1)) for c in coils[:ncoils]]
            max_torque = [np.max(np.linalg.norm(coil_torque(c, coils), axis=1)) for c in coils[:ncoils]]
        except ImportError:
            # Neither API available, use zeros as placeholder
            max_force = [0.0] * ncoils
            max_torque = [0.0] * ncoils
    # Calculate final B_N metrics
    # If virtual casing is used, we need to subtract B_external_normal from the coil B_N
    vc_target = kwargs.get('vc_target', None)
    
    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    bs.set_points(s.gamma().reshape((-1, 3)))
    B_field = bs.B().reshape((nphi, ntheta, 3))
    unit_normal = s.unitnormal().reshape((nphi, ntheta, 3))
    BdotN_coils = np.sum(B_field * unit_normal, axis=2)  # B_N from coils
    
    if vc_target is not None:
        # B_N error = |B_N_coils - B_external_normal|
        # vc_target is B_external_normal from virtual casing
        absBn = np.abs(BdotN_coils - vc_target)
    else:
        # Standard case: B_N error = |B_N_coils| (target is zero)
        absBn = np.abs(BdotN_coils)
    
    abs_B = bs.AbsB().reshape((nphi, ntheta))
    avg_BdotN_over_B = np.mean(absBn) / np.mean(abs_B) if np.mean(abs_B) > 0 else 0.0
    
    # For max calculation, use the same surface (with vc_target if available)
    # Avoid division by very small numbers
    abs_B_safe = np.where(abs_B > 1e-10, abs_B, 1e-10)
    max_BdotN_overB = np.max(absBn / abs_B_safe) if np.any(abs_B > 0) else 0.0

    # Check coil-surface interlinking: each base coil must encircle the
    # plasma by having points both inside the torus hole and outside the
    # plasma.  We compare each coil against the *local* surface
    # cross-section at the coil's toroidal angle, not against the global
    # R_min/R_max of the entire surface.  The global check fails on
    # strongly-shaped stellarators (e.g. HSX) where the surface
    # cross-section varies substantially with toroidal angle.
    surface_gamma = s.gamma()
    R_surface = np.sqrt(surface_gamma[:, :, 0]**2 + surface_gamma[:, :, 1]**2)

    # Per-phi-slice R_min and R_max  (axis 1 = theta)
    R_min_per_phi = np.min(R_surface, axis=1)   # (nphi,)
    R_max_per_phi = np.max(R_surface, axis=1)   # (nphi,)

    # Toroidal angle of each phi slice (use first theta point)
    phi_surface_slices = np.arctan2(
        surface_gamma[:, 0, 1], surface_gamma[:, 0, 0]
    )  # (nphi,)

    coils_linked_to_surface = True
    for c in base_curves:
        gamma = c.gamma()
        R_coil = np.sqrt(gamma[:, 0]**2 + gamma[:, 1]**2)
        phi_coil = np.arctan2(gamma[:, 1], gamma[:, 0])  # (npts,)

        # For each coil point find the nearest surface phi slice
        dphi = phi_coil[:, None] - phi_surface_slices[None, :]
        dphi = np.abs(np.arctan2(np.sin(dphi), np.cos(dphi)))
        nearest_phi_idx = np.argmin(dphi, axis=1)  # (npts,)

        local_R_min = R_min_per_phi[nearest_phi_idx]  # (npts,)
        local_R_max = R_max_per_phi[nearest_phi_idx]  # (npts,)

        has_inside = np.any(R_coil < local_R_min)
        has_outside = np.any(R_coil > local_R_max)
        if not (has_inside and has_outside):
            coils_linked_to_surface = False
            break
    
    # Generate 3D visualization plot
    try:
        # Get vc_target_plot from kwargs if provided (for virtual casing cases)
        vc_target_plot = kwargs.get('vc_target_plot', None)
        _plot_bn_error_3d(
            s_plot,
            bs,
            coils,
            out_dir,
            filename="bn_error_3d_plot.pdf",
            title="B_N/|B| Error on Plasma Surface with Optimized Coils",
            vc_target=vc_target_plot,
        )
    except Exception as e:
        print(f"Warning: Failed to generate 3D plot: {e}")
    
    # Record save and metrics time
    save_metrics_time = time.perf_counter() - save_metrics_start
    _timing_results["save_and_metrics"] = save_metrics_time
    
    # Run post-processing: QFM surface, Poincaré plots, iota profiles, quasisymmetry profiles
    # Skip if this is part of Fourier continuation (will run once at the end)
    # Initialize post_processing_results to empty dict
    post_processing_results = {}
    
    if not skip_post_processing:
        try:
            from .post_processing import run_post_processing
            
            # Determine case.yaml path for post-processing
            # case_path should already be resolved to absolute path by optimize_coils
            case_yaml_path = None
            if case_path is not None:
                case_path_obj = Path(case_path) if isinstance(case_path, str) else case_path
                # If it's already absolute and exists, use it directly
                if case_path_obj.is_absolute() and case_path_obj.exists():
                    if case_path_obj.is_file():
                        case_yaml_path = case_path_obj
                    elif case_path_obj.is_dir():
                        case_yaml_path = case_path_obj / "case.yaml"
                        if not case_yaml_path.exists():
                            case_yaml_path = None
                elif case_path_obj.exists():
                    # Resolve relative path
                    case_yaml_path = case_path_obj.resolve()
                    if case_yaml_path.is_dir():
                        case_yaml_path = case_yaml_path / "case.yaml"
                        if not case_yaml_path.exists():
                            case_yaml_path = None
            
            # Check if case.yaml is in out_dir (from submit-case)
            if case_yaml_path is None or not case_yaml_path.exists():
                case_yaml_path = out_dir / "case.yaml"
            if not case_yaml_path.exists():
                # Try parent directory (for Fourier continuation subdirectories)
                case_yaml_path = out_dir.parent / "case.yaml"
            
            # Also try searching relative to surface file and in cases directory
            if not case_yaml_path.exists() and hasattr(s, 'filename') and s.filename:
                # Try to find case.yaml relative to the surface file
                surface_dir = Path(s.filename).parent
                surface_stem = Path(s.filename).stem.replace("input.", "").replace(".focus", "")
                potential_case_paths = [
                    surface_dir / "case.yaml",
                    surface_dir.parent / "case.yaml",
                    Path("cases") / surface_stem / "case.yaml",
                ]
                for path in potential_case_paths:
                    if path.exists():
                        case_yaml_path = path
                        break
            
            # If still not found, search cases directory for YAML files that reference this surface
            # First try to find cases directory relative to repo root (go up from out_dir)
            if case_yaml_path is None or not case_yaml_path.exists():
                cases_dir = None
                current_dir = Path(out_dir)
                for _ in range(10):  # Search up to 10 levels
                    potential_cases_dir = current_dir / "cases"
                    if potential_cases_dir.exists() and potential_cases_dir.is_dir():
                        cases_dir = potential_cases_dir
                        break
                    if current_dir.parent == current_dir:  # Reached root
                        break
                    current_dir = current_dir.parent
                
                # Also try relative to current working directory
                if cases_dir is None:
                    cases_dir = Path("cases")
                
                if cases_dir.exists():
                    import yaml as yaml_module
                    surface_filename = Path(s.filename).name if hasattr(s, 'filename') and s.filename else ""
                    for yaml_file in cases_dir.glob("*.yaml"):
                        try:
                            case_data = yaml_module.safe_load(yaml_file.read_text())
                            if case_data and isinstance(case_data, dict):
                                surface_in_case = case_data.get("surface_params", {}).get("surface", "")
                                # Check if this case references the same surface file
                                if surface_filename and surface_filename in surface_in_case:
                                    case_yaml_path = yaml_file.resolve()
                                    break
                                elif surface_in_case in surface_filename:
                                    case_yaml_path = yaml_file.resolve()
                                    break
                        except Exception:
                            continue
            
            # Coils JSON path - check both biot_savart_optimized.json and coils.json
            coils_json_path = out_dir / "biot_savart_optimized.json"
            if not coils_json_path.exists():
                coils_json_path = out_dir / "coils.json"
            
            if coils_json_path.exists():
                print("\nRunning post-processing (QFM, Poincaré plots, profiles)...")
                
                # Determine helicity_n based on surface type (QA=0, QH=-1)
                # Default to QA (helicity_n=0)
                helicity_n = 0
                if case_yaml_path.exists():
                    import yaml
                    try:
                        case_data = yaml.safe_load(case_yaml_path.read_text())
                        surface_name = case_data.get("surface_params", {}).get("surface", "").lower()
                        # Check for QH surfaces
                        if "qh" in surface_name or "qash" in surface_name:
                            helicity_n = -1
                    except Exception:
                        pass  # Use default
                
                # Determine plasma_surfaces_dir - go up from output directory to find repo root
                plasma_surfaces_dir = None
                current_dir = Path(out_dir)
                for _ in range(5):  # Search up to 5 levels
                    potential_plasma_dir = current_dir / "plasma_surfaces"
                    if potential_plasma_dir.exists():
                        plasma_surfaces_dir = potential_plasma_dir
                        break
                    if current_dir.parent == current_dir:  # Reached root
                        break
                    current_dir = current_dir.parent
                
                post_processing_results = run_post_processing(
                    coils_json_path=coils_json_path,
                    output_dir=out_dir,
                    case_yaml_path=case_yaml_path if case_yaml_path.exists() else None,
                    plasma_surfaces_dir=plasma_surfaces_dir,  # Pass repo root plasma_surfaces directory
                    run_vmec=run_vmec,
                    helicity_m=1,
                    helicity_n=helicity_n,
                    ns=50,
                    plot_boozer=plot_boozer,
                    plot_poincare=plot_poincare,
                    nfieldlines=20,
                    run_simple=run_simple,
                    plot_finite_build=kwargs.get('plot_finite_build', False),
                    finite_build_width=kwargs.get('finite_build_width'),
                    finite_build_height=kwargs.get('finite_build_height'),
                )
                print("Post-processing complete!")
                if 'quasisymmetry_average' in post_processing_results:
                    print(f"  Average quasisymmetry error: {post_processing_results['quasisymmetry_average']:.2e}")
            else:
                print(f"Warning: Skipping post-processing (coils_json not found: {coils_json_path})")
        except Exception as e:
            print(f"Warning: Post-processing failed: {e}")
            import traceback
            traceback.print_exc()
            post_processing_results = {}  # Initialize empty dict if post-processing failed
    
    # Note: Individual file zipping is disabled - the entire submission directory
    # will be zipped by submit-case command after all files are written
    
    # Cache thresholds for continuation steps (remove internal cache key before returning)
    cached_thresholds = {
        'length_threshold': length_threshold,
        'flux_threshold': flux_threshold,
        'cc_threshold': cc_threshold,
        'cs_threshold': cs_threshold,
        'msc_threshold': msc_threshold,
        'arclength_variation_threshold': arclength_variation_threshold,
        'curvature_threshold': curvature_threshold,
        'force_threshold': force_threshold,
        'torque_threshold': torque_threshold,
        'coil_width': coil_width,
        'a0': a0,                        # Minor-radius scale factor: ARIES_CS_MINOR_RADIUS / minor_radius
        'major_radius': major_radius,    # actual device major radius [m]
        'minor_radius': minor_radius,    # actual device minor radius [m]
    }
    
    # Prepare results dictionary
    bs.set_points(s.gamma().reshape((-1, 3)))
    results = {
        'initial_B_field': B_initial,
        'final_B_field': B_final,
        'target_B_field': target_B,
        'optimization_time': end_time - start_time,
        'walltime_sec': end_time - start_time,
        'iterations_used': iterations_used,
        'final_squared_flux': Jf.J(),
        'optimization_success': opt_result.success if opt_result is not None and hasattr(opt_result, 'success') else None,
        'optimization_message': str(opt_result.message) if opt_result is not None and hasattr(opt_result, 'message') else None,
        'optimization_nfev': getattr(opt_result, 'nfev', None) if opt_result is not None else None,
        'optimization_njev': getattr(opt_result, 'njev', None) if opt_result is not None else None,
        '_cached_thresholds': cached_thresholds,  # Store for continuation steps
        'final_min_cs_separation': Jcsdist.shortest_distance(),
        'final_min_cc_separation': Jccdist.shortest_distance(),
        'final_length_per_coil': [float(CurveLength(c).J()) for c in base_curves],
        'final_current_per_coil': [float(abs(coils[i].current.get_value())) for i in range(ncoils)],
        'total_current_before': float(total_current),
        'total_current_after': float(total_current_final),
        'final_total_length': sum(CurveLength(c).J() for c in base_curves),
        'final_max_curvature': max(np.max(c.kappa()) for c in base_curves),
        'final_average_curvature': np.mean([c.kappa() for c in base_curves]),
        'final_arclength_variation': np.mean([ArclengthVariation(c).J() for c in base_curves]),
        'final_mean_squared_curvature': np.max([np.mean(c.kappa() ** 2) for c in base_curves]),
        'final_linking_number': Jlink.J(),
        'coils_linked_to_surface': coils_linked_to_surface,
        'final_max_max_coil_force': np.max(max_force),
        'final_avg_max_coil_force': np.mean(max_force),
        'final_max_force_per_coil': [float(f) for f in max_force],
        'final_max_torque_per_coil': [float(t) for t in max_torque],
        'final_max_max_coil_torque': np.max(max_torque),
        'final_avg_max_coil_torque': np.mean(max_torque),
        'avg_BdotN_over_B': avg_BdotN_over_B,
        'max_BdotN_over_B': max_BdotN_overB,
        'lagrange_multipliers': lag_mul,
        'output_directory': str(out_dir),
        'flux_threshold': flux_threshold,
        'cc_threshold': cc_threshold,
        'cs_threshold': cs_threshold,
        'msc_threshold': msc_threshold,
        'arclength_variation_threshold': arclength_variation_threshold,
        'curvature_threshold': curvature_threshold,
        'force_threshold': force_threshold,
        'torque_threshold': torque_threshold,
    }
    
    # Merge post-processing results (quasisymmetry_average, loss_fraction, etc.) into results
    if post_processing_results:
        # Only include numeric/metric values, not objects like 'vmec' or 'qfm_surface'
        for key, value in post_processing_results.items():
            if key in ['quasisymmetry_average', 'loss_fraction', 'BdotN', 'BdotN_over_B']:
                if isinstance(value, (int, float)):
                    results[key] = float(value)
    
    # Add timing results to output (printed in OPTIMIZATION RESULTS SUMMARY)
    results['timing'] = get_timing_results()
    
    return coils, results