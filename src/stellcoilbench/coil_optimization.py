from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import numpy as np
from typing import Callable
from datetime import datetime
import zipfile
from .config_scheme import CaseConfig

from simsopt.geo import SurfaceRZFourier
try:
    from simsopt.field import regularization_circ
except ImportError:  # pragma: no cover - fallback for older simsopt
    regularization_circ = None

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
    cm = None  # type: ignore
    Normalize = None  # type: ignore


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
        if isinstance(other, LinearPenalty):
            # Create a combined objective
            combined = self.objective + other.objective
            # Use the first threshold (they should be the same)
            return LinearPenalty(combined, self.threshold)
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
            return LinearPenalty(weighted_obj, scaled_threshold)
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
    from simsopt import save
    from .evaluate import load_case_config
    
    if case_cfg is None:
        case_cfg = load_case_config(case_path)
    
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
    else:
        raise ValueError(f"Unknown surface file: {surface_file}")
    coil_params['target_B'] = target_B

    try:
        os.chdir(output_dir)
        
        # Extract algorithm_options from optimizer_params if present
        # This allows users to specify algorithm-specific hyperparameters
        algorithm_options = optimizer_params.pop('algorithm_options', {})
        
        # Check if Fourier continuation is enabled
        fourier_continuation = case_cfg.fourier_continuation
        if fourier_continuation and fourier_continuation.get('enabled', False):
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
                case_path=case_path,  # Pass case_path for post-processing
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
                    **threshold_kwargs
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
    major_radius = s.get_rc(0, 0)
    minor_radius_component = abs(s.get_rc(1, 0))
    
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
    print(f"\nAdaptive coil positioning loop (max {max_adaptive_iterations} iterations):")
    print(f"  Initial: R0={R0:.3f} m ({R0_scale:.3f}x), R1={R1:.3f} m ({R1_scale:.3f}x)")
    print(f"  Limits: R0_scale <= {max_R0_scale}, R1_scale <= {max_R1_scale}")
    
    # Track previous values to detect oscillation
    prev_R0_scale = None
    prev_R1_scale = None
    oscillation_count = 0
    
    for adaptive_iter in range(max_adaptive_iterations):
        print(f"\n  Iteration {adaptive_iter + 1}/{max_adaptive_iterations}:")
        print(f"    R0: {R0:.3f} m (scale: {R0_scale:.3f}x)")
        print(f"    R1: {R1:.3f} m (scale: {R1_scale:.3f}x)")
        
        # Check for oscillation (values repeating)
        if prev_R0_scale is not None and prev_R1_scale is not None:
            if abs(R0_scale - prev_R0_scale) < 0.01 and abs(R1_scale - prev_R1_scale) < 0.01:
                oscillation_count += 1
                if oscillation_count >= 3:
                    print("    Warning: Detected oscillation (values repeating). Stopping adaptive loop.")
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
        
        print(f"    Constraints: cs_ok={cs_ok} (min={min_cs_sep:.4f} m, required={min_cs_distance*(1-adaptive_tolerance):.4f} m), "
              f"cc_ok={cc_ok} (min={min_cc_sep:.4f} m, required={min_cc_distance*(1-adaptive_tolerance):.4f} m), "
              f"no_interlink={no_coil_interlink} (LN={linking_number:.4f})")
        
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
        
        if coil_interlinks_plasma:
            print(f"    Plasma interlink: OK (found {points_inside_hole_count} in hole, {points_in_plasma_count} in plasma, {points_outside_plasma_count} outside)")
        else:
            print(f"    Plasma interlink: FAIL (found {points_inside_hole_count} in hole, {points_in_plasma_count} in plasma, {points_outside_plasma_count} outside)")
            print(f"      Surface R range: {R_min_surface:.3f} to {R_max_surface:.3f} m")
            # Additional diagnostics
            if points_inside_hole_count == 0 and points_outside_plasma_count == 0:
                print("      -> All coil points are in plasma volume (coils need to extend both inward and outward)")
            elif points_inside_hole_count == 0:
                print("      -> No points in torus hole (coils need to extend inward)")
            elif points_outside_plasma_count == 0:
                print("      -> No points outside plasma (coils need to extend outward)")
        
        if cs_ok and cc_ok and no_coil_interlink and plasma_interlink_ok:
            # Constraints satisfied, break out of adaptive loop
            print("    All constraints satisfied! Breaking out of adaptive loop.")
            break
        
        # Check if we've exceeded maximum scales
        if R0_scale > max_R0_scale or R1_scale > max_R1_scale:
            print(f"    Warning: R0_scale ({R0_scale:.3f}) or R1_scale ({R1_scale:.3f}) exceeded maximum.")
            print("    Stopping adaptive loop to prevent coils from going too far.")
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
        adjustment_made = False
        
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
                    print(f"    Adjusting: R1_scale={R1_scale:.3f}, R0_scale={R0_scale:.3f} (extending inward to reach hole)")
                else:
                    R0 = major_radius * R0_scale
                    print(f"    Adjusting: R1_scale={R1_scale:.3f} (extending inward to reach hole)")
                adjustment_made = True
            elif points_outside_plasma_count == 0:
                # No points outside plasma - coils don't extend outward enough, need to extend outward
                # Increase R1 primarily to extend outward (R0+R1 increases)
                R1_scale *= 1.2  # More aggressive increase for R1
                R1 = minor_radius_component * R1_scale
                # Optionally decrease R0 slightly to help extend outward while keeping inner edge similar
                if min_cs_sep > min_cs_distance * 1.5:  # Large safety margin
                    R0_scale *= 0.98
                    R0 = major_radius * R0_scale
                    print(f"    Adjusting: R1_scale={R1_scale:.3f}, R0_scale={R0_scale:.3f} (extending outward beyond plasma)")
                else:
                    R0 = major_radius * R0_scale
                    print(f"    Adjusting: R1_scale={R1_scale:.3f} (extending outward beyond plasma)")
                adjustment_made = True
            else:
                # Some points in hole and outside but not both - try increasing R1 to extend more
                R1_scale *= 1.15
                R1 = minor_radius_component * R1_scale
                print(f"    Adjusting: R1_scale={R1_scale:.3f} (increasing coil extent)")
                adjustment_made = True
        elif not cs_ok:
            # Priority 2: Coils must not intersect surface (only if interlinking is OK)
            # Move coils outward to increase distance from surface
            R0_scale *= 1.1
            R0 = major_radius * R0_scale
            print(f"    Adjusting: R0_scale={R0_scale:.3f} (coils intersecting surface)")
            adjustment_made = True
        elif not cc_ok:
            # Priority 2: Coils must not be too close to each other
            # Increase R0 to move coils further from center (increases toroidal separation)
            R0_scale *= 1.1
            R0 = major_radius * R0_scale
            print(f"    Adjusting: R0_scale={R0_scale:.3f} (coils too close to each other)")
            adjustment_made = True
        elif not no_coil_interlink:
            # Priority 4: Coils should not interlink each other
            R0_scale *= 1.1
            R0 = major_radius * R0_scale
            print(f"    Adjusting: R0_scale={R0_scale:.3f} (coils interlinking)")
            adjustment_made = True
        
        if not adjustment_made:
            print("    All constraints satisfied!")
    
    # Final coil creation with determined R0 and R1
    print("\nFinal coil positioning parameters:")
    print(f"  R0: {R0:.3f} m (scale: {R0_scale:.3f}x major radius)")
    print(f"  R1: {R1:.3f} m (scale: {R1_scale:.3f}x minor radius)")
    print(f"  Major radius: {major_radius:.3f} m")
    print(f"  Minor radius component: {minor_radius_component:.3f} m")
    
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=s.stellsym,
        R0=R0, R1=R1, order=order, numquadpoints=256)
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
        
        # Calculate field strength along major radius
        B_avg = calculate_modB_on_major_radius(bs, s)
        
        # Check convergence
        if abs(B_avg - target_B) / target_B < tolerance:
            break
        
        # Adjust current based on field difference
        # Use simple linear scaling: new_current = current * (target_B / achieved_B)
        current_scale_factor = target_B / B_avg
        total_current *= current_scale_factor
    
    return coils


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
        
        print(f"Created zip archive: {zip_path}")
        print(f"  Contains {len(files_to_zip)} files")
        
        # Remove original VTK files after zipping for compression
        for file_path in files_to_zip:
            file_path.unlink()
            print(f"  Removed {file_path.name} (now in zip archive)")
    
    return zip_path


def _plot_bn_error_3d(
    surface,
    bs,
    coils,
    out_dir: Path,
    filename: str = "bn_error_3d_plot.pdf",
    title: str = "B_N/|B| Error on Plasma Surface with Optimized Coils",
    plot_upsample: int = 2,
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
    BdotN = np.sum(B_field * unit_normal, axis=2)
    abs_B = bs.AbsB().reshape((nphi_plot, ntheta_plot))
    
    # Avoid division by zero
    abs_B = np.where(abs_B > 1e-10, abs_B, 1e-10)
    bn_over_b = np.abs(BdotN / abs_B)
    
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
    
    print(f"  Saved 3D B_N/|B| error plot to: {pdf_path}")


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
    R0 = s.get_rc(0, 0)
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
        Resolution of plasma surface (nphi=ntheta) for evaluation (default: 16).
        Lower values speed up optimization but reduce accuracy. Use 8 for faster unit tests.
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
        
        print(f"\nCompleted order={order} optimization:")
        print(f"  Final flux: {results.get('final_normalized_squared_flux', 'N/A'):.2e}")
        print(f"  Final B-field: {results.get('final_B_field', 'N/A'):.3f} T")
        print(f"  Files saved to: {order_dir}")
        print(f"    - VTK files: {order_dir}/coils_optimized.*")
        print(f"    - Bn error PDF: {order_dir}/bn_error_3d_plot.pdf")
    
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
    try:
        from .post_processing import run_post_processing
        import yaml as yaml_module
        
        # Find case YAML file - try case_path first if provided
        case_yaml_path = None
        if case_path is not None:
            case_yaml_path = Path(case_path) / "case.yaml"
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
        if not case_yaml_path.exists():
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
                run_vmec=True,
                helicity_m=1,
                helicity_n=helicity_n,
                ns=50,
                plot_boozer=True,
                plot_poincare=True,
                nfieldlines=20,
            )
            print("Post-processing complete!")
            if 'quasisymmetry_total' in post_processing_results:
                print(f"  Quasisymmetry error: {post_processing_results['quasisymmetry_total']:.2e}")
        else:
            print(f"Warning: Skipping post-processing (coils_json not found: {coils_json_path})")
    except Exception as e:
        print(f"Warning: Post-processing failed: {e}")
        import traceback
        traceback.print_exc()
    
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
        surface_resolution: Resolution of plasma surface (nphi=ntheta) for evaluation (default: 16).
            Lower values speed up optimization but reduce accuracy. Use 8 for faster unit tests.
        **kwargs: Additional keyword arguments for constraint thresholds.
            max_iter_subopt: Maximum number of suboptimization iterations (default: max_iterations // 2).
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
    import time
    from scipy.optimize import minimize
    from simsopt.geo import SurfaceRZFourier
    from simsopt.geo import LinkingNumber, CurveLength, CurveCurveDistance, ArclengthVariation
    from simsopt.geo import LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature
    from simsopt.objectives import SquaredFlux, QuadraticPenalty, Weight
    from simsopt.field import BiotSavart, coils_to_vtk
    from simsopt.field.force import LpCurveForce, LpCurveTorque, coil_force, coil_torque
    from simsopt.util import calculate_modB_on_major_radius

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if this is a continuation step (initial_coils provided) to avoid duplicate work
    is_continuation_step = kwargs.get('initial_coils') is not None
    
    # nturns is constant for all cases (defined here so it's always available)
    nturns = 1  # nturns = 1 for standardization
    
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
        R0 = cached['R0']
    else:
        # First step or no cache: compute thresholds normally
        # Set default constraint thresholds if not provided
        # Defaults here are reasonable for 10 m major radius
        #  reactor-scale device with 5.7 T target B-field
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

        # Rescale thresholds by the plasma major radius only if they were not explicitly passed
        # divided by the 10m assumption for the major radius
        major_radius = s.get_rc(0, 0)  # Major radius in meters [L]
        R0 = 10.0 / major_radius  # Dimensionless scaling factor for thresholds
        if 'length_threshold' not in kwargs:
            length_threshold /= R0
        if 'cc_threshold' not in kwargs:
            cc_threshold /= R0
        if 'cs_threshold' not in kwargs:
            cs_threshold /= R0
        if 'curvature_threshold' not in kwargs:
            curvature_threshold *= R0
        if 'msc_threshold' not in kwargs:
            msc_threshold *= R0
        if 'arclength_variation_threshold' not in kwargs:
            arclength_variation_threshold *= R0 ** 2
        # coil_width is not a threshold parameter, so always scale it
        coil_width /= R0

    # If there is a suboptimization, set the max iterations 
    max_iter_subopt = kwargs.get('max_iter_subopt', max_iterations // 2)
    algorithm = kwargs.get('algorithm', 'augmented_lagrangian')
    
    # Extract algorithm-specific options from kwargs
    # These will be passed to scipy.minimize for scipy algorithms
    algorithm_options = kwargs.get('algorithm_options', {})

    # Normalize algorithm name (handle case variations)
    if isinstance(algorithm, str):
        algorithm_lower = algorithm.lower()
        if algorithm_lower in ['l-bfgs', 'lbfgs', 'l-bfgs-b']:
            algorithm = 'L-BFGS-B'
        elif algorithm_lower == 'augmented_lagrangian':
            algorithm = 'augmented_lagrangian'
        # Keep other algorithm names as-is (they should match scipy method names)

    print(f"Starting coil optimization for target B-field: {target_B} T")
    print(f"Surface major radius: {s.get_rc(0, 0):.3f} m")
    print(f"Surface minor radius component: {s.get_rc(1, 0):.3f} m")
    print(f"Number of base coils: {ncoils}")
    print(f"Fourier order: {order}")

    # Step 1: Initialize coils with target B-field
    # print("Step 1: Initializing coils with target B-field...")
    # Check if this is a continuation step (initial_coils provided) to avoid duplicate work
    is_continuation_step = initial_coils is not None
    
    if initial_coils is None:
        coils = initialize_coils_loop(s, out_dir=out_dir, target_B=target_B, ncoils=ncoils, order=order, coil_width=coil_width, regularization=regularization)
    else:
        coils = initial_coils

    # Calculate total_current (needed for later printing and possibly for threshold scaling)
    # Sum the unique base coils (coils[:ncoils]) to get total current
    total_current = sum([c.current.get_value() for c in coils[:ncoils]])
    
    # Print individual coil currents before optimization
    print("\nCoil currents before optimization (unique base coils):")
    for i, coil in enumerate(coils[:ncoils]):
        print(f"  Coil {i+1}: {coil.current.get_value():.2e} A")
    print(f"  Total: {total_current:.2e} A")
    
    # Calculate current_scale_factor for force/torque threshold and weight scaling
    # This makes force/torque thresholds and weights dimensionless relative to reactor scale
    current_scale_factor = 1.0  # Default: no scaling
    total_current_reactor_scale = None  # Will be set if needed for weight scaling
    if not is_continuation_step and ('force_threshold' not in kwargs or 'torque_threshold' not in kwargs):
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
    # print("Step 2: Setting up plotting surface...")
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
    # print("Step 3: Creating BiotSavart object and saving initial state...")
    bs = BiotSavart(coils)
    B_avg = calculate_modB_on_major_radius(bs, s)
    print(f"  Total current: {total_current:.0f} A")
    print(f"  B-field averaged along major radius: {B_avg:.3f} T")
    print(f"  Number of coils: {len(coils)}")
    curves = [c.curve for c in coils]
    
    # Save initial coils
    coils_to_vtk(coils, out_dir / "coils_initial", nturns=nturns)
    
    # Calculate and display initial B-field
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    B_initial = calculate_modB_on_major_radius(bs, s_plot)
    print(f"\nInitial B-field on-axis: {B_initial:.3f} T")
    
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
    # print("Step 4: Setting up optimization objectives and constraints...")
    bs.set_points(s.gamma().reshape((-1, 3)))
    
    # Main objective: Squared flux (always included)
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
                "": lambda obj, thresh: obj,  # Empty string defaults to including linking number
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
    major_radius = s.get_rc(0, 0)  # Major radius in meters [L]
    
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
                else:
                    print(f"Warning: Unknown option '{term_value}' for {term_name}, skipping")
    
    # Step 5: Run optimization
    start_time = time.time()
    lag_mul = None  # Initialize lag_mul for scipy methods
    
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
            cs_weight = kwargs.get(f'constraint_weight_{cs_distance_index}', 1e3) if not cs_weight_specified else kwargs.get(f'constraint_weight_{cs_distance_index}', 1.0)
            # Apply scaling to make weight dimensionless (always apply scaling for distance objectives)
            if cs_distance_index in constraint_scaling:
                cs_weight *= constraint_scaling[cs_distance_index]
            c_list[cs_distance_index] = Weight(cs_weight) * c_list[cs_distance_index]
        if cc_distance_index is not None:
            cc_weight = kwargs.get(f'constraint_weight_{cc_distance_index}', 1e3) if not cc_weight_specified else kwargs.get(f'constraint_weight_{cc_distance_index}', 1.0)
            # Apply scaling to make weight dimensionless (always apply scaling for distance objectives)
            if cc_distance_index in constraint_scaling:
                cc_weight *= constraint_scaling[cc_distance_index]
            c_list[cc_distance_index] = Weight(cc_weight) * c_list[cc_distance_index]
        
        # Print initial thresholds and weights for augmented_lagrangian
        # (weights are embedded in Weight() wrappers)
        print("Initial thresholds and weights:")
        print(f"  [0] Flux: threshold={flux_threshold:.2e}, weight=1.0")
        # Print distance objectives (they're always included at indices 1 and 2)
        # Get weights that were applied (with scaling already applied above)
        weight_cc = kwargs.get(f'constraint_weight_{cc_distance_idx}', 1e3) if not cc_weight_specified else kwargs.get(f'constraint_weight_{cc_distance_idx}', 1.0)
        weight_cs = kwargs.get(f'constraint_weight_{cs_distance_idx}', 1e3) if not cs_weight_specified else kwargs.get(f'constraint_weight_{cs_distance_idx}', 1.0)
        # Apply scaling to weights for display (always apply for distance objectives)
        if cc_distance_idx in constraint_scaling:
            weight_cc *= constraint_scaling[cc_distance_idx]
        if cs_distance_idx in constraint_scaling:
            weight_cs *= constraint_scaling[cs_distance_idx]
        print(f"  [{cc_distance_idx}] CC Distance: threshold={cc_threshold:.2e}, weight={weight_cc:.2e}")
        print(f"  [{cs_distance_idx}] CS Distance: threshold={cs_threshold:.2e}, weight={weight_cs:.2e}")
        # Print other constraints
        constraint_idx_offset = 3  # After flux (0), CC distance (1), CS distance (2)
        for i, (name, threshold) in enumerate(constraint_names_and_thresholds[2:], start=constraint_idx_offset):
            if i < len(c_list):
                weight = 1.0
                if threshold is not None:
                    print(f"  [{i}] {name}: threshold={threshold:.2e}, weight={weight:.2e}")
                else:
                    print(f"  [{i}] {name}: weight={weight:.2e}")
        
        from simsopt.solve import augmented_lagrangian_method
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
        _, _, lag_mul = augmented_lagrangian_method(
            f=None,  # No main objective function
            **augmented_lagrangian_options,
            equality_constraints=c_list,
        )
    elif algorithm in ['BFGS', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead', 'Powell', 'CG', 'Newton-CG', 'TNC', 'COBYLA', 'trust-constr']:
        # Build weighted objective function from constraints
        # c_list includes flux first, then other constraints
        # Default weight is 1.0 for all constraints
        weights = []
        
        for i, constraint in enumerate(c_list):
            # Map constraint index to weight name (for backward compatibility)
            # Flux (index 0) always has weight 1.0 (dimensionless)
            if i == 0:
                weights.append(1.0)  # Flux weight
            else:
                # For other constraints, try to get specific weight or default to 1.0
                weight_key = f'constraint_weight_{i}'
                weight_specified = weight_key in kwargs
                weight = kwargs.get(weight_key, 1.0)
                
                # Apply weight to coil-surface distance and coil-coil distance constraints
                # Use specified weight or default to 1e3 for distance constraints
                if cs_distance_index is not None and i == cs_distance_index:
                    if not cs_weight_specified:
                        weight = kwargs.get(f'constraint_weight_{i}', 1e3)
                    else:
                        weight = kwargs.get(f'constraint_weight_{i}', 1.0)
                elif cc_distance_index is not None and i == cc_distance_index:
                    if not cc_weight_specified:
                        weight = kwargs.get(f'constraint_weight_{i}', 1e3)
                    else:
                        weight = kwargs.get(f'constraint_weight_{i}', 1.0)
                
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
        
        # Print initial thresholds and weights
        print("Initial thresholds and weights:")
        print(f"  [0] Flux: threshold={flux_threshold:.2e}, weight={weights[0]:.2e}")
        # Print distance objectives (always included at indices 1 and 2)
        print(f"  [{cc_distance_idx}] CC Distance: threshold={cc_threshold:.2e}, weight={weights[cc_distance_idx]:.2e}")
        print(f"  [{cs_distance_idx}] CS Distance: threshold={cs_threshold:.2e}, weight={weights[cs_distance_idx]:.2e}")
        
        # Print other constraints (skip indices 0, 1, 2 which are flux, CC distance, CS distance)
        constraint_idx_offset = 3
        for i, (name, threshold) in enumerate(constraint_names_and_thresholds[2:], start=constraint_idx_offset):
            if i < len(weights):
                if threshold is not None:
                    print(f"  [{i}] {name}: threshold={threshold:.2e}, weight={weights[i]:.2e}")
                else:
                    print(f"  [{i}] {name}: weight={weights[i]:.2e}")
        
        # Track iteration number for objective function
        iteration_count = [0]  # Use list to allow modification in nested function

        # Define the objective function and gradient
        def objective(x: np.ndarray) -> float:
            JF.x = x  # type: ignore[attr-defined]
            J = JF.J()  # type: ignore[attr-defined]
            if verbose:
                iteration_count[0] += 1
                grad = JF.dJ()  # type: ignore[attr-defined]
                outstr = f"[{iteration_count[0]}] J={J:.1e}, Jf={Jf.J():.1e}"
                # cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
                outstr += f", Len={sum(J.J() for J in Jls):.2f}"
                outstr += f", CC-Sep={Jccdist.shortest_distance():.2f}, CS-Sep={Jcsdist.shortest_distance():.2f}"
                kappa_values = [c.kappa().max() for c in base_curves]
                msc_values = [MeanSquaredCurvature(c).J() for c in base_curves]
                kappa_str = ",".join([f"{k:.1e}" for k in kappa_values])
                msc_str = ",".join([f"{m:.1e}" for m in msc_values])
                outstr += f", κ=[{kappa_str}]"  # type: ignore[attr-defined]
                outstr += f", MSC=[{msc_str}]"  # type: ignore[attr-defined]
                outstr += f", Link#={Jlink.J():.2e}"
                outstr += f", F={Jforce.J():.2e}"
                outstr += f", T={Jtorque.J():.2e}"
                outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
                print(outstr)
                
                # Print weighted contributions of each objective term
                contrib_str = ""
                contrib_parts = []
                # Flux contribution (index 0)
                flux_contrib = weights[0] * c_list[0].J()
                contrib_parts.append(f"Flux={flux_contrib:.1e}")
                # Other constraint contributions
                for idx, (name, _) in enumerate(constraint_names_and_thresholds, start=1):
                    if idx < len(c_list) and idx < len(weights):
                        constraint_contrib = weights[idx] * c_list[idx].J()
                        contrib_parts.append(f"{name}={constraint_contrib:.1e}")
                contrib_str += ", ".join(contrib_parts)
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
                    import sys
                    print(f"WARNING: Taylor test failed: error ratio {error_ratio:.3f} > 0.6 "
                          f"(ε={epsilons[i]:.1e} -> {epsilons[i+1]:.1e}, "
                          f"error={errors[i]:.2e} -> {errors[i+1]:.2e})", file=sys.stderr)
                    taylor_test_passed = False
        
        if not taylor_test_passed:
            import sys
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
            # options.setdefault('tol', 1e-12)  # scipy default
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
            jac=gradient,  # Provide gradient function
            options=options,
        )
        
        # Print optimization result message to help debug early exits
        if verbose:
            print(f"Optimization result: {result.message}")
            print(f"  Success: {result.success}")
            print(f"  Iterations: {result.nit}")
            print(f"  Function evaluations: {result.nfev}")
            if hasattr(result, 'njev'):
                print(f"  Gradient evaluations: {result.njev}")
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.1f} seconds")
    
    # Calculate and print final total current
    # Sum the unique base coils (coils[:ncoils]) to get total current
    total_current_final = sum([c.current.get_value() for c in coils[:ncoils]])
    
    # Print individual coil currents after optimization
    print("\nCoil currents after optimization (unique base coils):")
    for i, coil in enumerate(coils[:ncoils]):
        print(f"  Coil {i+1}: {coil.current.get_value():.2e} A")
    print(f"  Total: {total_current_final:.2e} A")
    
    print(f"\nTotal current before optimization: {total_current:.0f} A")
    print(f"Total current after optimization: {total_current_final:.0f} A")
    
    # Save optimized coils
    coils_to_vtk(coils, out_dir / "coils_optimized", nturns=nturns)
    bs.save(out_dir / "biot_savart_optimized.json")
    
    # Calculate and display final B-field
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    B_final = calculate_modB_on_major_radius(bs, s_plot)
    print(f"\nFinal B-field on-axis: {B_final:.3f} T")
    if 'B_initial' in locals() and B_initial is not None:
        print(f"B-field change: {B_final - B_initial:.3f} T ({((B_final / B_initial - 1) * 100):+.1f}%)")
    
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
    
    # Print final constraint values
    bs.set_points(s.gamma().reshape((-1, 3)))
    print("Final constraint values:")
    print(f"  Normalized flux: {Jf.J():.2e}")
    print(f"  CS separation: {Jcsdist.J():.2e} (min distance: {Jcsdist.shortest_distance():.3f})")
    print(f"  CC separation: {Jccdist.J():.2e} (min distance: {Jccdist.shortest_distance():.3f})")
    print(f"  Length constraint: {sum(Jls).J():.2e}")  # type: ignore[attr-defined]
    print(f"  Curvature constraint: {sum(Jcs).J():.2e}")  # type: ignore[attr-defined]
    print(f"  Linking number: {Jlink.J():.2e}")
    print(f"  Force constraint: {Jforce.J():.2e}")
    print(f"  Max curvatures: {[np.max(c.kappa()) for c in base_curves]}")
    print(f"  Lengths: {[CurveLength(c).J() for c in base_curves]}")
    
    # Calculate final forces
    max_force = [np.max(np.linalg.norm(coil_force(c, coils), axis=1)) for c in coils[:ncoils]]
    max_torque = [np.max(np.linalg.norm(coil_torque(c, coils), axis=1)) for c in coils[:ncoils]]
    print(f"  Max forces on each coil: {[f'{f:.2e}' for f in max_force]}")
    
    # Calculate final B_N metrics
    bs.set_points(s.gamma().reshape((-1, 3)))
    B_field = bs.B().reshape((-1, 3))
    unit_normal = s.unitnormal().reshape((-1, 3))
    BdotN = np.mean(np.abs(np.sum(B_field * unit_normal, axis=1)))
    abs_B = bs.AbsB().flatten()
    avg_BdotN_over_B = BdotN / abs_B.mean() if abs_B.mean() > 0 else 0.0
    
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    nphi_plot = len(s_plot.quadpoints_phi)
    ntheta_plot = len(s_plot.quadpoints_theta)
    B_plot = bs.B().reshape((nphi_plot, ntheta_plot, 3))
    unit_normal_plot = s_plot.unitnormal().reshape((nphi_plot, ntheta_plot, 3))
    BdotN_plot = np.sum(B_plot * unit_normal_plot, axis=2)
    abs_B_plot = bs.AbsB().reshape((nphi_plot, ntheta_plot))
    # Avoid division by very small numbers (same protection as in plotting function)
    abs_B_plot = np.where(abs_B_plot > 1e-10, abs_B_plot, 1e-10)
    max_BdotN_overB = np.max(np.abs(BdotN_plot / abs_B_plot)) if np.any(abs_B_plot > 0) else 0.0
    
    print(f"  <B_N>/<|B|> = {avg_BdotN_over_B:.2e}")
    print(f"  Max |B_N|/|B| = {max_BdotN_overB:.2e}")    
    print("Optimization completed successfully!")
    print(f"Results saved to: {out_dir}")
    
    # Generate 3D visualization plot
    try:
        _plot_bn_error_3d(
            s_plot,
            bs,
            coils,
            out_dir,
            filename="bn_error_3d_plot.pdf",
            title="B_N/|B| Error on Plasma Surface with Optimized Coils",
        )
    except Exception as e:
        print(f"Warning: Failed to generate 3D plot: {e}")
    
    # Run post-processing: QFM surface, Poincaré plots, iota profiles, quasisymmetry profiles
    # Skip if this is part of Fourier continuation (will run once at the end)
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
                    run_vmec=True,  # Run VMEC for iota and quasisymmetry
                    helicity_m=1,
                    helicity_n=helicity_n,
                    ns=50,
                    plot_boozer=True,
                    plot_poincare=True,
                    nfieldlines=20,
                )
                print("Post-processing complete!")
                if 'quasisymmetry_total' in post_processing_results:
                    print(f"  Quasisymmetry error: {post_processing_results['quasisymmetry_total']:.2e}")
            else:
                print(f"Warning: Skipping post-processing (coils_json not found: {coils_json_path})")
        except Exception as e:
            print(f"Warning: Post-processing failed: {e}")
            import traceback
            traceback.print_exc()
    
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
        'R0': R0,
    }
    
    # Prepare results dictionary
    bs.set_points(s.gamma().reshape((-1, 3)))
    results = {
        'initial_B_field': B_initial,
        'final_B_field': B_final,
        'target_B_field': target_B,
        'optimization_time': end_time - start_time,
        'final_normalized_squared_flux': Jf.J(),
        '_cached_thresholds': cached_thresholds,  # Store for continuation steps
        'final_min_cs_separation': Jcsdist.shortest_distance(),
        'final_min_cc_separation': Jccdist.shortest_distance(),
        'final_total_length': sum(CurveLength(c).J() for c in base_curves),
        'final_max_curvature': max(np.max(c.kappa()) for c in base_curves),
        'final_average_curvature': np.mean([c.kappa() for c in base_curves]),
        'final_arclength_variation': np.mean([ArclengthVariation(c).J() for c in base_curves]),
        'final_mean_squared_curvature': np.max([np.mean(c.kappa() ** 2) for c in base_curves]),
        'final_linking_number': Jlink.J(),
        'final_max_max_coil_force': np.max(max_force),
        'final_avg_max_coil_force': np.mean(max_force),
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
    
    return coils, results