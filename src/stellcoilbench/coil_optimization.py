"""
Coil optimization for StellCoilBench.

Provides modular and dipole coil optimization via simsopt, with support for
augmented Lagrangian, L-BFGS-B, and other scipy algorithms. Handles threshold
scaling by minor radius, constraint scaling for dimensionless objectives,
Fourier continuation, and post-processing integration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
import numpy as np
from datetime import datetime
import zipfile
import os
import sys
from contextlib import contextmanager
from .config_scheme import CaseConfig
from .mpi_utils import comm_world, is_proc0, proc0_print
from .path_utils import find_plasma_surfaces_dir, surface_stem_from_filename
from .post_processing import timed_section, get_timing_results, suppress_output

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

# Default coil objective terms for modular coils (case overrides)
DEFAULT_COIL_OBJECTIVE_TERMS = {
    "total_length": "l2_threshold",
    "coil_curvature": "lp_threshold",
    "coil_mean_squared_curvature": "l2_threshold",
    "linking_number": "",
    "coil_arclength_variation": "l2_threshold",
}


def _compute_thresholds_from_surface(
    s: "SurfaceRZFourier",
    kwargs: Dict[str, Any],
    *,
    nturns: int = 200,
) -> Dict[str, Any]:
    """
    Compute constraint thresholds scaled by plasma minor radius.

    Scales length, distance, curvature, and MSC thresholds by a0 = ARIES_CS_MINOR_RADIUS
    / minor_radius so that constraints are dimensionless across reactor scales.
    Force/torque thresholds are scaled by nturns. Values in kwargs override defaults.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface (used for major/minor radius).
    kwargs : Dict[str, Any]
        User overrides for thresholds (e.g. cc_threshold, cs_threshold).
    nturns : int, optional
        Number of turns for force/torque scaling (default: 200).

    Returns
    -------
    Dict[str, Any]
        Thresholds dict with keys: length_threshold, flux_threshold, cc_threshold,
        cs_threshold, msc_threshold, curvature_threshold, force_threshold,
        torque_threshold, major_radius, minor_radius, a0.
    """
    length_threshold = kwargs.get('length_threshold', 200.0)
    flux_threshold = kwargs.get('flux_threshold', 1e-8)
    cc_threshold = kwargs.get('cc_threshold', 0.8)
    cs_threshold = kwargs.get('cs_threshold', 1.3)
    msc_threshold = kwargs.get('msc_threshold', 1.0)
    curvature_threshold = kwargs.get('curvature_threshold', 1.0)
    force_threshold = kwargs.get('force_threshold', 1.0) * nturns
    torque_threshold = kwargs.get('torque_threshold', 1.0) * nturns

    major_radius = s.major_radius()
    minor_radius = float(s.minor_radius())
    a0 = ARIES_CS_MINOR_RADIUS / minor_radius

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

    return {
        'length_threshold': length_threshold,
        'flux_threshold': flux_threshold,
        'cc_threshold': cc_threshold,
        'cs_threshold': cs_threshold,
        'msc_threshold': msc_threshold,
        'curvature_threshold': curvature_threshold,
        'force_threshold': force_threshold,
        'torque_threshold': torque_threshold,
        'major_radius': major_radius,
        'minor_radius': minor_radius,
        'a0': a0,
    }


def _get_optimization_thresholds(
    s: "SurfaceRZFourier",
    kwargs: Dict[str, Any],
    *,
    is_continuation_step: bool = False,
    cached: Dict[str, Any] | None = None,
    nturns: int = 200,
    coil_width_default: float = 0.4,
) -> Dict[str, Any]:
    """
    Get full optimization thresholds for the coil optimization loop.

    On continuation steps, uses cached thresholds when available. Otherwise
    computes from surface and adds arclength_variation_threshold and coil_width.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    kwargs : Dict[str, Any]
        User overrides and _cached_thresholds for continuation.
    is_continuation_step : bool, optional
        If True and cached is provided, return cached thresholds.
    cached : Dict[str, Any] | None, optional
        Cached thresholds from previous Fourier continuation step.
    nturns : int, optional
        Number of turns for force/torque scaling.
    coil_width_default : float, optional
        Default coil width before minor-radius scaling.

    Returns
    -------
    Dict[str, Any]
        Full thresholds dict including arclength_variation_threshold and coil_width.
    """
    if is_continuation_step and cached is not None:
        return {
            **cached,
            'major_radius': cached.get('major_radius', s.major_radius()),
            'minor_radius': cached.get('minor_radius', 1.7),
            'a0': cached.get('a0', cached.get('R0')),
        }
    th = _compute_thresholds_from_surface(s, kwargs, nturns=nturns)
    th['arclength_variation_threshold'] = kwargs.get('arclength_variation_threshold', 0.0)
    if 'arclength_variation_threshold' not in kwargs:
        th['arclength_variation_threshold'] *= th['a0'] ** 2
    th['coil_width'] = coil_width_default / th['a0']
    return th


def _parse_optimizer_config(
    s: "SurfaceRZFourier",
    kwargs: Dict[str, Any],
    max_iterations: int,
    *,
    is_continuation_step: bool = False,
    default_algorithm: str = 'augmented_lagrangian',
) -> Dict[str, Any]:
    """
    Parse optimizer configuration from kwargs and surface.

    Clamps max_iterations to CI cap, resolves algorithm name (e.g. lbfgs -> L-BFGS-B),
    merges algorithm_options from kwargs, and builds full thresholds dict.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    kwargs : Dict[str, Any]
        User config: algorithm, algorithm_options, max_iter_subopt, thresholds, etc.
    max_iterations : int
        Requested maximum iterations (may be clamped).
    is_continuation_step : bool, optional
        Whether this is a Fourier continuation step.
    default_algorithm : str, optional
        Default algorithm if not specified (default: augmented_lagrangian).

    Returns
    -------
    Dict[str, Any]
        Keys: algorithm, algorithm_options, max_iter_subopt, max_iterations, thresholds.
    """
    _CI_MAX_ITER_CAP = 10_000
    if max_iterations > _CI_MAX_ITER_CAP:
        print(f"Warning: max_iterations ({max_iterations}) exceeds CI cap ({_CI_MAX_ITER_CAP}); clamping.")
        max_iterations = _CI_MAX_ITER_CAP
    cached = kwargs.get('_cached_thresholds') if is_continuation_step else None
    thresholds = _get_optimization_thresholds(
        s, kwargs,
        is_continuation_step=is_continuation_step,
        cached=cached,
    )
    max_iter_subopt = kwargs.get('max_iter_subopt', 10)
    algorithm = kwargs.get('algorithm', default_algorithm)
    if isinstance(algorithm, str):
        al = algorithm.lower()
        if al in ['l-bfgs', 'lbfgs', 'l-bfgs-b']:
            algorithm = 'L-BFGS-B'
        elif al == 'augmented_lagrangian':
            algorithm = 'augmented_lagrangian'
    algorithm_options = kwargs.get('algorithm_options', {}).copy()
    valid_opts = _get_scipy_algorithm_options(algorithm)
    for opt in valid_opts:
        if opt in kwargs and opt not in algorithm_options:
            algorithm_options[opt] = kwargs[opt]
    return {
        'algorithm': algorithm,
        'algorithm_options': algorithm_options,
        'max_iter_subopt': max_iter_subopt,
        'max_iterations': max_iterations,
        'thresholds': thresholds,
    }


def _create_plotting_surface(
    s: "SurfaceRZFourier",
    surface_resolution: int,
    kwargs: Dict[str, Any],
) -> Tuple["SurfaceRZFourier", int, int]:
    """
    Create a full-torus plotting surface from the optimization surface.

    Uses plot_upsample_factor from kwargs (default 2) to set quadrature points.
    If s has a filename (VMEC input), loads from file; otherwise copies Fourier
    coefficients from s.

    Parameters
    ----------
    s : SurfaceRZFourier
        Optimization surface (may be half-period for stellarator symmetry).
    surface_resolution : int
        Base resolution (nphi = ntheta = plot_upsample_factor * surface_resolution).
    kwargs : Dict[str, Any]
        May contain plot_upsample_factor (default: 2).

    Returns
    -------
    tuple
        (s_plot, qphi, qtheta) - plotting surface and grid dimensions.
    """
    from simsopt.geo import SurfaceRZFourier
    plot_upsample = kwargs.get('plot_upsample_factor', 2)
    qphi = plot_upsample * surface_resolution
    qtheta = plot_upsample * surface_resolution
    quadpoints_phi = np.linspace(0, 1, qphi)
    quadpoints_theta = np.linspace(0, 1, qtheta)
    if hasattr(s, 'filename') and s.filename is not None:
        s_plot = SurfaceRZFourier.from_vmec_input(
            s.filename, range="full torus",
            quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta,
        )
    else:
        s_plot = SurfaceRZFourier(
            nfp=s.nfp, stellsym=s.stellsym,
            mpol=s.mpol, ntor=s.ntor,
            quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta,
        )
    for m in range(s.mpol + 1):
        for n in range(-s.ntor, s.ntor + 1):
            if s.get_rc(m, n) != 0:
                s_plot.set_rc(m, n, s.get_rc(m, n))
            if s.get_zs(m, n) != 0:
                s_plot.set_zs(m, n, s.get_zs(m, n))
    return s_plot, qphi, qtheta


def _build_scipy_minimize_options(
    algorithm: str,
    max_iterations: int,
    algorithm_options: Dict[str, Any],
    max_iter_subopt: int | None = None,
) -> Dict[str, Any]:
    """
    Build options dict for scipy.optimize.minimize.

    Sets algorithm-specific defaults (ftol, gtol, maxfun) and merges user
    algorithm_options. Validates options against _get_scipy_algorithm_options.

    Parameters
    ----------
    algorithm : str
        Algorithm name (e.g. L-BFGS-B, BFGS, SLSQP).
    max_iterations : int
        Maximum iterations for outer loop.
    algorithm_options : Dict[str, Any]
        User-provided options to merge.
    max_iter_subopt : int | None, optional
        Unused; kept for API compatibility.

    Returns
    -------
    Dict[str, Any]
        Options dict suitable for scipy.optimize.minimize(..., options=...).
    """
    options = {'maxiter': max_iterations}
    if algorithm == 'L-BFGS-B':
        options.setdefault('ftol', 1e-12)
        options.setdefault('gtol', 1e-12)
    elif algorithm == 'TNC':
        options.setdefault('ftol', 1e-6)
        options.setdefault('gtol', 1e-05)
    elif algorithm == 'COBYLA':
        options.setdefault('tol', 1e-12)
    if algorithm in ['L-BFGS-B', 'TNC']:
        if 'maxfun' not in options:
            options['maxfun'] = max_iterations * 15000
    if algorithm_options:
        _validate_algorithm_options(algorithm, algorithm_options)
        options.update(algorithm_options)
    return options


def _run_taylor_test(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    verbose: bool = False,
) -> bool:
    """
    Run Taylor test to verify gradient correctness.

    Checks that J(x0 + εh) ≈ J(x0) + ε * ∇J · h for decreasing ε.
    Fails if error ratio between successive ε is > 0.6 (gradient inconsistent).

    Parameters
    ----------
    objective : Callable
        Scalar objective J(x).
    gradient : Callable
        Gradient function returning ∇J(x).
    x0 : np.ndarray
        Point at which to test.
    verbose : bool, optional
        If True, print success message when test passes.

    Returns
    -------
    bool
        True if test passed, False otherwise.
    """
    np.random.seed(42)
    h = np.random.randn(len(x0))
    h = h / np.linalg.norm(h)
    J0 = objective(x0)
    grad0 = gradient(x0)
    epsilons = [1e-6, 1e-7, 1e-8]
    errors = []
    for eps in epsilons:
        xp = x0 + eps * h
        Jp = objective(xp)
        Jpred = J0 + eps * np.dot(grad0, h)
        err = abs(Jp - Jpred) / (abs(J0) + 1e-12)
        errors.append(err)
    passed = True
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i + 1] / errors[i] > 0.6:
            print(f"WARNING: Taylor test failed: error ratio {errors[i+1]/errors[i]:.3f} > 0.6 "
                  f"(ε={epsilons[i]:.1e} -> {epsilons[i+1]:.1e})", file=sys.stderr)
            passed = False
    if passed and verbose:
        print("Taylor test passed: error decreases as expected")
    return passed


def _apply_distance_weights_for_auglag(
    c_list: list,
    constraint_scaling: Dict[int, float],
    cc_distance_idx: int | None,
    cs_distance_idx: int | None,
    kwargs: Dict[str, Any],
    extra_distance_indices: list[int] | None = None,
) -> None:
    """
    Apply weights to distance constraints for augmented Lagrangian (in-place).

    Replaces c_list[idx] with Weight(w) * c_list[idx] for each distance constraint
    index. Weight includes constraint_scaling for dimensionless objectives.
    extra_distance_indices supports dipole's second cc-distance constraint.

    Parameters
    ----------
    c_list : list
        List of constraint objectives (modified in-place).
    constraint_scaling : Dict[int, float]
        Scaling factors per constraint index.
    cc_distance_idx, cs_distance_idx : int | None
        Indices of coil-coil and coil-surface distance constraints.
    kwargs : Dict[str, Any]
        May contain constraint_weight_{idx} overrides.
    extra_distance_indices : list[int] | None, optional
        Additional distance indices (e.g. dipole cc_dist2).
    """
    from simsopt.objectives import Weight
    indices = [i for i in [cs_distance_idx, cc_distance_idx] if i is not None]
    if extra_distance_indices:
        indices.extend(extra_distance_indices)
    for idx in indices:
        w = kwargs.get(f'constraint_weight_{idx}', 1e3)
        if idx in constraint_scaling:
            w *= constraint_scaling[idx]
        c_list[idx] = Weight(w) * c_list[idx]


def _compute_constraint_scaling_for_term(
    term_name: str,
    term_value: str,
    major_radius: float,
    total_current: float,
    p_value: int,
    base_scaling: float,
) -> float:
    """
    Compute scaling factor to make weight * constraint dimensionless.

    Different formulas for l2/l2_threshold vs lp/lp_threshold. Ensures
    optimization is scale-invariant across reactor sizes.

    Parameters
    ----------
    term_name : str
        Constraint name (e.g. total_length, coil_curvature, coil_coil_force).
    term_value : str
        Option (l2, l2_threshold, lp, lp_threshold, l1, l1_threshold, "").
    major_radius : float
        Plasma major radius [m].
    total_current : float
        Total coil current [A].
    p_value : int
        Lp norm exponent for lp/lp_threshold terms.
    base_scaling : float
        Base scaling from _get_base_scaling_for_term.

    Returns
    -------
    float
        Scaling factor to multiply constraint weight.
    """
    if term_value in ["l2", "l2_threshold"]:
        if term_name == "total_length":
            return base_scaling / major_radius
        elif term_name == "coil_curvature":
            return base_scaling * major_radius
        elif term_name == "coil_mean_squared_curvature":
            return base_scaling * (major_radius ** 2)
        elif term_name == "coil_arclength_variation":
            return base_scaling / (major_radius ** 2)
        return base_scaling
    elif term_value in ["lp", "lp_threshold"]:
        if term_name == "coil_curvature":
            return major_radius ** (p_value - 1)
        elif term_name in ["coil_coil_force", "coil_coil_torque"]:
            return (major_radius ** (p_value - 1)) / (total_current ** (2 * p_value))
        elif term_name in ["total_length", "coil_coil_distance", "coil_surface_distance"]:
            return base_scaling / (major_radius ** (p_value - 1))
        elif term_name == "coil_mean_squared_curvature":
            return base_scaling * (major_radius ** (2 * p_value - 2))
        elif term_name == "coil_arclength_variation":
            return base_scaling / (major_radius ** (2 * p_value - 2))
        return base_scaling
    elif term_value == "":
        return base_scaling
    else:
        return base_scaling


def _get_base_scaling_for_term(term_name: str, major_radius: float) -> float:
    """
    Return base scaling for l1/l1_threshold (linear penalty) terms.

    Parameters
    ----------
    term_name : str
        Constraint name (e.g. total_length, coil_curvature).
    major_radius : float
        Plasma major radius [m].

    Returns
    -------
    float
        Base scaling factor (1/R0, 1/R0^2, R0, R0^2, or 1.0).
    """
    if term_name == "total_length":
        return 1.0 / major_radius
    elif term_name in ["coil_coil_distance", "coil_surface_distance"]:
        return 1.0 / (major_radius ** 2)
    elif term_name == "coil_curvature":
        return major_radius
    elif term_name == "coil_mean_squared_curvature":
        return major_radius ** 2
    elif term_name == "coil_arclength_variation":
        return 1.0 / (major_radius ** 2)
    elif term_name == "linking_number":
        return 1.0
    elif term_name in ["coil_coil_force", "coil_coil_torque"]:
        return 1.0
    return 1.0


def _build_weights_for_scipy_minimize(
    c_list: list,
    constraint_scaling: Dict[int, float],
    constraint_idx_to_term: Dict[int, str],
    cc_distance_idx: int | None,
    cs_distance_idx: int | None,
    kwargs: Dict[str, Any],
    coil_objective_terms: Dict[str, Any] | None,
) -> list:
    """
    Build weights list for weighted objective JF = sum(Weight(w)*c).

    Flux (index 0) gets flux_weight or 1.0. Other constraints get weights from
    coil_objective_terms (e.g. length_weight, cc_weight) or kwargs
    (constraint_weight_{i}). Distance constraints default to 1e3 if unspecified.
    Applies constraint_scaling for dimensionless objectives.

    Parameters
    ----------
    c_list : list
        Constraint objectives (flux first, then distance, length, etc.).
    constraint_scaling : Dict[int, float]
        Scaling per constraint index.
    constraint_idx_to_term : Dict[int, str]
        Maps constraint index to term name.
    cc_distance_idx, cs_distance_idx : int | None
        Indices of coil-coil and coil-surface distance constraints.
    kwargs : Dict[str, Any]
        constraint_weight_{i} overrides.
    coil_objective_terms : Dict[str, Any] | None
        Case config with flux_weight, length_weight, cc_weight, etc.

    Returns
    -------
    list
        List of float weights, one per constraint in c_list.
    """
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
    cs_weight_specified = cs_distance_idx is not None and f'constraint_weight_{cs_distance_idx}' in kwargs
    cc_weight_specified = cc_distance_idx is not None and f'constraint_weight_{cc_distance_idx}' in kwargs

    weights = []
    for i, _ in enumerate(c_list):
        if i == 0:
            if coil_objective_terms and "flux_weight" in coil_objective_terms:
                weights.append(float(coil_objective_terms["flux_weight"]))
            else:
                weights.append(1.0)
        else:
            weight_specified = f'constraint_weight_{i}' in kwargs
            weight = kwargs.get(f'constraint_weight_{i}', 1.0)
            term_name = constraint_idx_to_term.get(i)
            if term_name and coil_objective_terms:
                weight_param = term_to_weight_key.get(term_name)
                if weight_param and weight_param in coil_objective_terms:
                    weight = float(coil_objective_terms[weight_param])
                    weight_specified = True
            if cs_distance_idx is not None and i == cs_distance_idx:
                if coil_objective_terms and "cs_weight" in coil_objective_terms:
                    weight = float(coil_objective_terms["cs_weight"])
                    weight_specified = True
                elif cs_weight_specified:
                    weight = kwargs[f'constraint_weight_{i}']
                else:
                    weight = kwargs.get(f'constraint_weight_{i}', 1e3)
            elif cc_distance_idx is not None and i == cc_distance_idx:
                if coil_objective_terms and "cc_weight" in coil_objective_terms:
                    weight = float(coil_objective_terms["cc_weight"])
                    weight_specified = True
                elif cc_weight_specified:
                    weight = kwargs[f'constraint_weight_{i}']
                else:
                    weight = kwargs.get(f'constraint_weight_{i}', 1e3)
            if i in constraint_scaling:
                dist_indices = [x for x in [cc_distance_idx, cs_distance_idx] if x is not None]
                if i in dist_indices:
                    weight *= constraint_scaling[i]
                elif not weight_specified:
                    weight *= constraint_scaling[i]
            weights.append(weight)
    return weights


def _build_c_list_and_constraint_scaling_from_coil_objective_terms(
    Jf: Any,
    Jccdist: Any,
    Jcsdist: Any,
    Jls: list,
    Jcs: list,
    Jalenvar: list,
    Jmscs: list,
    Jlink: Any,
    Jforce: Any,
    Jtorque: Any,
    coil_objective_terms: Dict[str, Any] | None,
    thresholds: Dict[str, float],
    major_radius: float,
    total_current: float,
    *,
    dipole_length_split: tuple[list, list, float, float] | None = None,
) -> tuple[list, Dict[int, float], int, int, list, Dict[int, str]]:
    """
    Build constraint list and scaling from coil_objective_terms.

    Always includes flux (Jf), coil-coil distance (Jccdist), coil-surface distance
    (Jcsdist). Adds length, curvature, arclength_variation, MSC, linking_number,
    force, torque based on coil_objective_terms. Computes constraint_scaling for
    dimensionless weights.

    Parameters
    ----------
    Jf, Jccdist, Jcsdist : objectives
        Flux and distance objectives.
    Jls, Jcs, Jalenvar, Jmscs : list
        Per-coil objectives (length, curvature, arclength variation, MSC).
    Jlink, Jforce, Jtorque : objectives
        Linking number and force/torque.
    coil_objective_terms : Dict[str, Any] | None
        Case config specifying which terms and options (l2, lp_threshold, etc.).
    thresholds : Dict[str, float]
        Threshold values for each constraint type.
    major_radius, total_current : float
        For constraint scaling.

    Returns
    -------
    tuple
        (c_list, constraint_scaling, cc_distance_idx, cs_distance_idx,
         constraint_names_and_thresholds, constraint_idx_to_term).
    """
    from simsopt.objectives import QuadraticPenalty
    c_list = [Jf]
    cc_distance_idx = len(c_list)
    c_list.append(Jccdist)
    cs_distance_idx = len(c_list)
    c_list.append(Jcsdist)
    constraint_names_and_thresholds = [("CC Distance", thresholds["cc_threshold"]), ("CS Distance", thresholds["cs_threshold"])]
    constraint_scaling = {
        cc_distance_idx: 1.0 / (major_radius ** 2),
        cs_distance_idx: 1.0 / (major_radius ** 2),
    }
    constraint_idx_to_term = {}

    if not coil_objective_terms:
        return c_list, constraint_scaling, cc_distance_idx, cs_distance_idx, constraint_names_and_thresholds, constraint_idx_to_term

    length_threshold = thresholds["length_threshold"]
    curvature_threshold = thresholds["curvature_threshold"]
    arclength_variation_threshold = thresholds.get("arclength_variation_threshold", 0.0)
    msc_threshold = thresholds["msc_threshold"]
    force_threshold = thresholds["force_threshold"]
    torque_threshold = thresholds["torque_threshold"]

    term_map = {
        "total_length": {
            "obj": sum(Jls), "threshold": length_threshold,
            "l1": lambda obj, thresh: obj, "l1_threshold": lambda obj, thresh: obj,
            "l2": lambda obj, thresh: QuadraticPenalty(obj, 0.0, "max"),
            "l2_threshold": lambda obj, thresh: QuadraticPenalty(obj, thresh, "max"),
        },
        "coil_curvature": {
            "obj": sum(Jcs), "threshold": curvature_threshold,
            "lp": lambda obj, thresh: obj, "lp_threshold": lambda obj, thresh: obj,
        },
        "coil_arclength_variation": {
            "obj": Jalenvar, "threshold": arclength_variation_threshold,
            "l2": lambda obj, thresh: sum([QuadraticPenalty(j, 0.0, "max") for j in obj]),
            "l2_threshold": lambda obj, thresh: sum([QuadraticPenalty(j, thresh, "max") for j in obj]),
            "l1": lambda obj, thresh: sum(obj), "l1_threshold": lambda obj, thresh: sum(obj),
        },
        "coil_mean_squared_curvature": {
            "obj": Jmscs, "threshold": msc_threshold,
            "l2": lambda obj, thresh: sum([QuadraticPenalty(j, 0.0, "max") for j in obj]),
            "l2_threshold": lambda obj, thresh: sum([QuadraticPenalty(j, thresh, "max") for j in obj]),
            "l1": lambda obj, thresh: sum(obj), "l1_threshold": lambda obj, thresh: sum(obj),
        },
        "linking_number": {"obj": Jlink, "threshold": None, "": lambda obj, thresh: obj},
        "coil_coil_force": {"obj": Jforce, "threshold": force_threshold, "lp": lambda obj, thresh: obj, "lp_threshold": lambda obj, thresh: obj},
        "coil_coil_torque": {"obj": Jtorque, "threshold": torque_threshold, "lp": lambda obj, thresh: obj, "lp_threshold": lambda obj, thresh: obj},
    }
    name_map = {
        "total_length": ("Length", length_threshold),
        "coil_mean_squared_curvature": ("MSC", msc_threshold),
        "coil_arclength_variation": ("Arclength Var", arclength_variation_threshold),
        "coil_curvature": ("κ", curvature_threshold),
        "linking_number": ("Link #", None),
        "coil_coil_force": ("Force", force_threshold),
        "coil_coil_torque": ("Torque", torque_threshold),
    }

    for term_name, term_value in coil_objective_terms.items():
        if term_name.endswith("_p"):
            continue
        if term_name not in term_map:
            continue
        term_config = term_map[term_name]
        if term_value not in term_config:
            print(f"Warning: Unknown option '{term_value}' for {term_name}, skipping")
            continue

        # Special case: separate length penalties for dipole vs TF when dipole_length_split provided
        if term_name == "total_length" and dipole_length_split is not None:
            Jls_dipole, Jls_tf, thresh_dipole, thresh_tf = dipole_length_split
            obj_dipole = sum(Jls_dipole)
            obj_tf = sum(Jls_tf)
            penalty_fn = term_config[term_value]
            constraint = penalty_fn(obj_dipole, thresh_dipole) + penalty_fn(obj_tf, thresh_tf)
        else:
            obj = term_config["obj"]
            thresh = term_config["threshold"]
            constraint = term_config[term_value](obj, thresh)

        constraint_idx = len(c_list)
        c_list.append(constraint)
        p_value = 2
        if term_value in ["lp", "lp_threshold"]:
            p_value = coil_objective_terms.get(f"{term_name}_p", 2)
        base_scaling = _get_base_scaling_for_term(term_name, major_radius)
        constraint_scaling[constraint_idx] = _compute_constraint_scaling_for_term(
            term_name, term_value, major_radius, total_current, p_value, base_scaling
        )
        if term_name in name_map:
            constraint_names_and_thresholds.append(name_map[term_name])
        constraint_idx_to_term[constraint_idx] = term_name

    return c_list, constraint_scaling, cc_distance_idx, cs_distance_idx, constraint_names_and_thresholds, constraint_idx_to_term


def _build_modular_coil_constraint_objects(
    curves: list,
    base_curves: list,
    coils: list,
    ncoils: int,
    s: Any,
    cc_threshold: float,
    cs_threshold: float,
    curvature_threshold: float,
    force_threshold: float,
    torque_threshold: float,
    coil_objective_terms: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Build constraint objectives for modular (non-dipole) coils.

    Creates CurveCurveDistance, CurveSurfaceDistance, LinkingNumber,
    LpCurveCurvature, MeanSquaredCurvature, ArclengthVariation, LpCurveForce,
    LpCurveTorque. Force/torque thresholds are set to 0 for lp (no threshold)
    or to force_threshold/torque_threshold for lp_threshold.

    Parameters
    ----------
    curves, base_curves : list
        All curves and base (unique) curves.
    coils : list
        Coil objects (for force/torque).
    ncoils : int
        Number of base coils.
    s : Surface
        Plasma surface (for coil-surface distance).
    cc_threshold, cs_threshold, curvature_threshold : float
        Distance and curvature thresholds.
    force_threshold, torque_threshold : float
        Force/torque thresholds (used only for lp_threshold option).
    coil_objective_terms : Dict[str, Any] | None
        Case config for curvature_p, force_p, torque_p and lp vs lp_threshold.

    Returns
    -------
    Dict[str, Any]
        Keys: Jls, Jccdist, Jcsdist, Jalenvar, Jcs, Jlink, Jforce, Jtorque, Jmscs.
    """
    from simsopt.geo import (
        CurveCurveDistance,
        CurveSurfaceDistance,
        LinkingNumber,
        LpCurveCurvature,
        CurveLength,
        ArclengthVariation,
        MeanSquaredCurvature,
    )
    from simsopt.field.force import LpCurveForce, LpCurveTorque

    curvature_p = coil_objective_terms.get("coil_curvature_p", 2) if coil_objective_terms else 2
    force_p = coil_objective_terms.get("coil_coil_force_p", 2) if coil_objective_terms else 2
    torque_p = coil_objective_terms.get("coil_coil_torque_p", 2) if coil_objective_terms else 2
    force_thresh = force_threshold
    torque_thresh = torque_threshold
    if coil_objective_terms:
        if coil_objective_terms.get("coil_coil_force") and "threshold" in str(coil_objective_terms.get("coil_coil_force", "")):
            force_thresh = force_threshold
        else:
            force_thresh = 0.0
        if coil_objective_terms.get("coil_coil_torque") and "threshold" in str(coil_objective_terms.get("coil_coil_torque", "")):
            torque_thresh = torque_threshold
        else:
            torque_thresh = 0.0

    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, cs_threshold)
    Jalenvar = [ArclengthVariation(c) for c in base_curves]
    Jcs = [LpCurveCurvature(c, curvature_p, curvature_threshold) for c in base_curves]
    Jlink = LinkingNumber(curves, downsample=2)
    Jforce = LpCurveForce(coils[:ncoils], coils, p=force_p, threshold=force_thresh, downsample=2)
    Jtorque = LpCurveTorque(coils[:ncoils], coils, p=torque_p, threshold=torque_thresh, downsample=2)
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    return {
        "Jls": Jls,
        "Jccdist": Jccdist,
        "Jcsdist": Jcsdist,
        "Jalenvar": Jalenvar,
        "Jcs": Jcs,
        "Jlink": Jlink,
        "Jforce": Jforce,
        "Jtorque": Jtorque,
        "Jmscs": Jmscs,
    }


def _build_dipole_coil_constraint_objects(
    curves: list,
    base_curves_dipole: list,
    base_curves_TF: list,
    dipole_coils: list,
    coils: list,
    fix_shapes: bool,
    fix_center: bool,
    fix_orientation: bool,
    s: Any,
    cc_threshold: float,
    cs_threshold: float,
    curvature_threshold: float,
    force_threshold: float,
    torque_threshold: float,
    coil_objective_terms: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Build constraint objectives for dipole + TF coils.

    Creates CurveCurveDistance, CurveSurfaceDistance, LinkingNumber, CurveLength,
    LpCurveCurvature, MeanSquaredCurvature, ArclengthVariation, LpCurveForce,
    and LpCurveTorque objectives for the combined dipole and TF coil set.

    Parameters
    ----------
    curves : list
        All coil curves (dipole + TF, after symmetrization).
    base_curves_dipole : list
        Base dipole curves (before symmetrization).
    base_curves_TF : list
        Base TF curves (before symmetrization).
    dipole_coils : list
        Dipole coil objects (all symmetrized).
    coils : list
        TF coil objects (all symmetrized).
    fix_shapes : bool
        If True, exclude dipole base curves from curvature, MSC, and arclength
        variation (no shape penalties on dipole coils).
    fix_center, fix_orientation : bool
        If True (with fix_shapes), dipole positions/orientations are fixed;
        then Jccdist, Jcsdist, Jlink use only TF curves.
    s : Surface
        Plasma surface (for coil-surface distance).
    cc_threshold, cs_threshold, curvature_threshold : float
        Distance and curvature thresholds.
    force_threshold, torque_threshold : float
        Force/torque thresholds (used for lp_threshold option).
    coil_objective_terms : Dict[str, Any] | None
        Case config for curvature_p, force_p, torque_p and lp vs lp_threshold.

    Returns
    -------
    Dict[str, Any]
        Keys: Jls, Jccdist, Jcsdist, Jalenvar, Jcs, Jlink, Jforce, Jtorque, Jmscs.
    """
    from simsopt.geo import (
        CurveCurveDistance,
        CurveSurfaceDistance,
        LinkingNumber,
        LpCurveCurvature,
        CurveLength,
        ArclengthVariation,
        MeanSquaredCurvature,
    )
    from simsopt.field.force import LpCurveForce, LpCurveTorque

    curvature_p = coil_objective_terms.get("coil_curvature_p", 2) if coil_objective_terms else 2
    force_p = coil_objective_terms.get("coil_coil_force_p", 2) if coil_objective_terms else 2
    torque_p = coil_objective_terms.get("coil_coil_torque_p", 2) if coil_objective_terms else 2
    force_thresh = force_threshold
    torque_thresh = torque_threshold
    if coil_objective_terms:
        if coil_objective_terms.get("coil_coil_force") and "threshold" in str(coil_objective_terms.get("coil_coil_force", "")):
            force_thresh = force_threshold
        else:
            force_thresh = 0.0
        if coil_objective_terms.get("coil_coil_torque") and "threshold" in str(coil_objective_terms.get("coil_coil_torque", "")):
            torque_thresh = torque_threshold
        else:
            torque_thresh = 0.0

    bcd = list(base_curves_dipole)
    bct = list(base_curves_TF)
    ncoils_dipoles = len(bcd)
    ncoils_TF = len(bct)
    # When fix_shapes and fix_center, dipole positions are fixed; they cannot move or
    # become interlinked (with the initialization used). Exclude dipoles from Jls,
    # Jccdist, Jcsdist, Jlink since those terms are constant for dipoles.
    dipoles_position_fixed = fix_shapes and fix_center and ncoils_dipoles > 0
    curves_for_dist = [c.curve for c in coils] if dipoles_position_fixed else curves
    num_basecurves_dist = ncoils_TF if dipoles_position_fixed else ncoils_dipoles + ncoils_TF

    if dipoles_position_fixed:
        Jls = [CurveLength(c) for c in bct]
    else:
        Jls = [CurveLength(c) for c in bcd] + [CurveLength(c) for c in bct]
    Jccdist = CurveCurveDistance(curves_for_dist, cc_threshold, num_basecurves=num_basecurves_dist)
    Jcsdist = CurveSurfaceDistance(curves_for_dist, s, cs_threshold)
    Jlink = LinkingNumber(curves_for_dist, downsample=2)

    if fix_shapes:
        base_curves_for_shape = bct
    else:
        base_curves_for_shape = bcd + bct
    Jalenvar = [ArclengthVariation(c) for c in base_curves_for_shape]
    Jcs = [LpCurveCurvature(c, curvature_p, curvature_threshold) for c in base_curves_for_shape]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves_for_shape]
    # When dipole_coils is empty, use coils for both source args (LpCurveForce needs non-empty sources)
    src_coarse = dipole_coils if dipole_coils else coils
    src_fine = coils
    Jforce = LpCurveForce(
        coils[:ncoils_TF],
        source_coils_coarse=src_coarse,
        source_coils_fine=src_fine,
        p=force_p,
        threshold=force_thresh,
        downsample=2,
    )
    if ncoils_dipoles > 0:
        Jforce = Jforce + LpCurveForce(
            dipole_coils[:ncoils_dipoles],
            source_coils_coarse=dipole_coils,
            source_coils_fine=coils,
            p=force_p,
            threshold=force_thresh,
            downsample=2,
        )
    Jtorque = LpCurveTorque(
        coils[:ncoils_TF],
        source_coils_coarse=src_coarse,
        source_coils_fine=src_fine,
        p=torque_p,
        threshold=torque_thresh,
        downsample=2,
    )
    if ncoils_dipoles > 0:
        Jtorque = Jtorque + LpCurveTorque(
            dipole_coils[:ncoils_dipoles],
            source_coils_coarse=dipole_coils,
            source_coils_fine=coils,
            p=torque_p,
            threshold=torque_thresh,
            downsample=2,
        )

    return {
        "Jls": Jls,
        "Jccdist": Jccdist,
        "Jcsdist": Jcsdist,
        "Jalenvar": Jalenvar,
        "Jcs": Jcs,
        "Jlink": Jlink,
        "Jforce": Jforce,
        "Jtorque": Jtorque,
        "Jmscs": Jmscs,
    }


def _setup_biotSavart_and_initial_save(
    coils: list,
    s: Any,
    s_plot: Any,
    qphi: int,
    qtheta: int,
    out_dir: Path,
) -> tuple:
    """
    Create BiotSavart and save initial state before optimization.

    Saves coils to VTK (coils_initial), surface with B_N/|B| and modB to VTK
    (surface_initial), and generates bn_error_3d_plot_initial.pdf.

    Parameters
    ----------
    coils : list
        Coil objects.
    s, s_plot : Surface
        Optimization surface and plotting surface (full torus).
    qphi, qtheta : int
        Plotting grid dimensions.
    out_dir : Path
        Output directory.

    Returns
    -------
    tuple
        (bs, curves, B_initial) - BiotSavart, curve list, initial |B| on s_plot.
    """
    from simsopt.field import BiotSavart, coils_to_vtk
    from simsopt.util import calculate_modB_on_major_radius

    bs = BiotSavart(coils)
    with suppress_output():
        calculate_modB_on_major_radius(bs, s)
    curves = [c.curve for c in coils]

    try:
        coils_to_vtk(coils, out_dir / "coils_initial")
    except Exception as e:
        print(f"Warning: Failed to save initial coils to VTK: {e}")
        print("  Continuing optimization without VTK export...")

    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    with suppress_output():
        B_initial = calculate_modB_on_major_radius(bs, s_plot)

    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                          s_plot.unitnormal(), axis=2)[:, :, None] /
                    bs.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs.AbsB().reshape((qphi, qtheta, 1))
    }
    s_plot.to_vtk(out_dir / "surface_initial", extra_data=pointData)

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

    return bs, curves, B_initial


def _compute_lorentz_force_torque_fallback(
    coil_subset: list,
    all_coils: list,
) -> tuple[list[float], list[float]]:
    """Compute max force [N/m] and torque [N] per coil via Lorentz formula when coil_force unavailable.

    F/L = I * (t × B), τ = r × F. Excludes self-field to avoid singularity.
    """
    from simsopt.field import BiotSavart

    max_force = []
    max_torque = []
    for c in coil_subset:
        other_coils = [ac for ac in all_coils if id(ac) != id(c)]
        if not other_coils:
            max_force.append(0.0)
            max_torque.append(0.0)
            continue
        bs = BiotSavart(other_coils)
        curve = c.curve
        gamma = curve.gamma()
        gammadash = curve.gammadash() if hasattr(curve, "gammadash") else curve.dgamma_by_dphi()
        I_val = float(abs(c.current.get_value()))
        pts = gamma.reshape(-1, 3)
        bs.set_points(pts)
        B = bs.B().reshape(-1, 3)
        ds = np.linalg.norm(gammadash, axis=1, keepdims=True)
        ds = np.where(ds > 1e-14, ds, 1.0)
        tangent = gammadash / ds
        force_density = I_val * np.cross(tangent, B)
        force_mag = np.linalg.norm(force_density, axis=1)
        max_force.append(float(np.max(force_mag)))
        torque_density = np.cross(pts, force_density)
        torque_mag = np.linalg.norm(torque_density, axis=1)
        max_torque.append(float(np.max(torque_mag)))
    return max_force, max_torque


def _compute_coil_subset_metrics(
    coil_subset: list,
    base_curves_subset: list,
    all_coils: list,
    s: Any,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute metrics for a subset of coils (dipole or TF only).

    Used for dipole runs to report dipole and TF metrics separately.
    Force/torque use all_coils for mutual interaction.
    """
    from simsopt.geo import CurveLength, ArclengthVariation

    n = len(coil_subset)
    if n == 0 or len(base_curves_subset) == 0:
        return {
            "total_current": 0.0,
            "max_force": [],
            "max_torque": [],
            "coils_linked_to_surface": False,
            "final_length_per_coil": [],
            "final_current_per_coil": [],
            "final_total_length": 0.0,
            "final_max_curvature": 0.0,
            "final_average_curvature": 0.0,
            "final_arclength_variation": 0.0,
            "final_mean_squared_curvature": 0.0,
        }

    currents = [float(abs(c.current.get_value())) for c in coil_subset] if coil_subset else []
    total_current = sum(currents)

    if n > 0 and len(all_coils) > 0:
        try:
            from simsopt.field.force import coil_force, coil_torque
            max_force = [np.max(np.linalg.norm(coil_force(c, all_coils), axis=1)) for c in coil_subset]
            max_torque = [np.max(np.linalg.norm(coil_torque(c, all_coils), axis=1)) for c in coil_subset]
        except (ImportError, Exception):
            max_force, max_torque = _compute_lorentz_force_torque_fallback(coil_subset, all_coils)
    else:
        max_force = [0.0] * n
        max_torque = [0.0] * n

    coils_linked = _check_coils_linked_to_surface(s, base_curves_subset)

    lengths = [float(CurveLength(c).J()) for c in base_curves_subset]
    kappas = [c.kappa() for c in base_curves_subset]

    return {
        "total_current": float(total_current),
        "max_force": [float(f) for f in max_force],
        "max_torque": [float(t) for t in max_torque],
        "coils_linked_to_surface": coils_linked,
        "final_length_per_coil": lengths,
        "final_current_per_coil": currents,
        "final_total_length": float(sum(lengths)),
        "final_max_curvature": float(np.max([np.max(k) for k in kappas])) if kappas else 0.0,
        "final_average_curvature": float(np.mean([np.mean(k) for k in kappas])) if kappas else 0.0,
        "final_arclength_variation": float(np.mean([ArclengthVariation(c).J() for c in base_curves_subset])),
        "final_mean_squared_curvature": float(np.max([np.mean(c.kappa() ** 2) for c in base_curves_subset])),
    }


def _compute_optimization_metrics(
    bs: Any,
    coils: list,
    base_curves: list,
    ncoils: int,
    s: Any,
    s_plot: Any,
    qphi: int,
    qtheta: int,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute final metrics (B_final, force/torque, B_N, coil-surface linking).

    Shared by modular and dipole paths. Does not save files.
    """
    from simsopt.util import calculate_modB_on_major_radius

    try:
        total_current_final = sum(c.current.get_value() for c in coils[:ncoils])
    except (AttributeError, TypeError):
        total_current_final = sum(
            float(abs(coils[i].current.get_value()))
            for i in range(min(ncoils, len(coils)))
            if hasattr(coils[i], 'current')
        )

    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    with suppress_output():
        B_final = calculate_modB_on_major_radius(bs, s_plot)

    if ncoils > 0 and len(coils) > 0:
        if hasattr(coils[0], 'force') and hasattr(coils[0], 'torque'):
            max_force = [np.max(np.linalg.norm(c.force(coils), axis=1)) for c in coils[:ncoils]]
            max_torque = [np.max(np.linalg.norm(c.torque(coils), axis=1)) for c in coils[:ncoils]]
        else:
            try:
                from simsopt.field.force import coil_force, coil_torque
                max_force = [np.max(np.linalg.norm(coil_force(c, coils), axis=1)) for c in coils[:ncoils]]
                max_torque = [np.max(np.linalg.norm(coil_torque(c, coils), axis=1)) for c in coils[:ncoils]]
            except (ImportError, Exception):
                subset = coils[:ncoils]
                max_force, max_torque = _compute_lorentz_force_torque_fallback(subset, coils)
    else:
        max_force = []
        max_torque = []

    vc_target = kwargs.get('vc_target', None)
    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    bs.set_points(s.gamma().reshape((-1, 3)))
    B_field = bs.B().reshape((nphi, ntheta, 3))
    unit_normal = s.unitnormal().reshape((nphi, ntheta, 3))
    BdotN_coils = np.sum(B_field * unit_normal, axis=2)

    if vc_target is not None:
        absBn = np.abs(BdotN_coils - vc_target)
    else:
        absBn = np.abs(BdotN_coils)

    abs_B = bs.AbsB().reshape((nphi, ntheta))
    avg_BdotN_over_B = np.mean(absBn) / np.mean(abs_B) if np.mean(abs_B) > 0 else 0.0
    abs_B_safe = np.where(abs_B > 1e-10, abs_B, 1e-10)
    max_BdotN_overB = np.max(absBn / abs_B_safe) if np.any(abs_B > 0) else 0.0

    coils_linked_to_surface = _check_coils_linked_to_surface(s, base_curves)

    return {
        "B_final": B_final,
        "max_force": max_force,
        "max_torque": max_torque,
        "avg_BdotN_over_B": avg_BdotN_over_B,
        "max_BdotN_overB": max_BdotN_overB,
        "coils_linked_to_surface": coils_linked_to_surface,
        "total_current_final": total_current_final,
    }


def _save_optimized_coils_and_compute_metrics(
    coils: list,
    base_curves: list,
    ncoils: int,
    s: Any,
    s_plot: Any,
    qphi: int,
    qtheta: int,
    bs: Any,
    out_dir: Path,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Save optimized coils and compute final metrics.

    Saves coils to VTK (coils_optimized) and JSON (biot_savart_optimized.json),
    surface with B_N, B_N/|B|, modB to VTK (surface_optimized), and generates
    bn_error_3d_plot.pdf. Computes force/torque (new or legacy API), B_N metrics
    (with optional virtual casing target), and coil-surface linking check.

    Parameters
    ----------
    coils, base_curves : list
        Coil objects and base curves.
    ncoils : int
        Number of base coils.
    s, s_plot : Surface
        Optimization and plotting surfaces.
    qphi, qtheta : int
        Plotting grid dimensions.
    bs : BiotSavart
        BiotSavart with optimized coils.
    out_dir : Path
        Output directory.
    kwargs : Dict[str, Any]
        May contain vc_target, vc_target_plot for virtual casing.

    Returns
    -------
    Dict[str, Any]
        B_final, max_force, max_torque, avg_BdotN_over_B, max_BdotN_overB,
        coils_linked_to_surface, total_current_final.
    """
    from simsopt.field import coils_to_vtk

    try:
        coils_to_vtk(coils, out_dir / "coils_optimized")
    except Exception as e:
        print(f"Warning: Failed to save optimized coils to VTK: {e}")
        print("  Continuing without VTK export...")
    bs.save(out_dir / "biot_savart_optimized.json")

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

    metrics = _compute_optimization_metrics(bs, coils, base_curves, ncoils, s, s_plot, qphi, qtheta, kwargs)

    try:
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

    return metrics


def _build_optimization_results_dict(
    *,
    B_initial: Any,
    B_final: Any,
    target_B: float,
    end_time: float,
    start_time: float,
    iterations_used: int,
    Jf: Any,
    Jcsdist: Any,
    Jccdist: Any,
    Jlink: Any,
    opt_result: Any,
    cached_thresholds: Dict[str, Any],
    base_curves: list,
    coils: list,
    ncoils: int,
    total_current: float,
    total_current_final: float,
    max_force: list,
    max_torque: list,
    avg_BdotN_over_B: float,
    max_BdotN_overB: float,
    coils_linked_to_surface: bool,
    lag_mul: Any,
    out_dir: Path,
    th: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the optimization results dictionary.

    Aggregates flux, geometry, force/torque, B_N metrics, and thresholds
    into a single dict for reporting and continuation caching.

    Parameters
    ----------
    B_initial, B_final, target_B : float or array
        Initial/final |B| and target field.
    end_time, start_time : float
        Wall-clock times.
    iterations_used : int
        Total optimization iterations.
    Jf, Jcsdist, Jccdist : objectives
        Flux and distance objectives for final values.
    opt_result : object or None
        scipy minimize result (success, message, nfev, njev).
    cached_thresholds : Dict[str, Any]
        Thresholds to cache for Fourier continuation.
    base_curves, coils : list
        Base curves and coil objects.
    ncoils : int
        Number of base coils.
    total_current, total_current_final : float
        Current before/after optimization.
    max_force, max_torque : list
        Per-coil max force and torque.
    avg_BdotN_over_B, max_BdotN_overB : float
        B_N/|B| metrics.
    coils_linked_to_surface : bool
        Whether coils encircle plasma.
    lag_mul : Any
        Lagrange multipliers (auglag) or None.
    out_dir : Path
        Output directory.
    th : Dict[str, Any]
        Full thresholds dict for reporting.

    Returns
    -------
    Dict[str, Any]
        Results dict (without post_processing or timing; caller adds those).
    """
    from simsopt.geo import CurveLength, ArclengthVariation

    # Convert to flat floats (handles inhomogeneous shapes from dipole+TF mix)
    max_force_flat = [float(np.asarray(f).max()) for f in max_force]
    max_torque_flat = [float(np.asarray(t).max()) for t in max_torque]

    return {
        'initial_B_field': B_initial,
        'final_B_field': B_final,
        'target_B_field': target_B,
        'optimization_time': end_time - start_time,
        'walltime_sec': end_time - start_time,
        'iterations_used': iterations_used,
        'final_squared_flux': Jf.J(),
        'optimization_success': (
            opt_result.success
            if opt_result is not None and hasattr(opt_result, 'success')
            else True
        ),
        'optimization_message': (
            str(opt_result.message)
            if opt_result is not None and hasattr(opt_result, 'message')
            else 'Completed'
        ),
        'optimization_nfev': (
            getattr(opt_result, 'nfev', None) or iterations_used
            if opt_result is not None
            else iterations_used
        ),
        'optimization_njev': (
            getattr(opt_result, 'njev', None)
            if opt_result is not None
            else None
        ),
        '_cached_thresholds': cached_thresholds,
        'final_min_cs_separation': Jcsdist.shortest_distance(),
        'final_min_cc_separation': Jccdist.shortest_distance(),
        'final_length_per_coil': [float(CurveLength(c).J()) for c in base_curves],
        'final_current_per_coil': [float(abs(coils[i].current.get_value())) for i in range(ncoils)],
        'total_current_before': float(total_current),
        'total_current_after': float(total_current_final),
        'final_total_length': sum(CurveLength(c).J() for c in base_curves),
        'final_max_curvature': max(np.max(c.kappa()) for c in base_curves),
        'final_average_curvature': float(np.mean(np.concatenate([np.atleast_1d(c.kappa()).flatten() for c in base_curves]))),
        'final_arclength_variation': np.mean([ArclengthVariation(c).J() for c in base_curves]),
        'final_mean_squared_curvature': np.max([np.mean(c.kappa() ** 2) for c in base_curves]),
        'final_linking_number': Jlink.J(),
        'coils_linked_to_surface': coils_linked_to_surface,
        'final_max_max_coil_force': float(np.max(max_force_flat)) if max_force_flat else 0.0,
        'final_avg_max_coil_force': float(np.mean(max_force_flat)) if max_force_flat else 0.0,
        'final_max_force_per_coil': max_force_flat,
        'final_max_torque_per_coil': max_torque_flat,
        'final_max_max_coil_torque': float(np.max(max_torque_flat)) if max_torque_flat else 0.0,
        'final_avg_max_coil_torque': float(np.mean(max_torque_flat)) if max_torque_flat else 0.0,
        'avg_BdotN_over_B': avg_BdotN_over_B,
        'max_BdotN_over_B': max_BdotN_overB,
        'lagrange_multipliers': lag_mul,
        'output_directory': str(out_dir),
        'flux_threshold': th.get('flux_threshold'),
        'cc_threshold': th.get('cc_threshold'),
        'cs_threshold': th.get('cs_threshold'),
        'msc_threshold': th.get('msc_threshold'),
        'arclength_variation_threshold': th.get('arclength_variation_threshold'),
        'curvature_threshold': th.get('curvature_threshold'),
        'force_threshold': th.get('force_threshold'),
        'torque_threshold': th.get('torque_threshold'),
    }


def _check_coils_linked_to_surface(s: Any, base_curves: list) -> bool:
    """
    Check that each base coil encircles the plasma.

    A coil is linked if it has points both inside and outside the local
    surface cross-section (R_min, R_max) at each toroidal angle. Uses
    per-phi cross-sections for strongly-shaped stellarators.

    Parameters
    ----------
    s : Surface
        Plasma boundary surface.
    base_curves : list
        Base coil curves to check.

    Returns
    -------
    bool
        True if all coils encircle the plasma, False otherwise.
    """
    surface_gamma = s.gamma()
    R_surface = np.sqrt(surface_gamma[:, :, 0]**2 + surface_gamma[:, :, 1]**2)
    R_min_per_phi = np.min(R_surface, axis=1)
    R_max_per_phi = np.max(R_surface, axis=1)
    phi_surface_slices = np.arctan2(surface_gamma[:, 0, 1], surface_gamma[:, 0, 0])
    for c in base_curves:
        gamma = c.gamma()
        R_coil = np.sqrt(gamma[:, 0]**2 + gamma[:, 1]**2)
        phi_coil = np.arctan2(gamma[:, 1], gamma[:, 0])
        dphi = phi_coil[:, None] - phi_surface_slices[None, :]
        dphi = np.abs(np.arctan2(np.sin(dphi), np.cos(dphi)))
        nearest_phi_idx = np.argmin(dphi, axis=1)
        local_R_min = R_min_per_phi[nearest_phi_idx]
        local_R_max = R_max_per_phi[nearest_phi_idx]
        has_inside = np.any(R_coil < local_R_min)
        has_outside = np.any(R_coil > local_R_max)
        if not (has_inside and has_outside):
            return False
    return True


def evaluate_external_coils(
    coils_json_path: Path,
    surface_file: str,
    surface_range: str = "half period",
    surface_resolution: int = 32,
    plasma_surfaces_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Load coils from JSON and compute leaderboard metrics without running optimization.

    Used to evaluate external coil solutions (e.g. from Zenodo) for leaderboard inclusion.

    Parameters
    ----------
    coils_json_path : Path
        Path to coils.json (simsopt BiotSavart or MagneticFieldSum format).
    surface_file : str
        Plasma surface file (e.g. input.LandremanPaul2021_QA).
    surface_range : str
        Surface range: "half period" or "full torus".
    surface_resolution : int
        Quadrature resolution for surface evaluation.
    plasma_surfaces_dir : Path | None
        Directory containing plasma surface files. Defaults to plasma_surfaces/.

    Returns
    -------
    Dict[str, Any]
        Metrics dict suitable for results.json (metrics, score_primary, etc.).
    """
    from simsopt import load
    from simsopt.objectives import SquaredFlux
    from simsopt.geo import (
        CurveCurveDistance,
        CurveSurfaceDistance,
        LinkingNumber,
    )
    import json
    from .post_processing import _get_coils_from_bfield

    plasma_surfaces_dir = plasma_surfaces_dir or Path("plasma_surfaces")
    surface_path = plasma_surfaces_dir / surface_file

    # Load coils - strip version-incompatible keys (simsopt auglag_coils vs main)
    coils_data = json.loads(coils_json_path.read_text())
    for obj in coils_data.get("simsopt_objs", {}).values():
        if isinstance(obj, dict):
            if obj.get("@class") == "CurvePlanarFourier":
                obj.pop("nfp", None)
                obj.pop("stellsym", None)
            if obj.get("@class") == "Coil":
                obj.pop("regularization", None)  # auglag_coils branch only
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(coils_data, tf, indent=2)
        tmp_coils_path = Path(tf.name)
    try:
        bfield = load(str(tmp_coils_path))
    finally:
        tmp_coils_path.unlink(missing_ok=True)

    if not surface_path.exists():
        surface_path = Path(surface_file)
    if not surface_path.exists():
        raise FileNotFoundError(f"Surface not found: {surface_file}")

    surface_lower = str(surface_path).lower()
    if "input" in surface_lower:
        s = SurfaceRZFourier.from_vmec_input(
            str(surface_path), range=surface_range, nphi=surface_resolution, ntheta=surface_resolution
        )
    elif "wout" in surface_lower:
        s = SurfaceRZFourier.from_wout(
            str(surface_path), range=surface_range, nphi=surface_resolution, ntheta=surface_resolution
        )
    elif "focus" in surface_lower:
        s = SurfaceRZFourier.from_focus(
            str(surface_path), range=surface_range, nphi=surface_resolution, ntheta=surface_resolution
        )
    else:
        raise ValueError(f"Unknown surface type: {surface_path}")

    coils = _get_coils_from_bfield(bfield)
    if not coils:
        from simsopt.field import BiotSavart
        if isinstance(bfield, BiotSavart):
            coils = list(bfield.coils)
        else:
            raise ValueError("Could not extract coils from loaded object")
    from simsopt.field import BiotSavart
    bs = BiotSavart(coils) if not isinstance(bfield, BiotSavart) else bfield

    nfp = s.nfp
    stellsym = s.stellsym
    symmetry_factor = nfp * (2 if stellsym else 1)
    ncoils = max(1, len(coils) // symmetry_factor)
    step = max(1, len(coils) // ncoils)
    base_coil_indices = list(range(0, len(coils), step))[:ncoils]
    base_curves = [coils[i].curve for i in base_coil_indices]
    base_coils = [coils[i] for i in base_coil_indices]
    curves = [c.curve for c in coils]

    target_B = 1.0
    if "LandremanPaul2021_QA" in surface_file:
        target_B = 1.0
    elif "LandremanPaul2021_QH" in surface_file:
        target_B = 5.7
    elif "muse" in surface_file.lower():
        target_B = 0.15
    else:
        target_B = 5.7

    th = _compute_thresholds_from_surface(s, {})
    flux_threshold = th.get("flux_threshold", 1e-8)
    cc_threshold = th.get("cc_threshold", 0.1)
    cs_threshold = th.get("cs_threshold", 0.1)

    Jf = SquaredFlux(s, bs, threshold=flux_threshold)
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, cs_threshold)
    Jlink = LinkingNumber(curves, downsample=2)

    s_plot = SurfaceRZFourier.from_vmec_input(
        str(surface_path), range=surface_range, nphi=64, ntheta=64
    ) if "input" in surface_lower else s
    if "wout" in surface_lower:
        s_plot = SurfaceRZFourier.from_wout(str(surface_path), range=surface_range, nphi=64, ntheta=64)
    elif "focus" in surface_lower:
        s_plot = SurfaceRZFourier.from_focus(str(surface_path), range=surface_range, nphi=64, ntheta=64)

    opt_metrics = _compute_optimization_metrics(
        bs, coils, base_curves, ncoils, s, s_plot, 64, 64, {}
    )
    coil_metrics = _compute_coil_subset_metrics(
        base_coils, base_curves, coils, s, {}
    )

    try:
        total_current_final = sum(c.current.get_value() for c in coils[:ncoils])
    except (AttributeError, TypeError):
        total_current_final = sum(
            float(abs(coils[i].current.get_value()))
            for i in range(min(ncoils, len(coils)))
            if hasattr(coils[i], "current")
        )

    coil_order = int(base_curves[0].order) if base_curves and hasattr(base_curves[0], "order") else 16

    metrics = {
        "final_squared_flux": float(Jf.J()),
        "score_primary": float(Jf.J()),
        "final_min_cc_separation": float(Jccdist.shortest_distance()),
        "final_min_cs_separation": float(Jcsdist.shortest_distance()),
        "final_linking_number": float(Jlink.J()),
        "coils_linked_to_surface": opt_metrics["coils_linked_to_surface"],
        "avg_BdotN_over_B": float(opt_metrics["avg_BdotN_over_B"]),
        "max_BdotN_over_B": float(opt_metrics["max_BdotN_overB"]),
        "final_total_length": float(coil_metrics["final_total_length"]),
        "final_arclength_variation": float(coil_metrics["final_arclength_variation"]),
        "final_mean_squared_curvature": float(coil_metrics["final_mean_squared_curvature"]),
        "final_max_curvature": float(np.max([np.max(c.kappa()) for c in base_curves])),
        "num_coils": ncoils,
        "coil_order": coil_order,
        "target_B_field": target_B,
        "total_current_after": float(total_current_final),
        "optimization_time": 0.0,
        "iterations_used": 0,
        "final_length_per_coil": [float(x) for x in coil_metrics["final_length_per_coil"]],
        "final_current_per_coil": [float(x) for x in coil_metrics["final_current_per_coil"]],
        "_cached_thresholds": {
            "a0": th.get("a0"),
            "major_radius": th.get("major_radius"),
            "minor_radius": th.get("minor_radius"),
        },
    }
    if opt_metrics.get("max_force"):
        metrics["final_max_max_coil_force"] = float(np.max(opt_metrics["max_force"]))
        metrics["final_max_force_per_coil"] = [float(f) for f in opt_metrics["max_force"]]
    if opt_metrics.get("max_torque"):
        metrics["final_max_max_coil_torque"] = float(np.max(opt_metrics["max_torque"]))
        metrics["final_max_torque_per_coil"] = [float(t) for t in opt_metrics["max_torque"]]
    return metrics


def evaluate_external_dipole_coils(
    coils_json_path: Path,
    surface_file: str,
    ncoils_dipole: int,
    surface_range: str = "half period",
    surface_resolution: int = 32,
    plasma_surfaces_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Load dipole+TF coils from JSON and compute dipole metrics for leaderboard.

    Used for dipole optimization results (e.g. Zenodo 14934092, Kaptanoglu dipole).
    Coils are split: first ncoils_dipole are dipole, rest are TF.

    Parameters
    ----------
    coils_json_path : Path
        Path to coils.json (simsopt BiotSavart or MagneticFieldSum format).
    surface_file : str
        Plasma surface file (e.g. input.LandremanPaul2021_QA).
    ncoils_dipole : int
        Number of dipole coils (first ncoils_dipole in the coil list).
    surface_range : str
        Surface range: "half period" or "full torus".
    surface_resolution : int
        Quadrature resolution for surface evaluation.
    plasma_surfaces_dir : Path | None
        Directory containing plasma surface files. Defaults to plasma_surfaces/.

    Returns
    -------
    Dict[str, Any]
        Metrics dict with dipole_metrics, tf_metrics, flux, B·n, etc.
    """
    from simsopt import load
    from simsopt.objectives import SquaredFlux
    from simsopt.geo import (
        CurveCurveDistance,
        CurveSurfaceDistance,
        LinkingNumber,
    )
    import json
    from .post_processing import _get_coils_from_bfield

    plasma_surfaces_dir = plasma_surfaces_dir or Path("plasma_surfaces")
    surface_path = plasma_surfaces_dir / surface_file
    if not surface_path.exists():
        surface_path = Path(surface_file)
    if not surface_path.exists():
        raise FileNotFoundError(f"Surface not found: {surface_file}")

    coils_data = json.loads(coils_json_path.read_text())
    for obj in coils_data.get("simsopt_objs", {}).values():
        if isinstance(obj, dict):
            if obj.get("@class") == "CurvePlanarFourier":
                obj.pop("nfp", None)
                obj.pop("stellsym", None)
            if obj.get("@class") == "Coil":
                obj.pop("regularization", None)  # auglag_coils branch only
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(coils_data, tf, indent=2)
        tmp_coils_path = Path(tf.name)
    try:
        bfield = load(str(tmp_coils_path))
    finally:
        tmp_coils_path.unlink(missing_ok=True)

    coils = _get_coils_from_bfield(bfield)
    if not coils:
        from simsopt.field import BiotSavart
        if isinstance(bfield, BiotSavart):
            coils = list(bfield.coils)
        else:
            raise ValueError("Could not extract coils from loaded object")
    from simsopt.field import BiotSavart
    bs = BiotSavart(coils) if not isinstance(bfield, BiotSavart) else bfield

    surface_lower = str(surface_path).lower()
    if "input" in surface_lower:
        s = SurfaceRZFourier.from_vmec_input(
            str(surface_path), range=surface_range, nphi=surface_resolution, ntheta=surface_resolution
        )
    elif "wout" in surface_lower:
        s = SurfaceRZFourier.from_wout(
            str(surface_path), range=surface_range, nphi=surface_resolution, ntheta=surface_resolution
        )
    elif "focus" in surface_lower:
        s = SurfaceRZFourier.from_focus(
            str(surface_path), range=surface_range, nphi=surface_resolution, ntheta=surface_resolution
        )
    else:
        raise ValueError(f"Unknown surface type: {surface_path}")

    ncoils_dipole = min(ncoils_dipole, len(coils))
    dipole_coils = coils[:ncoils_dipole]
    tf_coils = coils[ncoils_dipole:]
    base_curves_dipole = [c.curve for c in dipole_coils]
    base_curves_tf = [c.curve for c in tf_coils]
    curves = [c.curve for c in coils]
    ncoils = len(coils)

    target_B = 1.0
    if "LandremanPaul2021_QA" in surface_file:
        target_B = 1.0
    elif "LandremanPaul2021_QH" in surface_file:
        target_B = 5.7
    elif "muse" in surface_file.lower():
        target_B = 0.15
    else:
        target_B = 5.7

    th = _compute_thresholds_from_surface(s, {})
    flux_threshold = th.get("flux_threshold", 1e-8)
    cc_threshold = th.get("cc_threshold", 0.1)
    cs_threshold = th.get("cs_threshold", 0.1)

    Jf = SquaredFlux(s, bs, threshold=flux_threshold)
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, cs_threshold)
    Jlink = LinkingNumber(curves, downsample=2)

    s_plot = SurfaceRZFourier.from_vmec_input(
        str(surface_path), range=surface_range, nphi=64, ntheta=64
    ) if "input" in surface_lower else s
    if "wout" in surface_lower:
        s_plot = SurfaceRZFourier.from_wout(str(surface_path), range=surface_range, nphi=64, ntheta=64)
    elif "focus" in surface_lower:
        s_plot = SurfaceRZFourier.from_focus(str(surface_path), range=surface_range, nphi=64, ntheta=64)

    base_curves = [c.curve for c in coils]
    opt_metrics = _compute_optimization_metrics(
        bs, coils, base_curves, ncoils, s, s_plot, 64, 64, {}
    )
    dipole_metrics = _compute_coil_subset_metrics(
        dipole_coils, base_curves_dipole, coils, s, {}
    )
    tf_metrics = _compute_coil_subset_metrics(
        tf_coils, base_curves_tf, coils, s, {}
    )
    coil_metrics = _compute_coil_subset_metrics(
        coils, base_curves, coils, s, {}
    )

    try:
        total_current_final = sum(c.current.get_value() for c in coils)
    except (AttributeError, TypeError):
        total_current_final = sum(
            float(abs(c.current.get_value()))
            for c in coils
            if hasattr(c, "current")
        )

    coil_order = int(base_curves[0].order) if base_curves and hasattr(base_curves[0], "order") else 16

    metrics = {
        "final_squared_flux": float(Jf.J()),
        "score_primary": float(Jf.J()),
        "final_min_cc_separation": float(Jccdist.shortest_distance()),
        "final_min_cs_separation": float(Jcsdist.shortest_distance()),
        "final_linking_number": float(Jlink.J()),
        "coils_linked_to_surface": opt_metrics["coils_linked_to_surface"],
        "avg_BdotN_over_B": float(opt_metrics["avg_BdotN_over_B"]),
        "max_BdotN_over_B": float(opt_metrics["max_BdotN_overB"]),
        "final_total_length": float(coil_metrics["final_total_length"]),
        "final_arclength_variation": float(coil_metrics["final_arclength_variation"]),
        "final_mean_squared_curvature": float(coil_metrics["final_mean_squared_curvature"]),
        "final_max_curvature": float(np.max([np.max(c.kappa()) for c in base_curves])) if base_curves else 0.0,
        "num_coils": ncoils,
        "coil_order": coil_order,
        "target_B_field": target_B,
        "total_current_after": float(total_current_final),
        "optimization_time": 0.0,
        "iterations_used": 0,
        "dipole_metrics": dipole_metrics,
        "tf_metrics": tf_metrics,
        "_cached_thresholds": {
            "a0": th.get("a0"),
            "major_radius": th.get("major_radius"),
            "minor_radius": th.get("minor_radius"),
        },
    }
    if opt_metrics.get("max_force"):
        metrics["final_max_max_coil_force"] = float(np.max(opt_metrics["max_force"]))
        metrics["final_max_force_per_coil"] = [float(f) for f in opt_metrics["max_force"]]
    if opt_metrics.get("max_torque"):
        metrics["final_max_max_coil_torque"] = float(np.max(opt_metrics["max_torque"]))
        metrics["final_max_torque_per_coil"] = [float(t) for t in opt_metrics["max_torque"]]
    return metrics


def _format_verbose_iteration_output(
    iteration: int,
    Jls: list,
    Jccdist: Any,
    Jcsdist: Any,
    base_curves: list,
    Jlink: Any,
    Jforce: Any,
    Jtorque: Any,
    grad: np.ndarray,
    weights: list,
    c_list: list,
    constraint_names_and_thresholds: list,
    J_total: float,
) -> tuple[str, str]:
    """
    Format verbose iteration output for scipy minimize callback.

    Builds two lines: (1) main line with iteration, L, d_cc, d_cs, κ, MSC,
    LN, F, τ, ‖∇J‖; (2) contrib line with weighted objective contributions
    per term and total.

    Parameters
    ----------
    iteration : int
        Current iteration number.
    Jls, Jccdist, Jcsdist, Jlink, Jforce, Jtorque : objectives
        Constraint objectives for value extraction.
    base_curves : list
        Base coil curves (for κ, MSC).
    grad : np.ndarray
        Gradient vector for ‖∇J‖.
    weights : list
        Per-constraint weights.
    c_list : list
        Constraint objectives.
    constraint_names_and_thresholds : list
        (name, threshold) pairs for contrib labels.
    J_total : float
        Total weighted objective value.

    Returns
    -------
    tuple[str, str]
        (main_line, contrib_line) - formatted strings for printing.
    """
    from simsopt.geo import MeanSquaredCurvature

    outstr = f"[{iteration}]"
    outstr += f" L={sum(J.J() for J in Jls):.2f}"
    outstr += f", d_cc={Jccdist.shortest_distance():.2f}, d_cs={Jcsdist.shortest_distance():.2f}"
    kappa_values = [c.kappa().max() for c in base_curves]
    msc_values = [MeanSquaredCurvature(c).J() for c in base_curves]
    kappa_str = ",".join([f"{k:.1f}" for k in kappa_values])
    msc_str = ",".join([f"{m:.1f}" for m in msc_values])
    outstr += f", κ=[{kappa_str}]"
    outstr += f", MSC=[{msc_str}]"
    outstr += f", LN={int(round(Jlink.J()))}"
    outstr += f", F={Jforce.J():.2e}"
    outstr += f", τ={Jtorque.J():.2e}"
    outstr += f", ‖∇J‖={np.linalg.norm(grad):.1e}"

    name_short = {"Flux": "J_f", "CC Distance": "d_cc", "CS Distance": "d_cs",
                  "Length": "L", "MSC": "MSC", "Arclength Var": "Var",
                  "κ": "κ", "Link #": "LN", "Force": "F", "Torque": "τ"}
    contrib_parts = []
    flux_contrib = weights[0] * c_list[0].J()
    contrib_parts.append(f"{name_short.get('Flux', 'Flux')}={flux_contrib:.1e}")
    for idx, (name, _) in enumerate(constraint_names_and_thresholds, start=1):
        if idx < len(c_list) and idx < len(weights):
            constraint_contrib = weights[idx] * c_list[idx].J()
            short = name_short.get(name, name)
            contrib_parts.append(f"{short}={constraint_contrib:.1e}")
    contrib_str = "Objs: " + ", ".join(contrib_parts)
    contrib_str += f", Total={J_total:.1e}"

    return outstr, contrib_str


def _run_augmented_lagrangian(
    c_list: list,
    max_iterations: int,
    max_iter_subopt: int,
    verbose: bool,
    kwargs: Dict[str, Any],
) -> None:
    """
    Run simsopt augmented Lagrangian optimization.

    Treats all c_list entries as equality constraints. Modifies objectives
    in-place via simsopt's augmented_lagrangian_method. Supports mu_init,
    tau, minimize_method from kwargs.

    Parameters
    ----------
    c_list : list
        Constraint objectives (equality constraints).
    max_iterations : int
        Maximum outer iterations.
    max_iter_subopt : int
        Maximum inner (L-BFGS-B) iterations per outer step.
    verbose : bool
        Print progress.
    kwargs : Dict[str, Any]
        Optional mu_init, tau, minimize_method for augmented Lagrangian.
    """
    try:
        from simsopt.solve import augmented_lagrangian_method
    except ImportError:
        from simsopt.solve.augmented_lagrangian import augmented_lagrangian_method
    import inspect
    _alm_sig = inspect.signature(augmented_lagrangian_method)
    _alm_params = set(_alm_sig.parameters.keys())
    opts = {"MAXITER": max_iterations, "MAXITER_lag": max_iter_subopt, "verbose": verbose}
    if "mu_init" in kwargs:
        opts["mu_init"] = kwargs["mu_init"]
    if "tau" in kwargs:
        opts["tau"] = kwargs["tau"]
    if "minimize_method" in kwargs:
        opts["minimize_method"] = kwargs["minimize_method"]
    opts = {k: v for k, v in opts.items() if k in _alm_params}
    augmented_lagrangian_method(f=None, equality_constraints=c_list, **opts)


def _run_scipy_minimize_for_modular_coils(
    c_list: list,
    constraint_scaling: Dict[int, float],
    constraint_idx_to_term: Dict[int, str],
    cc_distance_index: int | None,
    cs_distance_index: int | None,
    constraint_names_and_thresholds: list,
    base_curves: list,
    Jls: list,
    Jccdist: Any,
    Jcsdist: Any,
    Jlink: Any,
    Jforce: Any,
    Jtorque: Any,
    coil_objective_terms: Dict[str, Any] | None,
    algorithm: str,
    max_iterations: int,
    algorithm_options: Dict[str, Any],
    verbose: bool,
    kwargs: Dict[str, Any],
) -> tuple:
    """
    Run scipy minimize (BFGS, L-BFGS-B, SLSQP, etc.) for modular coil optimization.

    Builds weighted objective JF from c_list and weights, defines objective/gradient
    with verbose iteration output, runs Taylor test, then minimizes. Returns the
    scipy result and iteration count.

    Parameters
    ----------
    c_list : list
        Constraint objectives (flux first, then distance, length, etc.).
    constraint_scaling, constraint_idx_to_term : dict
        Scaling and term mapping for weight building.
    cc_distance_index, cs_distance_index : int | None
        Indices of coil-coil and coil-surface distance constraints.
    constraint_names_and_thresholds : list
        (name, threshold) pairs for verbose output.
    base_curves, Jls, Jccdist, Jcsdist, Jlink, Jforce, Jtorque : objectives
        Constraint objectives for verbose output.
    coil_objective_terms : Dict | None
        Case config for weight overrides.
    algorithm : str
        Scipy algorithm name (e.g. L-BFGS-B, BFGS, SLSQP).
    max_iterations : int
        Maximum iterations.
    algorithm_options : Dict
        User-provided options for scipy.
    verbose : bool
        Print iteration progress.
    kwargs : Dict
        constraint_weight_{i}, flux_weight, etc.

    Returns
    -------
    tuple
        (result, iterations_used) - scipy OptimizeResult and nit.
    """
    from scipy.optimize import minimize
    from simsopt.objectives import Weight

    weights = _build_weights_for_scipy_minimize(
        c_list, constraint_scaling, constraint_idx_to_term,
        cc_distance_index, cs_distance_index, kwargs, coil_objective_terms,
    )
    JF = sum([Weight(w) * c for c, w in zip(c_list, weights)])

    iteration_count = [0]

    def objective(x: np.ndarray) -> float:
        JF.x = x  # type: ignore[attr-defined]
        J = JF.J()  # type: ignore[attr-defined]
        iteration_count[0] += 1
        if verbose and (iteration_count[0] == 1 or iteration_count[0] % 100 == 0):
            grad = JF.dJ()  # type: ignore[attr-defined]
            main_line, contrib_line = _format_verbose_iteration_output(
                iteration_count[0], Jls, Jccdist, Jcsdist, base_curves,
                Jlink, Jforce, Jtorque, grad, weights, c_list,
                constraint_names_and_thresholds, J,
            )
            print(main_line)
            print(contrib_line)
        return J

    def gradient(x: np.ndarray) -> np.ndarray:
        JF.x = x  # type: ignore[attr-defined]
        return JF.dJ()  # type: ignore[attr-defined]

    x0 = JF.x.copy()  # type: ignore[attr-defined]
    _run_taylor_test(objective, gradient, x0, verbose=verbose)
    JF.x = x0  # type: ignore[attr-defined, assignment]

    options = _build_scipy_minimize_options(algorithm, max_iterations, algorithm_options)
    result = minimize(
        fun=objective,
        x0=JF.x,  # type: ignore[attr-defined]
        method=algorithm,
        jac=gradient,
        options=options,
    )
    iterations_used = getattr(result, 'nit', 0)
    return result, iterations_used


class LinearPenalty:
    """
    Linear penalty function that implements max(objective - threshold, 0).

    Wraps a simsopt objective so that the effective value is zero below the
    threshold and (J - threshold) above. Used for l1_threshold options in
    coil_objective_terms (length, curvature, distances, etc.).

    Attributes
    ----------
    objective : simsopt objective
        Underlying objective (must have J() and dJ() methods).
    threshold : float
        Value below which the penalty is zero.
    """

    def __init__(self, objective: Any, threshold: float) -> None:
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


def _get_scipy_algorithm_options(algorithm: str) -> Dict[str, List[type]]:
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

    .. deprecated::
        Use :func:`stellcoilbench.evaluate.load_case_config` and
        :class:`stellcoilbench.config_scheme.CaseConfig` instead for case configs
        with validation.

    Parameters
    ----------
    config_path : Path
        Path to the coils.yaml file.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the loaded coils configuration.

    Raises
    ------
    ValueError
        If the config file is not a dictionary.
    """
    import warnings

    import yaml

    warnings.warn(
        "load_coils_config is deprecated. Use load_case_config and CaseConfig for case configs.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    # For modular coils, merge case coil_objective_terms with defaults (case overrides).
    # Dipole coils use their own default when case has none.
    if coil_params.get("coil_type") == "dipole":
        coil_objective_terms = case_cfg.coil_objective_terms
    else:
        coil_objective_terms = dict(DEFAULT_COIL_OBJECTIVE_TERMS)
        if case_cfg.coil_objective_terms:
            coil_objective_terms.update(case_cfg.coil_objective_terms)

    # Extract threshold values from coil_objective_terms if present
    # These will be passed as kwargs to optimize_coils_loop
    threshold_kwargs = {}
    if coil_objective_terms:
        threshold_keys = [
            "length_threshold",
            "length_threshold_dipole",
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
        range=surface_params.get("range", "half period"),
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
    # Set filename and range for dipole mode (derived from s in optimize_coils_loop)
    try:
        surface.filename = surface_file
        surface.range = surface_params.get("range", "half period")
    except (AttributeError, TypeError):
        pass

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
        
        if is_proc0():
            fourier_continuation = case_cfg.fourier_continuation
            coil_type = coil_params.get("coil_type", "modular")
            if coil_type == "dipole":
                fourier_continuation_dipole = fourier_continuation and fourier_continuation.get('enabled', False)
                fourier_orders_dipole = fourier_continuation.get('orders', [coil_params.get('order', 16)]) if fourier_continuation else []
                if fourier_continuation_dipole and fourier_orders_dipole and isinstance(fourier_orders_dipole, list) and all(isinstance(o, int) for o in fourier_orders_dipole):
                    coil_kw = {k: v for k, v in coil_params.items() if k not in ("coil_type", "ncoils", "order", "target_B")}
                    opt_kw = {k: v for k, v in optimizer_params.items() if k not in ("max_iterations", "verbose")}
                    coils, results_dict = optimize_coils_with_fourier_continuation_dipole(
                        surface,
                        fourier_orders=fourier_orders_dipole,
                        target_B=coil_params.get('target_B', 5.7),
                        out_dir=str(output_dir),
                        max_iterations=optimizer_params.get("max_iterations", 100),
                        ncoils=coil_params.get('ncoils', 4),
                        verbose=optimizer_params.get("verbose", True),
                        regularization=regularization_circ if regularization_circ is not None else lambda x: None,
                        coil_objective_terms=coil_objective_terms,
                        surface_resolution=surface_resolution,
                        case_path=case_yaml_path_abs if case_yaml_path_abs and case_yaml_path_abs.exists() else case_path,
                        skip_post_processing=skip_post_processing_in_loop,
                        run_vmec=run_vmec,
                        run_simple=run_simple,
                        plot_poincare=plot_poincare,
                        plot_boozer=pp_params.get("plot_boozer", True),
                        **coil_kw,
                        **opt_kw,
                    )
                else:
                    coils, results_dict = optimize_coils_loop(
                        surface,
                        dipole_array=True,
                        out_dir=str(output_dir),
                        max_iterations=optimizer_params.get("max_iterations", 100),
                        verbose=optimizer_params.get("verbose", True),
                        skip_post_processing=skip_post_processing_in_loop,
                        case_path=case_yaml_path_abs if case_yaml_path_abs and case_yaml_path_abs.exists() else case_path,
                        run_vmec=run_vmec,
                        run_simple=run_simple,
                        plot_poincare=plot_poincare,
                        plot_boozer=pp_params.get("plot_boozer", True),
                        **{k: v for k, v in coil_params.items() if k not in ("coil_type", "ncoils", "order")},
                        **{k: v for k, v in optimizer_params.items() if k not in ("max_iterations", "verbose")},
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
                    verbose=optimizer_params.get('verbose', True),
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
    if is_proc0():
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
                if is_proc0() and case_yaml_path_abs and case_yaml_path_abs.exists():
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
                if not is_proc0() and case_yaml_path_abs and case_yaml_path_abs.exists():
                    import yaml
                    try:
                        case_data = yaml.safe_load(case_yaml_path_abs.read_text())
                        surface_name = case_data.get("surface_params", {}).get("surface", "").lower()
                        if "qh" in surface_name or "qash" in surface_name:
                            helicity_n = -1
                    except Exception:
                        pass
                
                # Find plasma_surfaces_dir (all processes need this)
                plasma_surfaces_dir = find_plasma_surfaces_dir(Path(output_dir))

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
                if is_proc0() and 'quasisymmetry_average' in post_processing_results:
                    proc0_print(f"  Average quasisymmetry error: {post_processing_results['quasisymmetry_average']:.2e}")
                
                # Add post-processing results to results_dict (only rank 0 returns this)
                if is_proc0():
                    results_dict['post_processing'] = post_processing_results
            else:
                proc0_print(f"Warning: Skipping post-processing (coils_json not found: {coils_json_path})")
        except Exception as e:
            proc0_print(f"Warning: Post-processing failed: {e}")
            if is_proc0():
                import traceback
                traceback.print_exc()
    
    return results_dict


def initialize_coils_loop(
    s: SurfaceRZFourier,
    out_dir: Path | str = "",
    target_B: float = 5.7,
    ncoils: int = 4,
    order: int = 16,
    coil_width: float = 0.4,
    regularization: Callable[..., Any] | None = regularization_circ,
) -> List[Any]:
    """
    Initialize modular coils with adaptive R0/R1 and target B-field scaling.

    Uses an adaptive strategy to determine R0 and R1 so that coils:
    - Do not intersect the plasma surface
    - Interlink the plasma (go around it)
    - Maintain safe coil-surface and coil-coil distances
    - Do not interlink each other (linking number ≈ 0)

    Iteratively adjusts R0/R1 until constraints are satisfied, then adjusts
    total current until the field along the major radius averages to target_B.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    out_dir : Path | str, optional
        Output directory for saved files.
    target_B : float, default=5.7
        Target magnetic field strength [T] on-axis.
    ncoils : int, default=4
        Number of base coils.
    order : int, default=16
        Fourier order for coil curves.
    coil_width : float, default=0.4
        Coil width [m] for regularization.
    regularization : callable, optional
        Regularization function (default: regularization_circ).

    Returns
    -------
    list
        List of simsopt Coil objects (including symmetric copies).
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
            R0=R0, R1=R1, order=order, numquadpoints=200)
        
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


def _surface_file_and_params_from_s(s: SurfaceRZFourier) -> tuple[str, Dict[str, Any]]:
    """
    Extract surface_file and surface_params from a SurfaceRZFourier.

    Uses s.filename for surface_file. For surface_params, uses s.range if
    available, otherwise defaults to "half period".

    Raises
    ------
    ValueError
        If s has no filename attribute or it is None/empty.
    """
    surface_file = getattr(s, "filename", None)
    if not surface_file:
        raise ValueError(
            "Dipole mode requires surface s to have a filename attribute "
            "(e.g. from SurfaceRZFourier.from_vmec_input or from_focus)"
        )
    surface_file = str(surface_file)
    range_param = getattr(s, "range", None) or "half period"
    surface_params = {"range": range_param}
    return surface_file, surface_params


def initialize_coils_dipole(
    s: SurfaceRZFourier,
    surface_file: str,
    surface_params: Dict[str, Any],
    poff: float = 1.5,
    coff: float = 3.0,
    Nx: int = 4,
    Ny: int | None = None,
    Nz: int | None = None,
    dipole_order: int = 2,
    dipole_coil_size: float = 0.1,
    remove_inboard_eps: float = -0.4,
    out_dir: Path | str = "",
    base_coils_TF: list | None = None,
    ncoils_TF: int | None = None,
) -> tuple:
    """
    Initialize dipole coils plus TF coils, modeled after dipole_array_tutorial_advanced.py.

    Creates planar dipole coils between two toroidal surfaces (inner and outer, extended
    from the plasma boundary), removes inboard dipoles, removes interlinking dipoles,
    and aligns dipole normals with the plasma surface.

    TF coils must be predefined (e.g. from initialize_coils_loop) and passed via
    base_coils_TF and ncoils_TF. They are used for remove_interlinking_dipoles_and_TFs.

    Args:
        s: Plasma boundary surface.
        surface_file: Path to surface file (for creating s_inner/s_outer).
        surface_params: Dict with 'range' and other surface params.
        poff: Inner surface extension distance [m] (reactor-scale; divided by a0).
        coff: Additional outer extension beyond inner [m] (reactor-scale; divided by a0).
        Nx, Ny, Nz: Grid dimensions for dipole placement (Ny, Nz default to Nx).
        dipole_order: Fourier order for planar dipole curves.
        dipole_coil_size: Dipole wire cross-section [m] (e.g. 0.1 for 10 cm).
        remove_inboard_eps: Eps for remove_inboard_dipoles.
        out_dir: Output directory for saved files.
        base_coils_TF: List of coil objects from initialize_coils_loop (required).
        ncoils_TF: Number of base TF coils (required).

    Returns:
        Tuple (coils, base_curves_dipole, base_curves_TF, ncoils_dipole_expanded, num_TF_unique)
        where coils = dipole_coils + TF_coils (all coils for BiotSavart).
        ncoils_dipole_expanded = len(dipole_coils) after symmetrization (for slicing coils).
    """
    from simsopt.geo import SurfaceRZFourier, create_planar_curves_between_two_toroidal_surfaces
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.util import (
        remove_inboard_dipoles,
        remove_interlinking_dipoles_and_TFs,
        align_dipoles_with_plasma,
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
    minor_radius = float(s.minor_radius())
    a0 = ARIES_CS_MINOR_RADIUS / minor_radius
    poff_scaled = poff / a0
    coff_scaled = coff / a0
    s_inner.extend_via_normal(poff_scaled)
    s_outer.extend_via_normal(poff_scaled + coff_scaled)

    if base_coils_TF is None or ncoils_TF is None:
        raise ValueError("base_coils_TF and ncoils_TF are required; TF coils must be predefined (e.g. from initialize_coils_loop)")
    base_curves_TF = [c.curve for c in base_coils_TF[:ncoils_TF]]

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
    coils = coils_dipole + base_coils_TF

    ncoils_dipole_expanded = len(coils_dipole)
    return coils, base_curves, base_curves_TF, ncoils_dipole_expanded, ncoils_TF


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
    Generate a 3D plot showing B_N/|B| error on the plasma surface.

    Renders the plasma surface colored by |B_N/|B|| (or B_N error vs vc_target
    if virtual casing is used). Coils are overlaid. Requires matplotlib.

    Parameters
    ----------
    surface : SurfaceRZFourier
        Plasma surface for plotting (should be full torus).
    bs : BiotSavart
        BiotSavart object containing the magnetic field from coils.
    coils : list
        List of coil objects to plot.
    out_dir : Path
        Directory where the PDF will be saved.
    filename : str, optional
        Output filename (default: bn_error_3d_plot.pdf).
    title : str, optional
        Plot title.
    plot_upsample : int, optional
        Factor to upsample surface quadrature for smoother plot (default: 2).
    vc_target : np.ndarray | None, optional
        Virtual casing target B_N; if provided, error = |B_N - vc_target|.
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
        R0=R0, R1=R1, order=new_order, numquadpoints=200
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
    skip_post_processing: bool
        If True, skip post-processing after optimization (default: False).
    run_vmec, run_simple, plot_poincare, plot_boozer: bool
        Post-processing flags (used only on final order).
    plot_finite_build: bool
        Generate finite-build coil VTK (default: False).
    finite_build_width, finite_build_height: float | None
        Cross-section dimensions for finite-build coils.
    **kwargs
        Same as optimize_coils_loop (thresholds, algorithm, plot_upsample_factor, etc.).

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
                surface_stem = surface_stem_from_filename(s.filename)
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
                plasma_surfaces_dir = find_plasma_surfaces_dir(out_dir_path)

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


def optimize_coils_with_fourier_continuation_dipole(
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
    **kwargs
) -> tuple[list, Dict[str, Any]]:
    """
    Perform dipole coil optimization with Fourier continuation on TF coils only.

    Iterates over fourier_orders for the TF coil Fourier order; dipole_order stays fixed.
    Dipole coils are unchanged during continuation; only TF coils are extended.
    """
    if not fourier_orders:
        raise ValueError("fourier_orders must be a non-empty list")
    if not all(isinstance(o, int) and o > 0 for o in fourier_orders):
        raise ValueError("All fourier_orders must be positive integers")
    if fourier_orders != sorted(fourier_orders):
        raise ValueError("fourier_orders must be in ascending order")

    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    coil_width = kwargs.get('coil_width', 0.4)
    cached_thresholds: Dict[str, Any] = {}

    coils: list | None = None
    ncoils_dipole = 0
    base_curves_dipole: list = []
    base_curves_TF: list = []
    all_results: list = []
    results: Dict[str, Any] = {}

    print(f"Starting dipole Fourier continuation (TF order only) with orders: {fourier_orders}")

    for i, order in enumerate(fourier_orders):
        print(f"\n{'='*60}")
        print(f"Dipole Fourier continuation step {i+1}/{len(fourier_orders)}: TF order={order}")
        print(f"{'='*60}")

        order_dir = out_dir_path / f"order_{order}"
        order_dir.mkdir(exist_ok=True)

        if i == 0:
            coils, results = optimize_coils_loop(
                s=s,
                dipole_array=True,
                target_B=target_B,
                out_dir=str(order_dir),
                max_iterations=max_iterations,
                ncoils=ncoils,
                order=order,
                verbose=verbose,
                regularization=regularization,
                coil_objective_terms=coil_objective_terms,
                surface_resolution=surface_resolution,
                skip_post_processing=True,
                **kwargs
            )
            cached_thresholds = results.get('_cached_thresholds', {})
            if coils:
                # Split by quadpoints: dipole (CurvePlanarFourier) ~40 qp; TF (CurveXYZFourier) 200 qp.
                # s.nfp/stellsym can give wrong ntoroidal for half-period surfaces.
                nqp0 = len(coils[0].curve.quadpoints)
                split_idx = len(coils)
                for j in range(1, len(coils)):
                    if len(coils[j].curve.quadpoints) != nqp0:
                        split_idx = j
                        break
                dipole_coils = coils[:split_idx]
                tf_coils = coils[split_idx:]
                ncoils_dipole = split_idx
                ntoroidal = len(tf_coils) // ncoils if ncoils and tf_coils else (s.nfp if s.stellsym else 2 * s.nfp)
                base_curves_dipole = [dipole_coils[j].curve for j in range(0, ncoils_dipole, ntoroidal)] if ntoroidal and dipole_coils else []
                base_curves_TF = [c.curve for c in tf_coils[:ncoils]]
        else:
            if coils is None:
                raise RuntimeError("Cannot extend coils: previous step produced None coils")
            # Split by quadpoints (same as step 1) so ncoils_dipole and ntoroidal are correct
            nqp0 = len(coils[0].curve.quadpoints)
            split_idx = len(coils)
            for j in range(1, len(coils)):
                if len(coils[j].curve.quadpoints) != nqp0:
                    split_idx = j
                    break
            dipole_coils = coils[:split_idx]
            tf_coils = coils[split_idx:]
            ncoils_dipole = split_idx
            ntoroidal = len(tf_coils) // ncoils if ncoils and tf_coils else (s.nfp if s.stellsym else 2 * s.nfp)
            extended_tf = _extend_coils_to_higher_order(
                tf_coils, order, s, ncoils, regularization, coil_width
            )
            initial_coils = dipole_coils + extended_tf
            base_curves_dipole = [dipole_coils[j].curve for j in range(0, ncoils_dipole, ntoroidal)] if ntoroidal and dipole_coils else []
            base_curves_TF = [c.curve for c in extended_tf[:ncoils]]

            continuation_kwargs = kwargs.copy()
            continuation_kwargs['_cached_thresholds'] = cached_thresholds
            continuation_kwargs['continuation_ncoils_dipole'] = ncoils_dipole
            continuation_kwargs['continuation_base_curves_dipole'] = base_curves_dipole
            continuation_kwargs['continuation_base_curves_TF'] = base_curves_TF

            coils, results = optimize_coils_loop(
                s=s,
                dipole_array=True,
                target_B=target_B,
                out_dir=str(order_dir),
                max_iterations=max_iterations,
                ncoils=ncoils,
                order=order,
                verbose=verbose,
                regularization=regularization,
                coil_objective_terms=coil_objective_terms,
                initial_coils=initial_coils,
                surface_resolution=surface_resolution,
                skip_post_processing=True,
                **continuation_kwargs
            )

        results['fourier_order'] = order
        results['continuation_step'] = i + 1
        all_results.append(results)

    last_results = all_results[-1] if all_results else {}
    combined_results = {
        'fourier_continuation': True,
        'fourier_orders': fourier_orders,
        'final_order': fourier_orders[-1],
        'continuation_results': all_results,
        **last_results,
    }

    print(f"\n{'='*60}")
    print("Dipole Fourier continuation completed!")
    print(f"Final TF order: {fourier_orders[-1]}")
    print(f"{'='*60}\n")

    if coils is None:
        raise RuntimeError("Dipole Fourier continuation failed: no coils were produced")

    if not skip_post_processing:
        try:
            from .post_processing import run_post_processing
            final_order_dir = out_dir_path / f"order_{fourier_orders[-1]}"
            coils_json_path = final_order_dir / "biot_savart_optimized.json"
            if not coils_json_path.exists():
                coils_json_path = out_dir_path / "biot_savart_optimized.json"
            if not coils_json_path.exists():
                coils_json_path = out_dir_path / "coils.json"
            if coils_json_path.exists():
                print("\nRunning post-processing on final optimized coils...")
                run_post_processing(
                    coils_json_path=coils_json_path,
                    output_dir=out_dir_path,
                    case_yaml_path=Path(case_path) / "case.yaml" if case_path else None,
                    plot_boozer=plot_boozer,
                    plot_poincare=plot_poincare,
                    run_simple=run_simple,
                )
        except Exception as e:
            print(f"Warning: Post-processing failed: {e}")

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
def _nullcontext() -> Generator[None, None, None]:
    """Null context manager that does nothing (no-op for with statement)."""
    yield


@contextmanager
def _redirect_verbose_to_file(output_file: Path) -> Generator[None, None, None]:
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


def optimize_coils_loop(
    s: SurfaceRZFourier,
    target_B: float = 5.7,
    out_dir: Path | str = "",
    max_iterations: int = 30,
    ncoils: int = 4,
    order: int = 16,
    verbose: bool = True,
    regularization: Callable[..., Any] | None = regularization_circ,
    coil_objective_terms: Dict[str, Any] | None = None,
    initial_coils: List[Any] | None = None,
    surface_resolution: int = 32,
    skip_post_processing: bool = False,
    case_path: Path | None = None,
    run_vmec: bool = False,
    run_simple: bool = False,
    plot_poincare: bool = True,
    plot_boozer: bool = True,
    dipole_array: bool = False,
    **kwargs: Any,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Optimize modular or dipole coils for a plasma surface.

    For modular coils (default): initializes coils with target B-field (or uses
    initial_coils for Fourier continuation), then optimizes flux plus constraints
    via augmented Lagrangian or scipy (L-BFGS-B, BFGS, etc.).

    For dipole coils: pass dipole_array=True. Surface file and params are derived
    from s. TF coils come from initialize_coils_loop (normal stellcoilbench logic).
    Dipole coils (Nx × Nx grid) are created via initialize_coils_dipole.
    Default algorithm for dipole is L-BFGS-B.

    Optionally runs post-processing (QFM, VMEC, Poincaré, quasisymmetry).
    Delegates to _optimize_coils_loop_impl.

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    target_B : float, optional
        Target |B| on-axis in Tesla (default: 5.7).
    out_dir : Path | str, optional
        Output directory for VTK, JSON, plots.
    max_iterations : int, optional
        Maximum optimization iterations (default: 30).
    ncoils : int, optional
        Number of base coils (default: 4).
    order : int, optional
        Fourier order for coil curves (default: 16).
    verbose : bool, optional
        Print iteration progress (default: False).
    regularization : Callable | None, optional
        Coil regularization (default: regularization_circ).
    coil_objective_terms : Dict[str, Any] | None, optional
        Case config for constraints (length, curvature, MSC, force, torque, etc.).
    initial_coils : list | None, optional
        Coils from previous Fourier step; if set, skips initialization.
    surface_resolution : int, optional
        Surface quadrature for evaluation (default: 32). Lower values speed up
        optimization; use 8 for faster unit tests.
    skip_post_processing : bool, optional
        If True, skip QFM/VMEC/Poincaré (default: False).
    case_path : Path | None, optional
        Path to case.yaml for post-processing.
    run_vmec, run_simple, plot_poincare, plot_boozer : bool, optional
        Post-processing flags.
    dipole_array : bool, optional
        If True, use dipole + modular coil optimization. Surface file/params
        are derived from s; TF coils from initialize_coils_loop.
    **kwargs
        algorithm, max_iter_subopt, thresholds (length_threshold, flux_threshold,
        cc_threshold, cs_threshold, msc_threshold, curvature_threshold),
        vc_target, dof_perturbation, poff, coff, dipole_coil_size,
        Nx, dipole_order (dipole mode), dipole_coils_planar (default True;
        assume CurvePlanarFourier), fix_shapes, fix_currents (any Curve),
        fix_center, fix_orientation (CurvePlanarFourier only; when fix_shapes=True,
        curvature/MSC penalties are not applied to dipole coils), etc.

    Returns
    -------
    tuple
        (coils, results) - optimized coils and metrics dict.
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

    impl_kwargs = dict(kwargs)
    if dipole_array:
        impl_kwargs["dipole_array"] = True

    with redirect_context:
        return _optimize_coils_loop_impl(
            s, target_B, out_dir, max_iterations, ncoils, order, verbose,
            regularization, coil_objective_terms, initial_coils, surface_resolution,
            skip_post_processing, case_path, run_vmec, run_simple, plot_poincare,
            plot_boozer, **impl_kwargs
        )


def _run_post_processing_after_optimization(
    out_dir: Path,
    s: Any,
    case_path: Path | str | None,
    run_vmec: bool,
    run_simple: bool,
    plot_poincare: bool,
    plot_boozer: bool,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run post-processing (QFM, Poincaré, VMEC, quasisymmetry) after coil optimization.

    Resolves case.yaml path via case_path, out_dir, surface filename, and cases/
    directory search. Runs run_post_processing if coils JSON exists. Returns empty
    dict on failure or if coils not found.

    Parameters
    ----------
    out_dir : Path
        Output directory (contains biot_savart_optimized.json or coils.json).
    s : Surface
        Plasma surface (for filename-based case.yaml search).
    case_path : Path | str | None
        Case path hint (file or directory).
    run_vmec, run_simple, plot_poincare, plot_boozer : bool
        Post-processing flags.
    kwargs : Dict[str, Any]
        May contain plot_finite_build, finite_build_width, finite_build_height.

    Returns
    -------
    Dict[str, Any]
        Post-processing results (quasisymmetry_average, loss_fraction, etc.) or {}.
    """
    try:
        from .post_processing import run_post_processing

        case_yaml_path = None
        if case_path is not None:
            case_path_obj = Path(case_path) if isinstance(case_path, str) else case_path
            if case_path_obj.is_absolute() and case_path_obj.exists():
                if case_path_obj.is_file():
                    case_yaml_path = case_path_obj
                elif case_path_obj.is_dir():
                    case_yaml_path = case_path_obj / "case.yaml"
                    if not case_yaml_path.exists():
                        case_yaml_path = None
            elif case_path_obj.exists():
                case_yaml_path = case_path_obj.resolve()
                if case_yaml_path.is_dir():
                    case_yaml_path = case_yaml_path / "case.yaml"
                    if not case_yaml_path.exists():
                        case_yaml_path = None

        if case_yaml_path is None or not case_yaml_path.exists():
            case_yaml_path = out_dir / "case.yaml"
        if not case_yaml_path.exists():
            case_yaml_path = out_dir.parent / "case.yaml"

        if case_yaml_path is None or not case_yaml_path.exists():
            if hasattr(s, 'filename') and s.filename:
                surface_dir = Path(s.filename).parent
                surface_stem = surface_stem_from_filename(s.filename)
                for path in [
                    surface_dir / "case.yaml",
                    surface_dir.parent / "case.yaml",
                    Path("cases") / surface_stem / "case.yaml",
                ]:
                    if path.exists():
                        case_yaml_path = path
                        break

        if case_yaml_path is None or not case_yaml_path.exists():
            cases_dir = None
            current_dir = Path(out_dir)
            for _ in range(10):
                potential_cases_dir = current_dir / "cases"
                if potential_cases_dir.exists() and potential_cases_dir.is_dir():
                    cases_dir = potential_cases_dir
                    break
                if current_dir.parent == current_dir:
                    break
                current_dir = current_dir.parent
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
                            if surface_filename and surface_filename in surface_in_case:
                                case_yaml_path = yaml_file.resolve()
                                break
                            if surface_in_case in surface_filename:
                                case_yaml_path = yaml_file.resolve()
                                break
                    except Exception:
                        continue

        coils_json_path = out_dir / "biot_savart_optimized.json"
        if not coils_json_path.exists():
            coils_json_path = out_dir / "coils.json"

        if not coils_json_path.exists():
            print(f"Warning: Skipping post-processing (coils_json not found: {coils_json_path})")
            return {}

        print("\nRunning post-processing (QFM, Poincaré plots, profiles)...")

        helicity_n = 0
        if case_yaml_path is not None and case_yaml_path.exists():
            import yaml
            try:
                case_data = yaml.safe_load(case_yaml_path.read_text())
                surface_name = case_data.get("surface_params", {}).get("surface", "").lower()
                if "qh" in surface_name or "qash" in surface_name:
                    helicity_n = -1
            except Exception:
                pass

        plasma_surfaces_dir = find_plasma_surfaces_dir(Path(out_dir))

        post_processing_results = run_post_processing(
            coils_json_path=coils_json_path,
            output_dir=out_dir,
            case_yaml_path=case_yaml_path if (case_yaml_path is not None and case_yaml_path.exists()) else None,
            plasma_surfaces_dir=plasma_surfaces_dir,
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
        return post_processing_results

    except Exception as e:
        print(f"Warning: Post-processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


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
    dipole_array: bool = False,
    **kwargs):
    """
    Internal implementation of modular coil optimization.

    Performs the full pipeline: (1) initialize coils with target B-field or use
    initial_coils for Fourier continuation; (2) create BiotSavart and save initial
    state; (3) build constraint list from coil_objective_terms; (4) run augmented
    Lagrangian or scipy (L-BFGS-B, BFGS, etc.); (5) save optimized coils and
    compute metrics; (6) optionally run post-processing (QFM, VMEC, Poincaré).

    Thresholds are scaled by minor radius (a0) unless overridden in kwargs.
    Constraint scaling makes weights dimensionless. Supports virtual casing
    target via kwargs['vc_target'].

    Parameters
    ----------
    s : SurfaceRZFourier
        Plasma boundary surface.
    target_B : float, optional
        Target |B| on-axis in Tesla (default: 5.7).
    out_dir : Path | str, optional
        Output directory for VTK, JSON, plots.
    max_iterations : int, optional
        Maximum optimization iterations (default: 30).
    ncoils : int, optional
        Number of base coils (default: 4).
    order : int, optional
        Fourier order for coil curves (default: 16).
    verbose : bool, optional
        Print iteration progress (default: False).
    regularization : Callable | None, optional
        Coil regularization (default: regularization_circ).
    coil_objective_terms : Dict[str, Any] | None, optional
        Case config for constraints (length, curvature, MSC, force, torque, etc.).
    initial_coils : list | None, optional
        Coils from previous Fourier step; if set, skips initialization.
    surface_resolution : int, optional
        Surface quadrature for evaluation (default: 32).
    skip_post_processing : bool, optional
        If True, skip QFM/VMEC/Poincaré (default: False).
    case_path : Path | None, optional
        Path to case.yaml for post-processing.
    run_vmec, run_simple, plot_poincare, plot_boozer : bool, optional
        Post-processing flags.
    **kwargs
        Algorithm, thresholds, vc_target, dof_perturbation, etc.

    Returns
    -------
    tuple
        (coils, results) - optimized coils and metrics dict.
    """
    import time
    from simsopt.objectives import SquaredFlux

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    is_continuation_step = initial_coils is not None and not dipole_array
    config = _parse_optimizer_config(s, kwargs, max_iterations, is_continuation_step=is_continuation_step)
    algorithm = config['algorithm']
    algorithm_options = config['algorithm_options']
    max_iter_subopt = config['max_iter_subopt']
    max_iterations = config['max_iterations']
    th = config['thresholds']
    length_threshold = th['length_threshold']
    flux_threshold = th['flux_threshold']
    cc_threshold = th['cc_threshold']
    cs_threshold = th['cs_threshold']
    msc_threshold = th['msc_threshold']
    arclength_variation_threshold = th.get('arclength_variation_threshold', 0.0)
    curvature_threshold = th['curvature_threshold']
    force_threshold = th['force_threshold']
    torque_threshold = th['torque_threshold']
    coil_width = th.get('coil_width', 0.4 / th['a0'])
    major_radius = th['major_radius']

    # Step 1: Initialize coils (TF coils) and optionally dipole coils
    dipole_coils = None
    ncoils_dipole = 0
    base_curves_dipole = []
    base_curves_TF = []
    fix_shapes = False
    with timed_section("coil_initialization", print_time=False):
        if dipole_array:
            continuation_ncoils_dipole = kwargs.get('continuation_ncoils_dipole')
            continuation_base_dipole = kwargs.get('continuation_base_curves_dipole')
            continuation_base_TF = kwargs.get('continuation_base_curves_TF')
            if initial_coils is not None and continuation_ncoils_dipole is not None and continuation_base_dipole is not None and continuation_base_TF is not None:
                # Split by quadpoints: dipole (CurvePlanarFourier) uses ~40 qp; extended TF
                # (CurveXYZFourier) uses 200. LpCurveForce requires uniform quadpoints per source.
                nqp0 = len(initial_coils[0].curve.quadpoints)
                split_idx = len(initial_coils)
                for j in range(1, len(initial_coils)):
                    if len(initial_coils[j].curve.quadpoints) != nqp0:
                        split_idx = j
                        break
                dipole_coils = initial_coils[:split_idx]
                coils = initial_coils[split_idx:]
                ncoils_dipole = split_idx
                # Caller now uses quadpoints split + TF-based ntoroidal, so continuation_base_dipole
                # has the correct count (e.g. 21 unique for 84 dipole coils).
                base_curves_dipole = list(continuation_base_dipole)
                base_curves_TF = [c.curve for c in coils[:ncoils]]
            else:
                surface_file, surface_params = _surface_file_and_params_from_s(s)
                Nx = kwargs.get('Nx', 4)
                dipole_order = kwargs.get('dipole_order', 2)
                with suppress_output():
                    coils = initialize_coils_loop(s, out_dir=out_dir, target_B=target_B, ncoils=ncoils, order=order, coil_width=coil_width, regularization=regularization)
                    all_coils, base_curves_dipole, base_curves_TF, ncoils_dipole, _ = initialize_coils_dipole(
                    s,
                    surface_file=surface_file,
                    surface_params=surface_params,
                    Nx=Nx,
                    dipole_order=dipole_order,
                    out_dir=out_dir,
                    base_coils_TF=coils,
                    ncoils_TF=ncoils,
                    **{k: v for k, v in kwargs.items() if k in ("poff", "coff", "Ny", "Nz", "dipole_coil_size", "remove_inboard_eps")},
                )
                print(len(all_coils), len(base_curves_dipole), len(base_curves_TF), ncoils_dipole)
                dipole_coils = all_coils[:ncoils_dipole]
                coils = all_coils[ncoils_dipole:]
            if not dipole_coils:
                print(
                    "Warning: All dipole coils were removed by inboard/interlinking filters. "
                    "Try increasing poff/coff or less aggressive remove_inboard_eps in coils_params."
                )
            dipole_coils_planar = kwargs.get('dipole_coils_planar', True)
            fix_shapes = kwargs.get('fix_shapes', False)
            fix_currents = kwargs.get('fix_currents', False)
            fix_center = kwargs.get('fix_center', False)
            fix_orientation = kwargs.get('fix_orientation', False)
            for c in dipole_coils:
                if fix_shapes and hasattr(c.curve, 'fix_all'):
                    c.curve.fix_all()
                if fix_currents and hasattr(c.current, 'fix_all'):
                    c.current.fix_all()
                if dipole_coils_planar and (fix_center or fix_orientation) and hasattr(c.curve, 'fix'):
                    names = getattr(c.curve, 'local_dof_names', None) or getattr(c.curve, 'dof_names', None)
                    if names is not None:
                        for i, name in enumerate(names):
                            if not name:
                                continue
                            n = str(name)
                            if fix_center and n in ('X', 'Y', 'Z'):
                                try:
                                    c.curve.fix(i)
                                except Exception:
                                    pass
                            elif fix_orientation and n in ('q0', 'qi', 'qj', 'qk'):
                                try:
                                    c.curve.fix(i)
                                except Exception:
                                    pass
        elif initial_coils is None:
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
    if not is_continuation_step and not dipole_array and ('force_threshold' not in kwargs or 'torque_threshold' not in kwargs):
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
    if dipole_array:
        base_curves = (list(base_curves_TF) if fix_shapes else list(base_curves_dipole) + list(base_curves_TF))
    else:
        base_curves = [coil.curve for coil in coils[:ncoils]]

    # Step 2: Create plotting surface for visualization
    s_plot, qphi, qtheta = _create_plotting_surface(s, surface_resolution, kwargs)

    # Step 3: Create BiotSavart object and save initial state
    with timed_section("biotsavart_setup", print_time=False):
        coils_for_bs = (dipole_coils + coils) if dipole_array else coils
        bs, curves, B_initial = _setup_biotSavart_and_initial_save(
            coils_for_bs, s, s_plot, qphi, qtheta, out_dir
        )

    # Step 4: Define objective function and constraints
    objective_setup_start = time.perf_counter()
    bs.set_points(s.gamma().reshape((-1, 3)))
    dipole_coil_objective_terms = coil_objective_terms  # set for dipole; overwritten in dipole block

    # Main objective: Squared flux (always included)
    vc_target = kwargs.get('vc_target', None)
    if vc_target is not None:
        print(f"Using virtual casing target for SquaredFlux (target shape: {vc_target.shape})")
        Jf = SquaredFlux(s, bs, target=vc_target, threshold=flux_threshold)
    else:
        Jf = SquaredFlux(s, bs, threshold=flux_threshold)

    if dipole_array:
        dipole_coil_objective_terms = coil_objective_terms
        if dipole_coil_objective_terms is None:
            dipole_coil_objective_terms = {
                "total_length": "l2_threshold",
                "coil_curvature": "lp_threshold",
                "coil_mean_squared_curvature": "l2_threshold",
                "linking_number": "",
                "coil_coil_force": "lp",
            }
        constraint_objs = _build_dipole_coil_constraint_objects(
            curves,
            base_curves_dipole,
            base_curves_TF,
            dipole_coils,
            coils,
            fix_shapes,
            kwargs.get("fix_center", True),
            kwargs.get("fix_orientation", True),
            s,
            cc_threshold,
            cs_threshold,
            curvature_threshold,
            force_threshold,
            torque_threshold,
            dipole_coil_objective_terms,
        )
        Jls = constraint_objs["Jls"]
        Jccdist = constraint_objs["Jccdist"]
        Jcsdist = constraint_objs["Jcsdist"]
        Jalenvar = constraint_objs["Jalenvar"]
        Jcs = constraint_objs["Jcs"]
        Jlink = constraint_objs["Jlink"]
        Jforce = constraint_objs["Jforce"]
        Jtorque = constraint_objs["Jtorque"]
        Jmscs = constraint_objs["Jmscs"]

        ncoils_dipoles = len(base_curves_dipole)
        dipole_length_split = None
        if not fix_shapes and ncoils_dipoles > 0 and "total_length" in (dipole_coil_objective_terms or {}):
            from simsopt.geo import CurveLength
            Jls_dipole = Jls[:ncoils_dipoles]
            Jls_tf = Jls[ncoils_dipoles:]
            initial_dipole_length = sum(float(CurveLength(c).J()) for c in base_curves_dipole)
            length_threshold_dipole = float(
                kwargs.get("length_threshold_dipole")
                or (dipole_coil_objective_terms or {}).get("length_threshold_dipole", initial_dipole_length)
            )
            length_threshold_tf = float(length_threshold)
            dipole_length_split = (Jls_dipole, Jls_tf, length_threshold_dipole, length_threshold_tf)

        thresholds_for_build = {
            "cc_threshold": cc_threshold,
            "cs_threshold": cs_threshold,
            "length_threshold": length_threshold,
            "curvature_threshold": curvature_threshold,
            "arclength_variation_threshold": arclength_variation_threshold,
            "msc_threshold": msc_threshold,
            "force_threshold": force_threshold,
            "torque_threshold": torque_threshold,
        }
        c_list, constraint_scaling, cc_distance_idx, cs_distance_idx, constraint_names_and_thresholds, constraint_idx_to_term = (
            _build_c_list_and_constraint_scaling_from_coil_objective_terms(
                Jf, Jccdist, Jcsdist, Jls, Jcs, Jalenvar, Jmscs, Jlink, Jforce, Jtorque,
                dipole_coil_objective_terms, thresholds_for_build, major_radius, total_current,
                dipole_length_split=dipole_length_split,
            )
        )
        cc_distance_index = cc_distance_idx
        cs_distance_index = cs_distance_idx
    else:
        # Build constraint objects for modular coils
        constraint_objs = _build_modular_coil_constraint_objects(
            curves, base_curves, coils, ncoils, s,
            cc_threshold, cs_threshold, curvature_threshold, force_threshold, torque_threshold,
            coil_objective_terms,
        )
        Jls = constraint_objs["Jls"]
        Jccdist = constraint_objs["Jccdist"]
        Jcsdist = constraint_objs["Jcsdist"]
        Jalenvar = constraint_objs["Jalenvar"]
        Jcs = constraint_objs["Jcs"]
        Jlink = constraint_objs["Jlink"]
        Jforce = constraint_objs["Jforce"]
        Jtorque = constraint_objs["Jtorque"]
        Jmscs = constraint_objs["Jmscs"]

        thresholds_for_build = {
            "cc_threshold": cc_threshold,
            "cs_threshold": cs_threshold,
            "length_threshold": length_threshold,
            "curvature_threshold": curvature_threshold,
            "arclength_variation_threshold": arclength_variation_threshold,
            "msc_threshold": msc_threshold,
            "force_threshold": force_threshold,
            "torque_threshold": torque_threshold,
        }
        c_list, constraint_scaling, cc_distance_idx, cs_distance_idx, constraint_names_and_thresholds, constraint_idx_to_term = (
            _build_c_list_and_constraint_scaling_from_coil_objective_terms(
                Jf, Jccdist, Jcsdist, Jls, Jcs, Jalenvar, Jmscs, Jlink, Jforce, Jtorque,
                coil_objective_terms, thresholds_for_build, major_radius, total_current,
            )
        )
        cs_distance_index = cs_distance_idx
        cc_distance_index = cc_distance_idx
    
    # Record objective setup time
    objective_setup_time = time.perf_counter() - objective_setup_start
    from .post_processing import _timing_results
    _timing_results["objective_setup"] = objective_setup_time

    # Print coil counts before optimization
    ncoils_tf_total = len(coils)
    print(f"TF coils: {ncoils} unique, {ncoils_tf_total} total (before optimization)")
    if dipole_array and dipole_coils is not None:
        ncoils_dipole_unique = len(base_curves_dipole)
        print(f"Dipole coils: {ncoils_dipole_unique} unique, {ncoils_dipole} total (before optimization)")

    # Step 5: Run optimization
    optimization_start = time.perf_counter()
    start_time = time.time()
    lag_mul = None  # Initialize lag_mul for scipy methods
    iterations_used = 0  # Track total iterations for CI reporting
    opt_result = None  # Scipy/minimize result for metadata (auglag does not provide this)
    
    if algorithm == "augmented_lagrangian":
        if dipole_array:
            _apply_distance_weights_for_auglag(
                c_list, constraint_scaling, cc_distance_index, cs_distance_index, kwargs,
            )
        else:
            _apply_distance_weights_for_auglag(
                c_list, constraint_scaling, cc_distance_index, cs_distance_index, kwargs
            )
        _run_augmented_lagrangian(c_list, max_iterations, max_iter_subopt, verbose, kwargs)
        iterations_used = max_iterations
    elif algorithm in ['BFGS', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead', 'Powell', 'CG', 'Newton-CG', 'TNC', 'COBYLA', 'trust-constr']:
        if dipole_array:
            result, iterations_used = _run_scipy_minimize_for_modular_coils(
                c_list, constraint_scaling, constraint_idx_to_term,
                cc_distance_index, cs_distance_index, constraint_names_and_thresholds,
                base_curves, Jls, Jccdist, Jcsdist, Jlink, Jforce, Jtorque,
                dipole_coil_objective_terms, algorithm, max_iterations, algorithm_options,
                verbose, kwargs,
            )
        else:
            result, iterations_used = _run_scipy_minimize_for_modular_coils(
                c_list, constraint_scaling, constraint_idx_to_term,
                cc_distance_index, cs_distance_index, constraint_names_and_thresholds,
                base_curves, Jls, Jccdist, Jcsdist, Jlink, Jforce, Jtorque,
                coil_objective_terms, algorithm, max_iterations, algorithm_options,
                verbose, kwargs,
            )
        opt_result = result
    
    end_time = time.time()
    optimization_time = time.perf_counter() - optimization_start
    _timing_results["coil_optimization"] = optimization_time
    
    # Start timing for save and metrics section
    save_metrics_start = time.perf_counter()

    if dipole_array:
        try:
            from simsopt.util import save_coil_sets
            from simsopt.field import BiotSavart
        except ImportError as e:
            raise ImportError(
                "Dipole coil optimization requires simsopt with auglag_coils branch "
                "(save_coil_sets). Install from: "
                "https://github.com/hiddenSymmetries/simsopt"
            ) from e
        btot_dipole = BiotSavart(dipole_coils) + BiotSavart(coils)
        save_coil_sets(btot_dipole, str(out_dir) + "/", "_optimized")
        out_dir_path = Path(out_dir)
        btot_dipole.save(out_dir_path / "biot_savart_optimized.json")
        # Save surface with B_N, B_N/|B|, modB (matches modular path)
        btot_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
        pointData = {
            "B_N": np.sum(btot_dipole.B().reshape((qphi, qtheta, 3)) *
                          s_plot.unitnormal(), axis=2)[:, :, None],
            "B_N/|B|": np.sum(btot_dipole.B().reshape((qphi, qtheta, 3)) *
                              s_plot.unitnormal(), axis=2)[:, :, None] /
                        btot_dipole.AbsB().reshape((qphi, qtheta, 1)),
            "modB": btot_dipole.AbsB().reshape((qphi, qtheta, 1))
        }
        s_plot.to_vtk(out_dir_path / "surface_optimized", extra_data=pointData)
        # Compute full metrics (same as modular path)
        ncoils_dipole_all = len(coils_for_bs)
        metrics = _compute_optimization_metrics(
            btot_dipole, coils_for_bs, base_curves, ncoils_dipole_all,
            s, s_plot, qphi, qtheta, kwargs
        )
        B_final = metrics["B_final"]
        max_force = metrics["max_force"]
        max_torque = metrics["max_torque"]
        avg_BdotN_over_B = metrics["avg_BdotN_over_B"]
        max_BdotN_overB = metrics["max_BdotN_overB"]
        coils_linked_to_surface = metrics["coils_linked_to_surface"]
        total_current_final = metrics["total_current_final"]
        try:
            vc_target_plot = kwargs.get('vc_target_plot', None)
            _plot_bn_error_3d(
                s_plot,
                btot_dipole,
                coils_for_bs,
                out_dir_path,
                filename="bn_error_3d_plot.pdf",
                title="B_N/|B| Error on Plasma Surface with Optimized Coils",
                vc_target=vc_target_plot,
            )
        except Exception as e:
            print(f"Warning: Failed to generate 3D plot: {e}")
        # Compute dipole-only and TF-only metrics for separate reporting
        dipole_metrics = _compute_coil_subset_metrics(
            dipole_coils, base_curves_dipole, coils_for_bs, s, kwargs
        )
        tf_metrics = _compute_coil_subset_metrics(
            coils, base_curves_TF, coils_for_bs, s, kwargs
        )
        coils_return = coils_for_bs
    else:
        metrics = _save_optimized_coils_and_compute_metrics(
            coils, base_curves, ncoils, s, s_plot, qphi, qtheta, bs, out_dir, kwargs
        )
        B_final = metrics["B_final"]
        max_force = metrics["max_force"]
        max_torque = metrics["max_torque"]
        avg_BdotN_over_B = metrics["avg_BdotN_over_B"]
        max_BdotN_overB = metrics["max_BdotN_overB"]
        coils_linked_to_surface = metrics["coils_linked_to_surface"]
        total_current_final = metrics["total_current_final"]
        coils_return = coils

    # Record save and metrics time
    save_metrics_time = time.perf_counter() - save_metrics_start
    _timing_results["save_and_metrics"] = save_metrics_time

    post_processing_results = (
        _run_post_processing_after_optimization(
            out_dir, s, case_path, run_vmec, run_simple, plot_poincare, plot_boozer, kwargs
        )
        if not skip_post_processing
        else {}
    )
    
    # Note: Individual file zipping is disabled - the entire submission directory
    # will be zipped by submit-case command after all files are written

    if dipole_array:
        cached_thresholds = {k: v for k, v in th.items() if k in (
            'length_threshold', 'flux_threshold', 'cc_threshold', 'cs_threshold',
            'msc_threshold', 'arclength_variation_threshold', 'curvature_threshold',
            'force_threshold', 'torque_threshold', 'coil_width', 'a0',
            'major_radius', 'minor_radius',
        )}
        btot_dipole.set_points(s.gamma().reshape((-1, 3)))
        results = _build_optimization_results_dict(
            B_initial=B_initial,
            B_final=B_final,
            target_B=target_B,
            end_time=end_time,
            start_time=start_time,
            iterations_used=iterations_used,
            Jf=Jf,
            Jcsdist=Jcsdist,
            Jccdist=Jccdist,
            Jlink=Jlink,
            opt_result=opt_result,
            cached_thresholds=cached_thresholds,
            base_curves=base_curves,
            coils=coils_for_bs,
            ncoils=len(coils_for_bs),
            total_current=total_current,
            total_current_final=total_current_final,
            max_force=max_force,
            max_torque=max_torque,
            avg_BdotN_over_B=avg_BdotN_over_B,
            max_BdotN_overB=max_BdotN_overB,
            coils_linked_to_surface=coils_linked_to_surface,
            lag_mul=None,
            out_dir=out_dir,
            th=th,
        )
        # Backward compatibility: success, fun, nit (used by tests and evaluate)
        results["success"] = results.get("optimization_success", True)
        results["fun"] = (
            float(opt_result.fun)
            if opt_result is not None and hasattr(opt_result, 'fun')
            else float(results.get("final_squared_flux", 0.0))
        )
        results["nit"] = int(iterations_used)
        # Fix total_current_after if metrics returned 0 (e.g. dipole coil API)
        if results.get("total_current_after", 0) == 0 and results.get("final_current_per_coil"):
            results["total_current_after"] = float(sum(results["final_current_per_coil"]))
        # Separate dipole and TF coil metrics for dipole runs
        results["dipole_metrics"] = dipole_metrics
        results["tf_metrics"] = tf_metrics
        if post_processing_results:
            for key, value in post_processing_results.items():
                if key in ['quasisymmetry_average', 'loss_fraction', 'BdotN', 'BdotN_over_B']:
                    if isinstance(value, (int, float)):
                        results[key] = float(value)
    else:
        cached_thresholds = {k: v for k, v in th.items() if k in (
            'length_threshold', 'flux_threshold', 'cc_threshold', 'cs_threshold',
            'msc_threshold', 'arclength_variation_threshold', 'curvature_threshold',
            'force_threshold', 'torque_threshold', 'coil_width', 'a0',
            'major_radius', 'minor_radius',
        )}
        bs.set_points(s.gamma().reshape((-1, 3)))
        results = _build_optimization_results_dict(
            B_initial=B_initial,
            B_final=B_final,
            target_B=target_B,
            end_time=end_time,
            start_time=start_time,
            iterations_used=iterations_used,
            Jf=Jf,
            Jcsdist=Jcsdist,
            Jccdist=Jccdist,
            Jlink=Jlink,
            opt_result=opt_result,
            cached_thresholds=cached_thresholds,
            base_curves=base_curves,
            coils=coils,
            ncoils=ncoils,
            total_current=total_current,
            total_current_final=total_current_final,
            max_force=max_force,
            max_torque=max_torque,
            avg_BdotN_over_B=avg_BdotN_over_B,
            max_BdotN_overB=max_BdotN_overB,
            coils_linked_to_surface=coils_linked_to_surface,
            lag_mul=lag_mul,
            out_dir=out_dir,
            th=th,
        )
        if post_processing_results:
            for key, value in post_processing_results.items():
                if key in ['quasisymmetry_average', 'loss_fraction', 'BdotN', 'BdotN_over_B']:
                    if isinstance(value, (int, float)):
                        results[key] = float(value)

    results['timing'] = get_timing_results()
    return coils_return, results