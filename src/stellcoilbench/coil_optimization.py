from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import numpy as np
from typing import Callable
from .config_scheme import CaseConfig

from simsopt.geo import SurfaceRZFourier
from simsopt.field import regularization_circ


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
    else:
        raise ValueError(f"Unknown surface file: {surface_file}")
    coil_params['target_B'] = target_B

    try:
        os.chdir(output_dir)
        
        # Pass output_dir to optimize_coils_loop for VTK file output
        # optimize_coils_loop saves VTK files to output_dir during optimization
        try:
            coils, results_dict = optimize_coils_loop(
                surface, 
                **coil_params, 
                **optimizer_params, 
                output_dir=str(output_dir)
            )
        except TypeError:
            # Fallback if optimize_coils_loop doesn't accept output_dir parameter
            # Files will be saved to current directory (which is now output_dir)
            coils, results_dict = optimize_coils_loop(
                surface, 
                **coil_params, 
                **optimizer_params, 
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
    regularization: Callable = regularization_circ):
    """
    Initializes four coils with order=16 and total current set to produce 
    a target B-field on-axis. The coil centers and radii are scaled by 
    the plasma surface major radius. The function iteratively adjusts the
    total current until the field strength along the major radius averages
    to the target value.

    Args:
        s: plasma boundary surface.
        out_dir: Path or string for the output directory for saved files.
        target_B: Target magnetic field strength in Tesla (default: 5.7).
        ncoils: Number of coils to create (default: 4).
        order: Fourier order for coil curves (default: 16).
        coil_width: Width of the coil in meters (default: 0.05).
        regularization: Regularization function (default: regularization_circ).
    Returns:
        coils: List of Coil class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, coils_via_symmetries, BiotSavart
    from simsopt.util.coil_optimization_helper_functions import calculate_modB_on_major_radius

    out_dir = Path(out_dir)

    if regularization is not None:
        regularizations = [regularization(coil_width) for _ in range(ncoils)]
    else:
        regularizations = None
    
    # Get the major radius from the surface and scale coil parameters
    R0 = s.get_rc(0, 0)  # Major radius
    R1 = s.get_rc(1, 0) * 2.5  # Scale the minor radius component
    
    # Initial guess for total current (using QH configuration as reference)
    total_current = 5e7  # 50 MA initial guess is not bad for reactor-scale
    
    # Create equally spaced curves with the specified parameters
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=s.stellsym,
        R0=R0, R1=R1, order=order, numquadpoints=256)
    coils = coils_via_symmetries(base_curves, np.ones(ncoils), s.nfp, s.stellsym, regularizations=regularizations)
    
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
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym, regularizations=regularizations)
        
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


def optimize_coils_loop(
    s : SurfaceRZFourier, target_B : float = 5.7, out_dir : Path | str = '', 
    max_iterations : int = 30, max_iter_lag : int = 10, 
    ncoils : int = 4, order : int = 16, nphi : int = 32, ntheta : int = 32, 
    verbose : bool = False, coil_width : float = 0.4,
    regularization : Callable = regularization_circ, **kwargs):
    """
    Performs complete coil optimization including initialization and optimization.
    This function initializes coils with the target B-field and then optimizes
    them using the augmented Lagrangian method.

    Args:
        s: plasma boundary surface.
        target_B: Target magnetic field strength in Tesla (default: 5.7).
        out_dir: Path or string for the output directory for saved files.
        max_iterations: Maximum number of optimization iterations (default: 1500).
        max_iter_lag: Maximum number of Lagrangian iterations (default: 50).
        ncoils: Number of base coils to create (default: 4).
        order: Fourier order for coil curves (default: 16).
        nphi: Number of phi points for surface discretization (default: 32).
        ntheta: Number of theta points for surface discretization (default: 32).
        verbose: Print out progress and results (default: False).
        **kwargs: Additional keyword arguments for constraint thresholds.
    Returns:
        coils: List of optimized Coil class objects.
        results: Dictionary containing optimization results and metrics.
    """
    import time
    from pathlib import Path
    from simsopt.geo import SurfaceRZFourier
    from simsopt.geo import LinkingNumber, CurveLength, CurveCurveDistance
    from simsopt.geo import LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature
    from simsopt.objectives import SquaredFlux, QuadraticPenalty, Weight
    from simsopt.solve import augmented_lagrangian_method
    from simsopt.field import BiotSavart, coils_to_vtk
    from simsopt.field.force import LpCurveForce, LpCurveTorque, coil_force, coil_torque
    from simsopt.util import calculate_modB_on_major_radius

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set default constraint thresholds if not provided
    # Defaults here are reasonable for 10 m major radius
    #  reactor-scale device with 5.7 T target B-field
    length_target = kwargs.get('length_target', 210.0)
    flux_threshold = kwargs.get('flux_threshold', 1e-8)
    cc_threshold = kwargs.get('cc_threshold', 1.0)
    cs_threshold = kwargs.get('cs_threshold', 1.5)
    msc_threshold = kwargs.get('msc_threshold', 1.0)
    curvature_threshold = kwargs.get('curvature_threshold', 1.0)

    nturns = 1  # nturns = 1 for standardization
    force_threshold = kwargs.get('force_threshold', 1.0) * nturns
    torque_threshold = kwargs.get('torque_threshold', 1.0) * nturns

    # Rescale all the length thresholds by the plasma major radius
    # divided by the 10m assumption for the major radius
    R0 = 10.0 / s.get_rc(0, 0)
    length_target /= R0
    length_target *= (ncoils / 7.0) ** 0.5
    cc_threshold /= R0
    cs_threshold /= R0 
    curvature_threshold *= R0
    msc_threshold *= R0
    coil_width /= R0

    print(f"Starting coil optimization for target B-field: {target_B} T")
    print(f"Surface major radius: {s.get_rc(0, 0):.3f} m")
    print(f"Surface minor radius component: {s.get_rc(1, 0):.3f} m")
    print(f"Number of base coils: {ncoils}")
    print(f"Fourier order: {order}")

    # Step 1: Initialize coils with target B-field
    # print("Step 1: Initializing coils with target B-field...")
    coils = initialize_coils_loop(s, out_dir=out_dir, target_B=target_B, ncoils=ncoils, order=order, coil_width=coil_width, regularization=regularization)

    # Rescale force_threshold
    total_current = sum([c.current.get_value() for c in coils[:ncoils]]) / (s.stellsym + 1) / s.nfp
    coils_backup = initialize_coils_loop(s, out_dir=out_dir, ncoils=ncoils, order=order, coil_width=coil_width, regularization=regularization)
    total_current_reactor_scale = sum([c.current.get_value() for c in coils_backup[:ncoils]]) / (s.stellsym + 1) / s.nfp
    force_threshold *= (total_current / total_current_reactor_scale) ** 2
    torque_threshold *= (total_current / total_current_reactor_scale) ** 2

    # Extract base curves and currents from the initialized coils
    base_curves = [coil.curve for coil in coils[:ncoils]]
    
    # Step 2: Create plotting surface for visualization
    # print("Step 2: Setting up plotting surface...")
    qphi = 4 * nphi
    qtheta = 4 * ntheta
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
    # print(f"Initial B-field on-axis: {B_initial:.3f} T")
    
    # Save initial surface data
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                          s_plot.unitnormal(), axis=2)[:, :, None] / 
                    bs.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs.AbsB().reshape((qphi, qtheta, 1))
    }
    s_plot.to_vtk(out_dir / "surface_initial", extra_data=pointData)

    # Step 4: Define objective function and constraints
    # print("Step 4: Setting up optimization objectives and constraints...")
    bs.set_points(s.gamma().reshape((-1, 3)))
    
    # Main objective: Squared flux
    Jf = SquaredFlux(s, bs, definition="normalized", threshold=flux_threshold)
    
    # Constraint terms
    Jls = [CurveLength(c) for c in base_curves]
    Jl = sum(QuadraticPenalty(jj, length_target, "max") for jj in Jls)
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, cs_threshold)
    Jcs = [LpCurveCurvature(c, 2, curvature_threshold) for c in base_curves]
    Jlink = LinkingNumber(curves, downsample=2)
    Jforce = LpCurveForce(coils[:ncoils], coils, p=2.0, threshold=force_threshold, downsample=2)
    Jtorque = LpCurveTorque(coils[:ncoils], coils, p=2.0, threshold=torque_threshold, downsample=2)
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    # Print initial constraint values
    print("Initial thresholds:")
    print(f" Flux Threshold: {flux_threshold:.2e}")
    print(f" Length Target: {length_target:.2e}")
    print(f" CC Threshold: {cc_threshold:.2e}")
    print(f" CS Threshold: {cs_threshold:.2e}")
    print(f" MSC Threshold: {msc_threshold:.2e}")
    print(f" Curvature Threshold: {curvature_threshold:.2e}")
    print(f" Force Threshold: {force_threshold:.2e}")
    print(f" Torque Threshold: {torque_threshold:.2e}")

    # Step 5: Run optimization
    # print("Step 5: Running optimization...")
    start_time = time.time()
    
    # Constraint list for augmented Lagrangian method
    c_list = [
        Jf,
        Jccdist,
        Weight(1e3) * Jcsdist,  # Special attention to avoiding coil-surface intersections
        QuadraticPenalty(sum(Jls), length_target, "max"),
        sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs),
        sum(Jcs),
        Jlink,
        Jforce,
        Jtorque
    ]
    
    # Run optimization
    _, _, lag_mul = augmented_lagrangian_method(
        f=None,  # No main objective function
        equality_constraints=c_list,
        MAXITER=max_iterations,
        MAXITER_lag=max_iter_lag,
        verbose=verbose,
    )
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.1f} seconds")
    
    # Save optimized coils
    coils_to_vtk(coils, out_dir / "coils_optimized", nturns=nturns)
    bs.save(out_dir / "biot_savart_optimized.json")
    
    # Calculate and display final B-field
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    B_final = calculate_modB_on_major_radius(bs, s_plot)
    print(f"Final B-field on-axis: {B_final:.3f} T")
    
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
    print(f"  Length constraint: {Jl.J():.2e}")
    print(f"  Curvature constraint: {sum(Jcs).J():.2e}")
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
    nphi_s = len(s.quadpoints_phi)
    ntheta_s = len(s.quadpoints_theta)
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi_s, ntheta_s, 3)) * s.unitnormal(), axis=2)))
    avg_BdotN_over_B = BdotN / bs.AbsB().mean()
    
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    nphi_plot = len(s_plot.quadpoints_phi)
    ntheta_plot = len(s_plot.quadpoints_theta)
    max_BdotN_overB = np.max(np.abs(np.sum(bs.B().reshape((nphi_plot, ntheta_plot, 3)) *
                                          s_plot.unitnormal(), axis=2)) /
                            bs.AbsB().reshape((nphi_plot, ntheta_plot, 1)))
    
    print(f"  <B_N>/<|B|> = {avg_BdotN_over_B:.2e}")
    print(f"  Max |B_N|/|B| = {max_BdotN_overB:.2e}")    
    print("Optimization completed successfully!")
    print(f"Results saved to: {out_dir}")
    
    # Prepare results dictionary
    bs.set_points(s.gamma().reshape((-1, 3)))
    results = {
        'initial_B_field': B_initial,
        'final_B_field': B_final,
        'target_B_field': target_B,
        'optimization_time': end_time - start_time,
        'final_normalized_squared_flux': Jf.J(),
        'final_min_cs_separation': Jcsdist.shortest_distance(),
        'final_min_cc_separation': Jccdist.shortest_distance(),
        'final_total_length': sum(CurveLength(c).J() for c in base_curves),
        'final_max_curvature': max(np.max(c.kappa()) for c in base_curves),
        'final_average_curvature': np.mean([c.kappa() for c in base_curves]),
        'final_mean_squared_curvature': np.max([np.mean(c.kappa() ** 2) for c in base_curves]),
        'final_linking_number': Jlink.J(),
        'final_max_max_coil_force': np.max(max_force),
        'final_avg_max_coil_force': np.mean(max_force),
        'final_max_max_coil_torque': np.max(max_torque),
        'final_avg_max_coil_torque': np.mean(max_torque),
        'avg_BdotN_over_B': avg_BdotN_over_B,
        'max_BdotN_over_B': max_BdotN_overB,
        'lagrange_multipliers': lag_mul,
        'output_directory': str(out_dir)
    }
    
    return coils, results