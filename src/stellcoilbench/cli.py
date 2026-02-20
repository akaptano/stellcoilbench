# src/stellcoilbench/cli.py
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import typer

# MPI utilities for rank-aware file operations
# Use lazy import to avoid MPI library loading issues on systems without MPI (e.g., ReadTheDocs)
try:
    from simsopt.util import comm_world
except (ImportError, RuntimeError):
    # ImportError: simsopt not installed
    # RuntimeError: mpi4py installed but MPI library not available
    comm_world = None

def _is_proc0() -> bool:
    """Check if this is rank 0 (or non-MPI environment)."""
    return comm_world is None or not hasattr(comm_world, 'rank') or comm_world.rank == 0


def _get_version_info() -> dict:
    """Get version information for reproducibility tracking.
    
    Returns a dict with:
    - stellcoilbench_commit: git commit hash of stellcoilbench repo
    - simsopt_version: simsopt package version
    - simsopt_git_info: simsopt git info if installed from source (branch, commit)
    """
    info = {}
    
    # Get stellcoilbench git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["stellcoilbench_commit"] = result.stdout.strip()
        
        # Also get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["stellcoilbench_branch"] = result.stdout.strip()
    except Exception:
        info["stellcoilbench_commit"] = "unknown"
    
    # Get simsopt version
    try:
        import simsopt
        version = getattr(simsopt, "__version__", "unknown")
        info["simsopt_version"] = version
        
        # Try to extract git commit from version string (e.g., "0.1.dev5270+gee055f063")
        # The format is: version.devN+gCOMMIT where COMMIT is the abbreviated hash
        if "+g" in version:
            # Extract commit hash after +g
            commit_part = version.split("+g")[-1]
            # Remove any additional suffixes (e.g., .dirty)
            commit_hash = commit_part.split(".")[0]
            info["simsopt_commit"] = commit_hash
        
        # Try to get simsopt git info if installed from source (editable install)
        simsopt_file = getattr(simsopt, "__file__", None)
        if simsopt_file is None:
            return info
        simsopt_path = Path(simsopt_file).parent
        # Check both parent and grandparent for .git (handles different install layouts)
        for parent in [simsopt_path.parent, simsopt_path.parent.parent]:
            simsopt_git_dir = parent / ".git"
            if simsopt_git_dir.exists():
                # Installed from source - get git info
                try:
                    result = subprocess.run(
                        ["git", "-C", str(parent), "rev-parse", "HEAD"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        info["simsopt_commit"] = result.stdout.strip()
                    
                    result = subprocess.run(
                        ["git", "-C", str(parent), "rev-parse", "--abbrev-ref", "HEAD"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        info["simsopt_branch"] = result.stdout.strip()
                    
                    # Get remote URL to identify fork
                    result = subprocess.run(
                        ["git", "-C", str(parent), "remote", "get-url", "origin"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        info["simsopt_remote"] = result.stdout.strip()
                    break  # Found git info, stop searching
                except Exception:
                    pass
    except ImportError:
        info["simsopt_version"] = "not installed"
    
    return info


# Standard reactor-scale reference parameters (ARIES-CS)
REACTOR_REFERENCE = {
    "major_radius": 7.5,  # meters (ARIES-CS baseline)
    "B_field": 5.7,  # Tesla (ARIES-CS on-axis field)
    "description": "ARIES-CS reactor-scale reference"
}


############################################################################
# REBCO critical-current model and Stellaris-style winding-pack parameters
# -------------------------------------------------------------------------
# Following Lion et al., "Stellaris: A high-field quasi-isodynamic
# stellarator for a prototypical fusion power plant", Fusion Engineering
# and Design 214 (2025) 114868, Table 8 and Section 2.9.
############################################################################

# Stellaris-style winding-pack constants
STELLARIS_T_OP = 20.0          # Operating temperature [K]
STELLARIS_ETA = 0.80           # Utilization cap  j_op / j_crit ≤ η
STELLARIS_I_LEAD_MAX = 50e3    # Current-lead limit [A]  (50 kA)
STELLARIS_A_TURN = 400e-6      # Turn cross-section area [m²]  (20 mm × 20 mm)
STELLARIS_A_HTS = 36e-6        # HTS tape-stack area [m²]  (6 mm × 6 mm)

# Winding-pack self-field enhancement factor.
# B_peak ≈ f_wp × B_external.  For typical stellarator winding packs with
# hundreds of turns the self-field adds ~20-40% to the background field at
# the inner edge of the pack (validated against Stellaris Table 8).
WP_B_ENHANCEMENT = 1.3


def _rebco_jc_tape_stack(B: float, T_op: float = 20.0) -> float:
    """Engineering critical-current density of an optimally-aligned REBCO
    tape stack at temperature *T_op* and peak field *B* [T].

    Returns j_crit in **A/m²** (SI).

    The model is a Kim-like parametrization calibrated to the Stellaris
    magneto-angular Jc data at 20 K with field-aligned tapes
    (Lion et al., FED 214, 2025, Table 8 / Fig. 42).

    Model:  j_crit(B) = C₀ / (1 + (B/B₀)^α)

    At T = 20 K the fitted constants are:
        C₀ = 5.0 × 10⁹  A/m²   (≈ 5000 A/mm²  at self-field)
        B₀ = 18.14 T
        α  = 0.902

    Validation against Stellaris Table 8 (tape-stack j_op/j_crit):
        B=20 T → j_crit ≈ 2450 A/mm²,  B=25 T → j_crit ≈ 2200 A/mm²

    For temperatures other than 20 K a simple linear scaling
    Jc(T) ∝ (1 − T/Tc) with Tc = 92 K is applied.
    """
    # --- Fitted parameters at T_ref = 20 K ---
    T_REF = 20.0
    T_C = 92.0          # REBCO critical temperature [K]
    C0 = 5.0e9           # A/m²  (= 5000 A/mm²)
    B0 = 18.14           # T
    ALPHA = 0.902

    if B < 0:
        B = 0.0

    jc_20K = C0 / (1.0 + (B / B0) ** ALPHA)

    # Temperature correction (linear in reduced temperature)
    if abs(T_op - T_REF) > 0.01:
        jc = jc_20K * (1.0 - T_op / T_C) / (1.0 - T_REF / T_C)
    else:
        jc = jc_20K

    return max(jc, 0.0)


def _compute_N_turns_critical_current(
    per_coil_forces: list,
    per_coil_currents: list | None,
    per_coil_lengths: list | None,
    L_scale: float,
    B_scale: float,
    target_B: float,
    *,
    T_op: float = STELLARIS_T_OP,
    eta: float = STELLARIS_ETA,
    I_lead_max: float = STELLARIS_I_LEAD_MAX,
    A_HTS: float = STELLARIS_A_HTS,
    wp_enhancement: float = WP_B_ENHANCEMENT,
) -> dict:
    """Compute per-coil turn counts based on critical-current density.

    Following the Stellaris winding-pack model (Lion et al., FED 2025):
      - 20 K operating temperature
      - 80 % utilization margin (η = j_op / j_crit)
      - 50 kA current-lead limit
      - 6 mm × 6 mm HTS tape-stack cross-section

    Algorithm for each coil *i*:

    1. **Required ampere-turns** at reactor scale::

           NI_i = I_device_i × B_scale × L_scale

       where ``I_device_i`` is the simsopt single-turn current.  If
       per-coil currents are unavailable, *NI* is estimated from
       the force data:  ``NI_i ∝ sqrt(F_i / (μ₀ / (4π L_i)))``.

    2. **Peak field estimate** at the conductor::

           B_ext_i  = (F/L)_device_i / I_device_i × B_scale
           B_peak_i = B_ext_i × wp_enhancement

       The winding-pack self-field enhancement (*wp_enhancement*, default
       1.3) accounts for the additional field produced by the multi-turn
       pack itself.

    3. **Critical current of the HTS cable** (tape-stack area A_HTS)::

           Ic_cable = j_crit(B_peak, T_op) × A_HTS

    4. **Operating current per turn** (lead-limited or tape-limited)::

           I_turn = min(I_lead_max, η × Ic_cable)

    5. **Number of turns**::

           N_turns_jc_i = ⌈ NI_i / I_turn_i ⌉

    Parameters
    ----------
    per_coil_forces : list[float]
        Device-scale maximum force/length per base coil [N/m].
    per_coil_currents : list[float] | None
        Device-scale current per base coil [A].  If *None*, currents
        are estimated from force data.
    per_coil_lengths : list[float] | None
        Device-scale centreline length per base coil [m].
    L_scale, B_scale : float
        Geometric and magnetic-field scaling ratios (reactor / device).
    target_B : float
        Device-scale target B-field [T].

    Returns
    -------
    dict with keys:
        N_turns_jc : list[int]
            Per-coil turn count from Jc requirements.
        NI_reactor : list[float]
            Required ampere-turns per coil at reactor scale [A].
        I_turn : list[float]
            Operating current per turn [A].
        B_peak_estimate : list[float]
            Estimated peak conductor field [T].
        jc_tape_stack : list[float]
            Tape-stack j_crit at the peak field [A/m²].
        Ic_cable : list[float]
            Critical current of the HTS cable [A].
        model_params : dict
            Constants used (T_op, eta, I_lead_max, A_HTS, wp_enhancement).
    """
    n_coils = len(per_coil_forces)

    # ----- per-coil device currents -----
    if per_coil_currents is not None and len(per_coil_currents) == n_coils:
        I_dev = [abs(float(c)) for c in per_coil_currents]
    else:
        # Fallback: estimate I from F/L and B_device.
        # F/L ≈ I × B_ext and B_ext ≈ target_B at the coil location
        # → I ≈ (F/L) / target_B   (rough but usable).
        I_dev = [abs(float(f)) / max(target_B, 0.01) for f in per_coil_forces]

    # ----- compute per-coil quantities -----
    NI_list: list[float] = []
    I_turn_list: list[float] = []
    B_peak_list: list[float] = []
    jc_list: list[float] = []
    Ic_list: list[float] = []
    N_turns_jc: list[int] = []

    for i in range(n_coils):
        # 1. Required ampere-turns at reactor scale
        NI_i = I_dev[i] * B_scale * L_scale
        NI_list.append(float(NI_i))

        # 2. Peak field estimate
        #    B_ext ≈ (F/L)_device / I_device × B_scale
        if I_dev[i] > 0:
            B_ext_i = (per_coil_forces[i] / I_dev[i]) * B_scale
        else:
            B_ext_i = target_B * B_scale  # fallback to on-axis field
        B_peak_i = B_ext_i * wp_enhancement
        B_peak_list.append(float(B_peak_i))

        # 3. Critical current
        jc_i = _rebco_jc_tape_stack(B_peak_i, T_op)
        jc_list.append(float(jc_i))
        Ic_cable_i = jc_i * A_HTS
        Ic_list.append(float(Ic_cable_i))

        # 4. Operating current per turn
        I_turn_i = min(I_lead_max, eta * Ic_cable_i)
        I_turn_list.append(float(I_turn_i))

        # 5. Number of turns
        if I_turn_i > 0:
            n_i = max(1, int(np.ceil(NI_i / I_turn_i)))
        else:
            n_i = 1  # degenerate case
        N_turns_jc.append(n_i)

    return {
        "N_turns_jc": N_turns_jc,
        "NI_reactor": NI_list,
        "I_turn": I_turn_list,
        "B_peak_estimate": B_peak_list,
        "jc_tape_stack": jc_list,
        "Ic_cable": Ic_list,
        "model_params": {
            "T_op_K": T_op,
            "eta": eta,
            "I_lead_max_A": I_lead_max,
            "A_HTS_m2": A_HTS,
            "wp_enhancement": wp_enhancement,
        },
    }


def _compute_reactor_scale_metrics(metrics: dict, case_cfg=None) -> dict:
    """Convert final device-scale metrics to reactor-scale equivalents.

    Scaling relationships (L = L_reactor/L_device, B = B_reactor/B_device):

    - Lengths [m] (d_cc, d_cs, total_length): × L
    - Curvature [1/m] (κ): × 1/L
    - Mean squared curvature [1/m²]: × 1/L²
    - Force/length [N/m → MN/m]: × B²L / 1e6
    - Torque/length [N → MN]: × B²L² / 1e6
    - Arclength variation [m²]: × L²
    - SquaredFlux [T²m²]: × B²L²
    - Normalised quantities (B·n/|B|, linking_number): no scaling

    Also computes per-coil quantities:

    - **N_turns_per_coil** = max(N_force, N_jc) — force-based and REBCO-Jc-
      based turn counts (see ``_compute_N_turns_critical_current``).
    - **winding_pack_width_per_coil** — finite-build side length
      w = sqrt(N_turns) × 20 mm (Stellaris geometry).
    - **finite_build_cc_clearance** — d_cc_min − w_max.  Negative values
      indicate the finite-build winding packs would physically overlap.
    - **total_superconductor_length_km** — Σ_i N_turns_i × length_i / 1000.

    Returns
    -------
    dict
        Reactor-scale metrics, scaling factors, and derived winding-pack data.
    """
    reactor_metrics: dict = {
        "reference": REACTOR_REFERENCE.copy(),
    }
    
    # Get device parameters from metrics
    target_B = metrics.get("target_B_field", None)
    
    # Get the actual device major radius [m] and minor radius [m].
    # NOTE: _cached_thresholds["a0"] = ARIES_CS_MINOR_RADIUS / minor_radius (matches vmec_RZ_scale).
    cached = metrics.get("_cached_thresholds", {})
    major_radius = cached.get("major_radius", None)
    minor_radius = cached.get("minor_radius", None)
    
    # Fallback: try to compute from the a0 scale factor (or legacy R0 key)
    if major_radius is None:
        a0 = cached.get("a0", cached.get("R0", None))
        if a0 is not None and a0 != 0 and minor_radius is None:
            # Legacy: R0 = 10/major_radius → major_radius = 10/R0
            major_radius = 10.0 / a0
    
    # If not in metrics, try to load the plasma surface to get major radius
    if major_radius is None and case_cfg is not None:
        try:
            from stellcoilbench.config_scheme import CaseConfig
            if isinstance(case_cfg, CaseConfig):
                surface_name = case_cfg.surface_params.get("surface", "")
            elif isinstance(case_cfg, dict):
                surface_name = case_cfg.get("surface_params", {}).get("surface", "")
            else:
                surface_name = ""
            if surface_name:
                # Search for surface file in plasma_surfaces/
                candidates = [
                    Path("plasma_surfaces") / surface_name,
                    Path.cwd() / "plasma_surfaces" / surface_name,
                ]
                surface_path = None
                for p in candidates:
                    if p.exists():
                        surface_path = p
                        break
                # Case-insensitive fallback
                if surface_path is None:
                    ps_dir = Path("plasma_surfaces")
                    if not ps_dir.exists():
                        ps_dir = Path.cwd() / "plasma_surfaces"
                    if ps_dir.exists():
                        for f in ps_dir.iterdir():
                            if f.name.lower() == surface_name.lower():
                                surface_path = f
                                break
                if surface_path is not None:
                    from simsopt.geo import SurfaceRZFourier
                    s = SurfaceRZFourier.from_vmec_input(
                        str(surface_path), range="half period", nphi=16, ntheta=16,
                    )
                    major_radius = float(s.major_radius())
        except Exception:
            pass
    
    if major_radius is None or target_B is None:
        reactor_metrics["error"] = "Could not determine device scale parameters"
        return reactor_metrics
    
    # Compute scaling factors (use minor radius when available, consistent with vmec_RZ_scale)
    from stellcoilbench.coil_optimization import ARIES_CS_MINOR_RADIUS
    if minor_radius is not None and minor_radius > 0:
        L_scale = ARIES_CS_MINOR_RADIUS / minor_radius  # Same as vmec_RZ_scale
    else:
        L_scale = REACTOR_REFERENCE["major_radius"] / major_radius  # Legacy: major radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B  # B-field scale factor
    
    reactor_metrics["scaling_factors"] = {
        "length_scale": float(L_scale),
        "B_field_scale": float(B_scale),
        "device_major_radius": float(major_radius),
        "device_target_B": float(target_B),
    }
    
    # Scale length quantities (multiply by L_scale)
    length_metrics = [
        "final_min_cs_separation",
        "final_min_cc_separation", 
        "final_total_length",
    ]
    for key in length_metrics:
        if key in metrics:
            reactor_key = key.replace("final_", "reactor_scale_")
            reactor_metrics[reactor_key] = float(metrics[key]) * L_scale
    
    # Scale curvature quantities (divide by L_scale, since κ ~ 1/L)
    curvature_metrics = [
        "final_max_curvature",
        "final_average_curvature",
        "final_mean_squared_curvature",  # This scales as 1/L²
    ]
    for key in curvature_metrics:
        if key in metrics:
            reactor_key = key.replace("final_", "reactor_scale_")
            if "mean_squared" in key:
                # MSC scales as 1/L²
                reactor_metrics[reactor_key] = float(metrics[key]) / (L_scale ** 2)
            else:
                reactor_metrics[reactor_key] = float(metrics[key]) / L_scale
    
    # Scale force-per-length quantities [N/m]: dF/dℓ = I × B
    # I ∝ B·L (current scales to maintain field), B_ext ∝ B → dF/dℓ ∝ B²·L
    # Report in MN/m (divide by 1e6).
    force_scale = (B_scale ** 2) * L_scale
    force_metrics = [
        "final_max_max_coil_force",
        "final_avg_max_coil_force",
    ]
    for key in force_metrics:
        if key in metrics:
            reactor_key = key.replace("final_", "reactor_scale_")
            reactor_metrics[reactor_key] = float(metrics[key]) * force_scale / 1e6  # MN/m
    
    # Scale torque-per-length quantities [N]: dτ/dℓ = r × dF/dℓ
    # r ∝ L (lever arm), dF/dℓ ∝ B²·L → dτ/dℓ ∝ B²·L²
    # Report in MN (divide by 1e6).
    torque_scale = (B_scale ** 2) * (L_scale ** 2)
    torque_metrics = [
        "final_max_max_coil_torque",
        "final_avg_max_coil_torque",
    ]
    for key in torque_metrics:
        if key in metrics:
            reactor_key = key.replace("final_", "reactor_scale_")
            reactor_metrics[reactor_key] = float(metrics[key]) * torque_scale / 1e6  # MN
    
    # Per-coil reactor-scale force [MN/m] and N_turns required.
    #
    # Two independent requirements set N_turns for each coil:
    #
    # (A) **Force limit** — simsopt models single-turn coils carrying total
    #     current I.  With N turns each carrying I/N the Lorentz force per
    #     turn drops by a factor of N.  Therefore:
    #         N_turns_force_i = ⌈ F_reactor_i / F_limit ⌉
    #
    # (B) **Critical-current density** — following the Stellaris winding-pack
    #     model (Lion et al., FED 214, 2025):  at 20 K with 80 % utilisation,
    #     50 kA lead limit, and a 6 mm × 6 mm REBCO tape stack, the number
    #     of turns is:
    #         N_turns_jc_i = ⌈ NI_reactor_i / I_turn_i ⌉
    #     where NI is the total reactor-scale ampere-turns and I_turn is
    #     constrained by the REBCO Jc at the estimated peak conductor field.
    #
    # The reported N_turns is the element-wise maximum:
    #     N_turns_i = max(N_turns_force_i, N_turns_jc_i)
    #
    FORCE_LIMIT_MN_PER_M = 0.5  # MN/m engineering limit
    per_coil_forces = metrics.get("final_max_force_per_coil")
    if per_coil_forces is not None and len(per_coil_forces) > 0:
        reactor_force_per_coil = [f * force_scale / 1e6 for f in per_coil_forces]  # MN/m
        n_turns_force = [max(1, int(np.ceil(f / FORCE_LIMIT_MN_PER_M)))
                         for f in reactor_force_per_coil]

        # --- Critical-current–based turn count ---
        per_coil_currents = metrics.get("final_current_per_coil")
        per_coil_lengths = metrics.get("final_length_per_coil")
        jc_result = _compute_N_turns_critical_current(
            per_coil_forces=per_coil_forces,
            per_coil_currents=per_coil_currents,
            per_coil_lengths=per_coil_lengths,
            L_scale=L_scale,
            B_scale=B_scale,
            target_B=target_B,
        )
        n_turns_jc = jc_result["N_turns_jc"]

        # --- Element-wise maximum of the two requirements ---
        n_turns_per_coil = [max(nf, nj) for nf, nj in zip(n_turns_force, n_turns_jc)]

        reactor_metrics["reactor_scale_force_per_coil_MN_per_m"] = reactor_force_per_coil
        reactor_metrics["N_turns_per_coil"] = n_turns_per_coil
        reactor_metrics["N_turns_force"] = n_turns_force
        reactor_metrics["N_turns_jc"] = n_turns_jc
        reactor_metrics["force_limit_MN_per_m"] = FORCE_LIMIT_MN_PER_M
        reactor_metrics["jc_model"] = {
            "NI_reactor": jc_result["NI_reactor"],
            "I_turn": jc_result["I_turn"],
            "B_peak_estimate": jc_result["B_peak_estimate"],
            "jc_tape_stack_A_per_m2": jc_result["jc_tape_stack"],
            "Ic_cable_A": jc_result["Ic_cable"],
            "params": jc_result["model_params"],
        }

        # --- Finite-build (winding-pack) estimate ---
        # Stellaris turn geometry: each turn occupies A_turn = 20 mm × 20 mm
        # = 400 mm² = 400e-6 m².  This includes HTS tape stack, copper
        # stabiliser, solder, structural steel jacket, and helium cooling
        # channel (Lion et al., FED 2025, Table 7).
        #
        # For a square winding pack with N turns:
        #   side_length = sqrt(N) × 20 mm
        #
        # Validation against Stellaris Table 8:
        #   Coil 0: N=324 → w = sqrt(324)×20 mm = 18×20 = 360 mm  ✓
        #   Coil 5: N=225 → w = sqrt(225)×20 mm = 15×20 = 300 mm  ✓
        turn_side_m = np.sqrt(STELLARIS_A_TURN)  # = 0.020 m = 20 mm
        wp_widths = [float(np.sqrt(n) * turn_side_m) for n in n_turns_per_coil]
        reactor_metrics["winding_pack_width_per_coil"] = wp_widths
        max_wp = float(max(wp_widths)) if wp_widths else 0.0
        reactor_metrics["max_winding_pack_width"] = max_wp

        # --- Finite-build coil-coil clearance ---
        # simsopt's CurveCurveDistance measures centreline-to-centreline
        # distance.  Each coil's winding pack extends w/2 from the
        # centreline, so for two packs not to intersect we need:
        #
        #     d_cc > w_i/2 + w_j/2
        #
        # We don't know which pair is the closest, so conservatively use
        # the largest winding pack for both:
        #
        #     d_cc_min > w_max/2 + w_max/2 = w_max
        #
        # The *clearance* is the remaining gap after accounting for the
        # finite build:
        #
        #     clearance = d_cc_min − w_max
        #
        # Negative clearance → winding packs would physically overlap.
        d_cc_rs = reactor_metrics.get("reactor_scale_min_cc_separation")
        if d_cc_rs is not None and max_wp > 0:
            reactor_metrics["finite_build_cc_clearance"] = float(d_cc_rs - max_wp)

        # --- Per-turn force and torque after multi-turn winding pack ---
        # With N_turns turns each carrying I/N, the force per turn on
        # coil i is F_reactor_i / N_turns_i (force scales linearly with
        # the current carried by a single turn).  Similarly for torque.
        # These are the engineering-relevant quantities: the structural
        # load on each individual turn of the winding pack.
        per_turn_forces = [f / n for f, n in zip(reactor_force_per_coil, n_turns_per_coil)]
        reactor_metrics["per_turn_max_force"] = float(max(per_turn_forces))  # MN/m

        per_coil_torques = metrics.get("final_max_torque_per_coil")
        if per_coil_torques is not None and len(per_coil_torques) == len(n_turns_per_coil):
            reactor_torque_per_coil = [t * torque_scale / 1e6 for t in per_coil_torques]  # MN
            per_turn_torques = [t / n for t, n in zip(reactor_torque_per_coil, n_turns_per_coil)]
            reactor_metrics["per_turn_max_torque"] = float(max(per_turn_torques))  # MN
        elif "reactor_scale_max_max_coil_torque" in reactor_metrics:
            # Fallback: divide the overall max torque by the min N_turns
            # (conservative — actual per-turn torque could be lower)
            max_tau = reactor_metrics["reactor_scale_max_max_coil_torque"]
            min_n = min(n_turns_per_coil)
            reactor_metrics["per_turn_max_torque"] = float(max_tau / min_n)  # MN

        # Total superconductor length [km]:
        #   Σ_i  N_turns_i × reactor_scale_length_i
        # where reactor_scale_length_i = device_length_i × L_scale
        if per_coil_lengths is not None and len(per_coil_lengths) == len(n_turns_per_coil):
            reactor_lengths = [ln * L_scale for ln in per_coil_lengths]
            total_sc_km = sum(n * ln for n, ln in zip(n_turns_per_coil, reactor_lengths)) / 1e3
            reactor_metrics["total_superconductor_length_km"] = float(total_sc_km)
        elif "final_total_length" in metrics:
            # Fallback: assume uniform coil length (total_length / num_coils)
            num_coils = len(n_turns_per_coil)
            avg_len = float(metrics["final_total_length"]) * L_scale / num_coils
            total_sc_km = sum(n * avg_len for n in n_turns_per_coil) / 1e3
            reactor_metrics["total_superconductor_length_km"] = float(total_sc_km)

    # Arclength variation scales as L² (since it's variance of length)
    if "final_arclength_variation" in metrics:
        reactor_metrics["reactor_scale_arclength_variation"] = float(metrics["final_arclength_variation"]) * (L_scale ** 2)
    
    # SquaredFlux [T²m²]: J = ½∫(B·n̂)²dS where n̂ is the unit normal
    # (B·n̂)² [T²] scales as B², surface element dS [m²] scales as L²
    # → J scales as B²·L²
    if "final_squared_flux" in metrics:
        flux_scale = (B_scale ** 2) * (L_scale ** 2)
        reactor_metrics["reactor_scale_squared_flux"] = float(metrics["final_squared_flux"]) * flux_scale
    
    # Dimensionless quantities - no scaling needed
    # (BdotN_over_B, linking_number, etc. are already normalized)
    
    return reactor_metrics


def _extract_surface_name(surface_file: str) -> str:
    """Extract a clean surface name from a surface filename.

    Strips common prefixes (``input.``, ``wout.``) and file extensions
    (e.g. ``.focus``) so that ``"input.LandremanPaul2021_QA"`` becomes
    ``"LandremanPaul2021_QA"``.
    """
    name = Path(surface_file).name
    for prefix in ("input.", "wout."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if "." in name:
        name = name.split(".", 1)[0]
    return name


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and arrays."""
    def default(self, o):
        # Handle numpy integer types
        if isinstance(o, np.integer):
            return int(o)
        # Handle numpy floating point types
        elif isinstance(o, np.floating):
            return float(o)
        # Handle numpy arrays
        elif isinstance(o, np.ndarray):
            return o.tolist()
        # Handle numpy boolean
        elif isinstance(o, np.bool_):
            return bool(o)
        # Handle jax/jaxlib arrays and other array-like objects
        elif hasattr(o, '__array__'):
            try:
                return np.asarray(o).tolist()
            except (TypeError, ValueError):
                pass
        # Handle simsopt objects (SurfaceRZFourier, Vmec, etc.) by converting to string
        # These are not JSON serializable but we want to include them in results
        elif hasattr(o, '__module__') and 'simsopt' in str(o.__module__):
            return str(o)
        return super().default(o)


def _print_submission_summary(submission: dict) -> None:
    """Print a clearly formatted summary of the submission results."""
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("  OPTIMIZATION RESULTS SUMMARY")
    typer.echo("=" * 60)
    typer.echo("")
    meta = submission.get("metadata", {})
    typer.echo("  Metadata:")
    for k, v in meta.items():
        typer.echo(f"    {k}: {v}")
    typer.echo("")
    metrics = submission.get("metrics", {})
    # Exclude 'timing' from metrics - shown in Timing section below
    metrics_no_timing = {k: v for k, v in metrics.items() if k != "timing"}
    if metrics_no_timing:
        typer.echo("  Metrics:")
        for k, v in sorted(metrics_no_timing.items()):
            if isinstance(v, (int, float)):
                av = abs(float(v))
                fmt = f"{v:.4e}" if (av and (av < 1e-2 or av >= 1e4)) else str(v)
                typer.echo(f"    {k}: {fmt}")
            else:
                typer.echo(f"    {k}: {v}")
        typer.echo("")
    # Coil optimization timing (merged from COIL OPTIMIZATION TIMING SUMMARY)
    timing = metrics.get("timing") or submission.get("timing")
    if timing:
        coil_opt_keys = ['coil_initialization', 'biotsavart_setup', 'objective_setup', 'coil_optimization', 'save_and_metrics']
        total_coil_opt = sum(timing.get(k, 0) for k in coil_opt_keys)
        if total_coil_opt > 0:
            typer.echo("  Timing:")
            for key in coil_opt_keys:
                if key in timing:
                    typer.echo(f"    {key}: {timing[key]:.2f}s")
            typer.echo(f"    {'─' * 30}")
            typer.echo(f"    Total coil optimization: {total_coil_opt:.2f}s")
            typer.echo("")
    typer.echo("=" * 60)
    typer.echo("")


app = typer.Typer(help="StellCoilBench: benchmarking framework for stellarator coil optimization.")


def _detect_github_username() -> str:
    """
    Try to detect GitHub username from remote URL or environment variables.
    Returns empty string if not found.
    
    Note: git config user.name returns the display name, not the GitHub username,
    so we prioritize extracting from the remote URL.
    """
    try:
        # Try to get from remote URL first (most reliable for GitHub username)
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract username from common GitHub URL patterns
            if "github.com" in url:
                # Handle https://github.com/user/repo format
                if url.startswith("https://") or url.startswith("http://"):
                    parts = url.replace(".git", "").split("/")
                    # URL format: https://github.com/user/repo
                    # parts: ['https:', '', 'github.com', 'user', 'repo']
                    if len(parts) >= 4 and parts[2] == "github.com":
                        username = parts[3]
                        if username and username != "github.com":
                            return username
                # Handle git@github.com:user/repo format
                elif url.startswith("git@"):
                    # URL format: git@github.com:user/repo
                    # Split on ':' to get the part after github.com:
                    if ":" in url:
                        after_colon = url.split(":", 1)[1]
                        parts = after_colon.replace(".git", "").split("/")
                        if len(parts) >= 1:
                            username = parts[0]
                            if username:
                                return username
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Try environment variable (useful in CI)
    import os
    github_user = os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USER")
    if github_user:
        return github_user
    
    return ""


def _zip_submission_directory(submission_dir: Path) -> Path:
    """
    Zip the submission files (excluding PDFs and post-processing outputs) into all_files.zip.
    
    Creates a zip file named "all_files.zip" inside the submission directory.
    PDF files and post-processing outputs (QFM surface, Poincaré plots, VMEC plots, etc.)
    are kept in the directory alongside the zip file.
    
    Parameters
    ----------
    submission_dir: Path
        Directory containing submission files to zip.
    
    Returns
    -------
    Path
        Path to the created zip file (submission_dir / "all_files.zip").
    """
    submission_dir = Path(submission_dir)
    
    if not submission_dir.exists() or not submission_dir.is_dir():
        typer.echo(f"Warning: Submission directory does not exist: {submission_dir}")
        return submission_dir / "all_files.zip"
    
    # Create zip file inside the submission directory
    zip_filename = "all_files.zip"
    zip_path = submission_dir / zip_filename
    
    # Find all files in the submission directory
    files_to_zip = []
    pdf_files_to_keep = []
    for file_path in submission_dir.rglob("*"):
        if file_path.is_file():
            # PDF plots stay in the DATE directory and are NOT zipped
            if file_path.suffix.lower() == ".pdf":
                pdf_files_to_keep.append(file_path)
            else:
                files_to_zip.append(file_path)
    
    if not files_to_zip:
        typer.echo(f"Warning: No files found in {submission_dir} to zip")
        return zip_path
    
    # Note: PDF files are kept in the DATE directory and not included in the zip

    # Create zip file with remaining files (excluding PDFs)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            # Add file to zip with relative path from submission_dir
            arcname = file_path.relative_to(submission_dir)
            zipf.write(file_path, arcname=arcname)
    
    # Keep post-processing files in addition to PDFs:
    # - PDF files (bn_error plots)
    # - Post-processing outputs: .vts (QFM surface), .png (plots), post_processing_results.json
    # Note: finite_build_coils.vtk / finite_build_coils_parastell.vtk NOT kept outside zip
    post_processing_patterns = [
        'qfm_surface',
        'poincare',
        'boozer',
        'iota',
        'quasisymmetry',
        'post_processing_results',
        'simple_loss_fraction',  # SIMPLE fast particle tracing plot
        'simple',  # Also match any file with 'simple' in name
    ]
    
    # Remove files that should be zipped, but keep PDFs and post-processing files
    for file_path in files_to_zip:
        # Keep if it's a post-processing file (check filename patterns)
        is_post_processing_file = any(
            pattern.lower() in file_path.name.lower() 
            for pattern in post_processing_patterns
        ) and file_path.suffix.lower() in {'.vts', '.vtk', '.png', '.json'}
        
        if not is_post_processing_file:
            try:
                file_path.unlink()
                # Try to remove parent directory if it's empty (but not the submission_dir itself)
                parent = file_path.parent
                if parent != submission_dir and parent.exists() and not any(parent.iterdir()):
                    try:
                        parent.rmdir()
                    except (OSError, FileNotFoundError):
                        pass  # Directory not empty or other error, skip
            except (OSError, FileNotFoundError) as e:
                typer.echo(f"Warning: Failed to remove {file_path}: {e}")
    
    return zip_path


def _detect_hardware() -> str:
    """
    Detect hardware information (CPU, GPU, memory).
    Returns a formatted string describing the hardware.
    """
    parts = []
    
    # CPU info
    try:
        cpu_info = platform.processor() or platform.machine()
        if cpu_info:
            parts.append(f"CPU: {cpu_info}")
    except Exception:
        pass
    
    # Try to get more detailed CPU info
    try:
        if platform.system() == "Linux":
            result = subprocess.run(
                ["lscpu"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Model name:" in line:
                        cpu_name = line.split("Model name:")[-1].strip()
                        if cpu_name:
                            parts[0] = f"CPU: {cpu_name}"
                            break
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts[0] = f"CPU: {result.stdout.strip()}"
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # GPU info (NVIDIA)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_names = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            if gpu_names:
                gpu_str = ", ".join(gpu_names)
                parts.append(f"GPU: {gpu_str}")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Memory info (optional, requires psutil)
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        mem_gb = mem.total / (1024**3)
        parts.append(f"RAM: {mem_gb:.1f}GB")
    except (ImportError, Exception):
        # psutil not available or error, skip
        pass
    
    return " | ".join(parts) if parts else platform.platform()

@app.command("update-db")
def update_db_cmd(
    submissions_dir: Path = typer.Argument(
        Path("submissions"),
        help="Directory containing per-method submissions (results.json files).",
    ),
    docs_dir: Path = typer.Option(
        Path("docs"),
        "--docs-dir",
        help="Directory where docs/leaderboards/ leaderboards and leaderboard.json are written.",
    ),
) -> None:
    """
    Rebuild the on-repo 'database' of submissions and leaderboards.

    This scans submissions_dir for results.json files, aggregates them into
    docs/leaderboard.json, and writes per-surface leaderboards in docs/leaderboards/.
    """
    from .update_db import update_database
    repo_root = Path.cwd()
    update_database(
        repo_root=repo_root,
        submissions_root=submissions_dir,
        docs_dir=docs_dir,
    )
    typer.echo(f"Updated leaderboard.json and surface leaderboards in {docs_dir / 'leaderboards'}")


@app.command("submit-case")
def submit_case(
    case_path: Path = typer.Argument(
        ...,
        help="Path to case.yaml file (e.g., cases/case.yaml).",
    ),
    method_name: Optional[str] = typer.Option(
        None,
        "--method-name",
        "-m",
        help="Name of your optimization method (optional, stored in metadata).",
    ),
    notes: str = typer.Option("", "--notes", "-n", help="Additional notes."),
    submissions_dir: Path = typer.Option(
        Path("submissions"),
        "--submissions-dir",
        help="Directory where submission results.json will be written.",
    ),
    run_vmec: bool = typer.Option(
        False,
        "--run-vmec/--no-vmec",
        help="Run QFM and VMEC equilibrium calculation (expensive, disabled by default).",
    ),
    run_simple: bool = typer.Option(
        False,
        "--run-simple/--no-simple",
        help="Run SIMPLE fast particle tracing (requires --run-vmec, expensive).",
    ),
    plot_poincare: bool = typer.Option(
        False,
        "--plot-poincare/--no-plot-poincare",
        help="Generate Poincaré plot (disabled by default; expensive).",
    ),
    plot_finite_build: bool = typer.Option(
        False,
        "--plot-finite-build/--no-plot-finite-build",
        help="Generate finite-build coil VTK (rectangular cross-section swept along centerline). Output: finite_build_coils.vtk (and finite_build_coils_parastell.vtk if ParaStell available).",
    ),
    finite_build_width: Optional[float] = typer.Option(
        None,
        "--finite-build-width",
        help="Cross-section width [m] for finite-build coils. Default: 5 cm.",
    ),
    finite_build_height: Optional[float] = typer.Option(
        None,
        "--finite-build-height",
        help="Cross-section height [m] for finite-build coils. Default: 5 cm.",
    ),
) -> None:
    """
    Run a case and generate a submission results.json file.
    
    This command:
    1. Loads case.yaml from cases/
    2. Runs the coil optimization
    3. Evaluates the results (B·n, Poincaré plot by default)
    4. Optionally runs QFM/VMEC/SIMPLE (with --run-vmec --run-simple)
    5. Optionally generates finite-build coil VTK (with --plot-finite-build)
    6. Generates results.json in submissions/<surface>/<username>/<case>/<datetime>/ with metadata and metrics
    
    Directory structure: submissions/<github_username>/<MM-DD-YYYY_HH-MM>/all_files.zip
    GitHub username and hardware are auto-detected if not provided.
    
    Examples:
        stellcoilbench submit-case cases/case.yaml
        stellcoilbench submit-case cases/case.yaml --run-vmec
        stellcoilbench submit-case cases/case.yaml --run-vmec --run-simple
        stellcoilbench submit-case cases/case.yaml --plot-finite-build
        stellcoilbench submit-case cases/case.yaml --plot-finite-build --finite-build-width 0.05 --finite-build-height 0.05
    """
    from .coil_optimization import optimize_coils
    from .evaluate import load_case_config, evaluate_case

    # Auto-detect GitHub username for directory structure
    github_username = _detect_github_username()
    if not github_username:
        github_username = "unknown_user"
        typer.echo("Warning: Could not auto-detect GitHub username. Using 'unknown_user'.")
        typer.echo("Use --contact to specify your GitHub username.")
    else:
        typer.echo(f"Using GitHub username: {github_username}")

    # Auto-detect contact (use GitHub username)
    contact = github_username
    typer.echo(f"Auto-detected contact: {contact}")

    # Auto-detect hardware
    hardware = _detect_hardware()
    if not hardware:
        hardware = "Unknown hardware"
        typer.echo("Warning: Could not auto-detect hardware.")
    else:
        typer.echo(f"Auto-detected hardware: {hardware}")

    # Load case configuration
    case_cfg = load_case_config(case_path)

    # Extract surface name from case config
    surface_file = case_cfg.surface_params.get("surface", "")
    if not surface_file:
        raise ValueError("case.yaml must specify surface_params.surface")
    surface_name = _extract_surface_name(surface_file)
    
    # 3) Build submission directory first (needed for output_dir)
    now = datetime.now()
    run_date = now.isoformat()
    datetime_str = now.strftime("%m-%d-%Y_%H-%M")  # Format: MM-DD-YYYY_HH-MM
    
    # Get case name from case file (e.g., "basic_LandremanPaulQA" from "basic_LandremanPaulQA.yaml")
    case_name = case_path.stem if case_path.suffix == ".yaml" else case_path.name
    
    # Write to submissions directory: submissions/<surface>/<username>/<case_name>/<datetime>/
    submission_dir = submissions_dir / surface_name / github_username / case_name / datetime_str
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Coils filename is always coils.json
    coils_filename = "coils.json"
    coils_out_path = submission_dir / coils_filename

    # 1) Run the optimizer, writing coils_out_path and VTK files to submission_dir.
    # Note: optimize_coils handles MPI internally - only rank 0 runs optimization,
    # but all ranks participate in post-processing (fieldline tracing)
    if _is_proc0():
        typer.echo("Running optimizer...")
    results_dict = optimize_coils(
        case_path=case_path, 
        coils_out_path=coils_out_path, 
        case_cfg=case_cfg,
        output_dir=submission_dir,  # VTK files will be saved here
        run_vmec=run_vmec,
        run_simple=run_simple,
        plot_poincare=plot_poincare,
        plot_finite_build=plot_finite_build,
        finite_build_width=finite_build_width,
        finite_build_height=finite_build_height,
    )
    
    # Only rank 0 should write files and print messages
    if not _is_proc0():
        return  # Non-rank-0 processes exit after optimization/post-processing

    # 2) Evaluate the resulting coils.
    metrics = evaluate_case(case_cfg=case_cfg, results_dict=results_dict)

    # 3) Compute reactor-scale equivalent metrics
    reactor_scale_metrics = _compute_reactor_scale_metrics(metrics, case_cfg)

    # 4) Build submission results.json
    version_info = _get_version_info()
    submission = {
        "metadata": {
            "method_name": method_name or "",
            "contact": contact,
            "hardware": hardware,
            "notes": notes,
            "run_date": run_date,
        },
        "version_info": version_info,
        "metrics": metrics,
        "reactor_scale_metrics": reactor_scale_metrics,
    }
    
    # Write results.json
    submission_path = submission_dir / "results.json"
    submission_path.write_text(json.dumps(submission, indent=2, cls=NumpyJSONEncoder))
    _print_submission_summary(submission)
    
    # Copy case.yaml file to submission directory for reference
    # Also add source_case_file field to track which case file was used
    case_yaml_path = case_path if case_path.is_file() else (case_path / "case.yaml")
    if case_yaml_path.exists() and case_yaml_path.is_file():
        submission_case_yaml = submission_dir / "case.yaml"
        # Read the case file and add source_case_file field
        import yaml
        case_data = yaml.safe_load(case_yaml_path.read_text())
        # Store relative path from repo root for portability
        repo_root = Path.cwd()
        try:
            source_case_file = str(case_yaml_path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            # If relative path fails, use absolute path
            source_case_file = str(case_yaml_path.resolve())
        case_data["source_case_file"] = source_case_file
        # Write modified case.yaml to submission directory
        submission_case_yaml.write_text(yaml.dump(case_data, default_flow_style=False, sort_keys=False))
    
    # Zip the entire submission directory and remove original files
    _zip_submission_directory(submission_dir)


@app.command("run-case")
def run_case(
    case_path: Path = typer.Argument(
        ...,
        help="Path to case directory containing case.yaml and coils.yaml, or a single YAML file.",
    ),
    submissions_dir: Path = typer.Option(
        Path("submissions"),
        "--submissions-dir",
        help="Directory where submission results will be written.",
    ),
    results_out: Optional[Path] = typer.Option(
        None,
        "--results-out",
        "-o",
        help="Where to write the results JSON (default: <submissions_dir>/<surface>/<username>/<datetime>/results.json).",
    ),
) -> None:
    """
    Run a coil optimization for one case using parameters from case.yaml,
    then evaluate the resulting coil set.
    
    Creates a subdirectory in submissions/ with structure:
    submissions/<surface>/<username>/<datetime>/
    
    Note: For generating submissions, use 'submit-case' instead.
    """
    from .coil_optimization import optimize_coils
    from .evaluate import load_case_config, evaluate_case

    # Load case configuration
    case_cfg = load_case_config(case_path)

    # Extract surface name from case config
    surface_file = case_cfg.surface_params.get("surface", "")
    if not surface_file:
        raise ValueError("case.yaml must specify surface_params.surface")
    surface_name = _extract_surface_name(surface_file)

    # Auto-detect GitHub username for directory structure
    github_username = _detect_github_username()
    if not github_username:
        github_username = "unknown_user"
        typer.echo("Warning: Could not auto-detect GitHub username. Using 'unknown_user'.")

    # Create submission directory with case name to avoid race conditions
    now = datetime.now()
    datetime_str = now.strftime("%m-%d-%Y_%H-%M")  # Format: MM-DD-YYYY_HH-MM
    
    # Get case name from case file (e.g., "basic_LandremanPaulQA" from "basic_LandremanPaulQA.yaml")
    case_name = case_path.stem if case_path.suffix == ".yaml" else case_path.name
    
    # Create submission directory: submissions/<surface>/<username>/<case_name>/<datetime>/
    # Including case_name prevents race conditions when multiple cases use the same surface
    submission_dir = submissions_dir / surface_name / github_username / case_name / datetime_str
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Coils filename is always coils.json
    coils_filename = "coils.json"
    coils_out_path = submission_dir / coils_filename

    # 1) Run the optimizer, writing coils_out_path.
    # Note: optimize_coils handles MPI internally - only rank 0 runs optimization,
    # but all ranks participate in post-processing (fieldline tracing)
    if _is_proc0():
        typer.echo("Running optimizer...")
    results_dict = optimize_coils(case_path=case_path, coils_out_path=coils_out_path, case_cfg=case_cfg)
    
    # Only rank 0 should write files and print messages
    if not _is_proc0():
        return  # Non-rank-0 processes exit after optimization/post-processing

    # 2) Evaluate the resulting coils.
    metrics = evaluate_case(case_cfg=case_cfg, results_dict=results_dict)

    # 3) Compute reactor-scale equivalent metrics
    reactor_scale_metrics = _compute_reactor_scale_metrics(metrics, case_cfg)

    # Decide results filename.
    if results_out is None:
        results_out = submission_dir / "results.json"
    
    # Ensure output path has .json extension for JSON format
    if not str(results_out).endswith('.json'):
        results_out = results_out.with_suffix('.json')

    # Build submission with version info
    version_info = _get_version_info()
    run_date = datetime.now().isoformat()
    submission = {
        "metadata": {
            "run_date": run_date,
        },
        "version_info": version_info,
        "metrics": metrics,
        "reactor_scale_metrics": reactor_scale_metrics,
    }

    results_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.write_text(json.dumps(submission, indent=2, cls=NumpyJSONEncoder))
    _print_submission_summary(submission)


@app.command("run-ci-case")
def run_ci_case(
    case_file: Path = typer.Argument(
        ...,
        help="Path to a CI case JSON file (cases/pending/<case_id>.json).",
    ),
    output_dir: Path = typer.Option(
        Path("cases/done"),
        "--output-dir",
        "-o",
        help="Root directory for completed case results.",
    ),
    policy_file: Optional[Path] = typer.Option(
        None,
        "--policy",
        help="Path to proposer_policy.yaml for resource-cap validation.",
    ),
) -> None:
    """
    Run a single CI autopilot case from a JSON file and write a summary.

    This command is used by the CI runner workflow.  It:

    1. Validates the case JSON against resource caps.
    2. Writes a temporary case.yaml from the embedded ``case_config``.
    3. Runs the coil optimisation via ``optimize_coils``.
    4. Writes ``cases/done/<case_id>/summary.json`` with metrics, timing, and
       the original ``case_config`` for traceability.
    """
    import time as _time
    import yaml as _yaml

    from .coil_optimization import optimize_coils
    from .config_scheme import CaseConfig
    from .validate_config import validate_ci_case

    # ---- load & validate -------------------------------------------------
    case_text = case_file.read_text()
    try:
        case_data = json.loads(case_text)
    except json.JSONDecodeError as exc:
        typer.echo(f"ERROR: invalid JSON in {case_file}: {exc}", err=True)
        raise typer.Exit(code=1)

    policy: dict | None = None
    if policy_file and policy_file.exists():
        policy = _yaml.safe_load(policy_file.read_text())

    errors = validate_ci_case(case_data, policy=policy, file_path=case_file)
    if errors:
        for err in errors:
            typer.echo(f"VALIDATION ERROR: {err}", err=True)
        # Write a failure summary so the CI can still commit something
        case_id = case_data.get("case_id", case_file.stem)
        _write_ci_summary(
            output_dir / case_id / "summary.json",
            case_id=case_id,
            success=False,
            failure_reason="validation_error",
            failure_class="validation",
            case_config=case_data.get("case_config", {}),
        )
        raise typer.Exit(code=1)

    case_id = case_data["case_id"]
    case_config_dict = case_data["case_config"]
    resource = case_data.get("resource", {})
    random_seed = case_data.get("random_seed")

    # ---- seed ------------------------------------------------------------
    if random_seed is not None:
        np.random.seed(random_seed)

    # ---- write temp case.yaml and run ------------------------------------
    case_cfg = CaseConfig.from_dict(case_config_dict)
    out = output_dir / case_id
    out.mkdir(parents=True, exist_ok=True)

    # Write case.yaml into the output directory for traceability
    case_yaml_path = out / "case.yaml"
    import yaml as _yaml2
    case_yaml_path.write_text(_yaml2.dump(case_config_dict, default_flow_style=False))

    coils_out_path = out / "coils.json"
    wall_start = _time.time()
    results_dict: dict = {}  # set before try so it's always bound

    try:
        timeout_sec = resource.get("timeout_minutes", 120) * 60
        results_dict = optimize_coils(
            case_path=case_yaml_path,
            coils_out_path=coils_out_path,
            case_cfg=case_cfg,
            output_dir=out,
            skip_post_processing=False,
            run_vmec=False,
            run_simple=False,
            plot_poincare=False,
        )
        wall_end = _time.time()
        walltime = wall_end - wall_start

        # Check timeout (informational; the runner workflow should also enforce)
        timed_out = walltime > timeout_sec
        if timed_out:
            typer.echo(
                f"WARNING: case {case_id} exceeded timeout "
                f"({walltime:.0f}s > {timeout_sec}s)"
            )

        # ---- Build summary -----------------------------------------------
        # Extract the key metrics the proposer / guardrails need
        metrics = {
            k: v for k, v in results_dict.items()
            if isinstance(v, (int, float)) and not k.startswith("_")
        }

        summary: dict = {
            "case_id": case_id,
            "random_seed": random_seed,
            "tags": case_data.get("tags", []),
            "parent_ids": case_data.get("parent_ids", []),
            "success": True,
            "total_score": float(results_dict.get("final_squared_flux", float("inf"))),
            "iterations_used": int(results_dict.get("iterations_used", 0)),
            "walltime_sec": round(walltime, 2),
            "failure_reason": "",
            "failure_class": "",
            "config_hash": _config_hash(case_config_dict),
            "margins": _compute_margins(metrics),
            "metrics": metrics,
            "case_config": case_config_dict,
        }
        if "timing" in results_dict:
            summary["timing"] = results_dict["timing"]

    except Exception as exc:
        wall_end = _time.time()
        import traceback
        tb = traceback.format_exc()
        typer.echo(f"ERROR running case {case_id}: {exc}\n{tb}", err=True)
        summary = {
            "case_id": case_id,
            "random_seed": random_seed,
            "tags": case_data.get("tags", []),
            "parent_ids": case_data.get("parent_ids", []),
            "success": False,
            "total_score": float("inf"),
            "iterations_used": 0,
            "walltime_sec": round(wall_end - wall_start, 2),
            "failure_reason": str(exc),
            "failure_class": _canonical_failure_class(exc),
            "config_hash": _config_hash(case_config_dict),
            "margins": {},
            "metrics": {},
            "case_config": case_config_dict,
        }

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, cls=NumpyJSONEncoder))
    typer.echo(f"Wrote summary to {summary_path}")

    # ---- Also create a proper submissions/ entry for the leaderboard ------
    if summary.get("success"):
        try:
            _write_autopilot_submission(
                case_id=case_id,
                results_dict=results_dict,
                case_cfg=case_cfg,
                case_config_dict=case_config_dict,
                walltime=summary["walltime_sec"],
                repo_root=Path.cwd(),
                case_output_dir=out,
            )
        except Exception as exc:
            typer.echo(
                f"WARNING: could not create submission entry: {exc}", err=True
            )


def _write_autopilot_submission(
    *,
    case_id: str,
    results_dict: dict,
    case_cfg,
    case_config_dict: dict,
    walltime: float,
    repo_root: Path,
    submissions_dir: Path | None = None,
    case_output_dir: Path | None = None,
) -> None:
    """Create a ``submissions/`` entry so autopilot results appear on the leaderboard.

    The entry is formatted identically to human submissions so that no extra
    columns appear.  Directory structure:
    ``submissions/<surface>/auto/<case_id>/results.json``
    """
    import shutil
    from datetime import datetime as _dt

    submissions_dir = submissions_dir or (repo_root / "submissions")

    # --- extract surface name ---
    surface_file = ""
    sp = case_config_dict.get("surface_params", {})
    if isinstance(sp, dict):
        surface_file = sp.get("surface", "")
    elif hasattr(sp, "get"):
        surface_file = sp.get("surface", "")
    if not surface_file:
        raise ValueError("case_config must specify surface_params.surface")
    surface_name = _extract_surface_name(surface_file)

    # --- build metrics that match the human-submission format ---
    # Strip fields that human submissions don't have so no extra columns
    # appear on the leaderboard.
    _STRIP_KEYS = {"iterations_used", "walltime_sec", "output_directory"}
    full_metrics = {
        k: v for k, v in results_dict.items()
        if k not in _STRIP_KEYS
    }

    version_info = _get_version_info()
    reactor_scale_metrics = _compute_reactor_scale_metrics(full_metrics, case_cfg)

    submission = {
        "metadata": {
            "method_name": case_config_dict.get("description", f"autopilot_{case_id}"),
            "contact": "auto",
            "hardware": "CPU: self-hosted runner",
            "notes": f"Autopilot case {case_id}",
            "run_date": _dt.now().isoformat(),
        },
        "version_info": version_info,
        "metrics": full_metrics,
        "reactor_scale_metrics": reactor_scale_metrics,
    }

    sub_dir = submissions_dir / surface_name / "auto" / case_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_path = sub_dir / "results.json"
    sub_path.write_text(json.dumps(submission, indent=2, cls=NumpyJSONEncoder))

    # Write case.yaml so the leaderboard can extract N (ncoils) and n (order)
    import yaml as _yaml_mod
    case_yaml_path = sub_dir / "case.yaml"
    case_yaml_path.write_text(
        _yaml_mod.dump(case_config_dict, default_flow_style=False)
    )

    # Copy plot files so leaderboard links (i, f, PP) work.
    # Only Poincaré + Bnormal PDFs are generated by CI runs (VMEC/SIMPLE are off).
    if case_output_dir and case_output_dir.is_dir():
        # For Fourier continuation cases, plots live in order_X/ subdirectories.
        # Check the highest-order directory first, then fall back to the top level.
        fc_cfg = case_config_dict.get("fourier_continuation", {})
        search_dirs = [case_output_dir]
        if fc_cfg and fc_cfg.get("enabled", False):
            orders = fc_cfg.get("orders", [])
            if orders:
                highest = max(orders)
                fc_dir = case_output_dir / f"order_{highest}"
                if fc_dir.is_dir():
                    search_dirs.insert(0, fc_dir)
                # Also copy each order_X/ subdirectory into the submission
                for order in sorted(orders):
                    src_order_dir = case_output_dir / f"order_{order}"
                    dst_order_dir = sub_dir / f"order_{order}"
                    if src_order_dir.is_dir():
                        dst_order_dir.mkdir(parents=True, exist_ok=True)
                        for f in src_order_dir.iterdir():
                            if f.is_file():
                                shutil.copy2(f, dst_order_dir / f.name)

        for plot_name in (
            "bn_error_3d_plot.pdf",
            "bn_error_3d_plot_initial.pdf",
            "poincare_plot.png",
        ):
            for search_dir in search_dirs:
                src = search_dir / plot_name
                if src.exists():
                    shutil.copy2(src, sub_dir / plot_name)
                    break

    typer.echo(f"Wrote submission to {sub_path}")


def _config_hash(cfg: dict) -> str:
    """Deterministic hash of case_config for KB dedup and novelty checking."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _canonical_failure_class(exc: BaseException) -> str:
    """Map raw exception to canonical failure_class for KB queryability."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg or name == "TimeoutError":
        return "timeout"
    if "nan" in msg or "inf" in msg or name in ("ValueError", "FloatingPointError"):
        return "nan_in_objective"
    if "vmec" in msg or "VMEC" in name:
        return "vmec_nonconverged"
    if "line search" in msg or "linesearch" in msg or "trust region" in msg:
        return "line_search_fail"
    if "sep" in msg or "separation" in msg or "min_sep" in msg:
        return "min_sep_violation"
    if "validation" in msg:
        return "validation"
    return "unknown"


def _compute_margins(metrics: dict) -> dict:
    """Compute constraint margins from metrics vs thresholds (positive = satisfied)."""
    margins: dict = {}
    # cc: separation >= threshold → margin = actual - threshold
    if "final_min_cc_separation" in metrics and "cc_threshold" in metrics:
        margins["cc"] = float(metrics["final_min_cc_separation"]) - float(
            metrics["cc_threshold"]
        )
    if "final_min_cs_separation" in metrics and "cs_threshold" in metrics:
        margins["cs"] = float(metrics["final_min_cs_separation"]) - float(
            metrics["cs_threshold"]
        )
    # msc, curvature, flux, force, torque: value <= threshold → margin = threshold - value
    if "final_mean_squared_curvature" in metrics and "msc_threshold" in metrics:
        margins["msc"] = float(metrics["msc_threshold"]) - float(
            metrics["final_mean_squared_curvature"]
        )
    if "final_max_curvature" in metrics and "curvature_threshold" in metrics:
        margins["curvature"] = float(metrics["curvature_threshold"]) - float(
            metrics["final_max_curvature"]
        )
    if "final_squared_flux" in metrics and "flux_threshold" in metrics:
        margins["flux"] = float(metrics["flux_threshold"]) - float(
            metrics["final_squared_flux"]
        )
    if "final_max_max_coil_force" in metrics and "force_threshold" in metrics:
        margins["force"] = float(metrics["force_threshold"]) - float(
            metrics["final_max_max_coil_force"]
        )
    if "final_max_max_coil_torque" in metrics and "torque_threshold" in metrics:
        margins["torque"] = float(metrics["torque_threshold"]) - float(
            metrics["final_max_max_coil_torque"]
        )
    return margins


def _write_ci_summary(
    path: Path,
    *,
    case_id: str,
    success: bool,
    failure_reason: str = "",
    failure_class: str = "",
    case_config: dict | None = None,
    metrics: dict | None = None,
    total_score: float = float("inf"),
    iterations_used: int = 0,
    walltime_sec: float = 0.0,
    config_hash: str | None = None,
    margins: dict | None = None,
) -> None:
    """Helper to write a CI summary JSON (success or failure)."""
    cfg = case_config or {}
    m = metrics or {}
    summary = {
        "case_id": case_id,
        "success": success,
        "total_score": total_score,
        "iterations_used": iterations_used,
        "walltime_sec": walltime_sec,
        "failure_reason": failure_reason,
        "failure_class": failure_class,
        "config_hash": config_hash or _config_hash(cfg),
        "margins": margins if margins is not None else _compute_margins(m),
        "metrics": m,
        "case_config": cfg,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


@app.command("generate-submission")
def generate_submission(
    case_path: Path = typer.Argument(
        ...,
        help="Path to case.yaml file or directory containing case.yaml.",
    ),
    metadata_path: Path = typer.Argument(
        ...,
        help="Path to metadata.yaml file containing submission metadata.",
    ),
    coils_path: Path = typer.Option(
        None,
        "--coils",
        help="Path to coils.json file (default: <case_dir>/coils.json).",
    ),
    submission_out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Where to write the submission results.json file.",
    ),
) -> None:
    """
    Generate a results.json submission file from a case and coils file.
    
    This command:
    1. Loads metadata from metadata.yaml
    2. Loads case.yaml and evaluates the coils
    3. Creates a results.json file ready for submission
    
    Note: For running optimizations and generating submissions, use 'submit-case' instead.
    This command is for creating submissions from pre-existing coils files.
    """
    from .evaluate import load_case_config, evaluate_case
    from .config_scheme import SubmissionMetadata
    import yaml

    # Load metadata
    metadata_data = yaml.safe_load(metadata_path.read_text())
    metadata = SubmissionMetadata(
        method_name=metadata_data.get("method_name", "UNKNOWN"),
        method_version=metadata_data.get("method_version", "0.0.0"),
        contact=metadata_data.get("contact", ""),
        hardware=metadata_data.get("hardware", ""),
        notes=metadata_data.get("notes", ""),
    )

    # Load case configuration
    case_cfg = load_case_config(case_path)
    
    # Determine coils path
    if coils_path is None:
        if case_path.is_dir():
            coils_path = case_path / "coils.json"
        else:
            coils_path = case_path.parent / "coils.json"
    
    if not coils_path.exists():
        typer.echo(f"Error: Coils file not found: {coils_path}", err=True)
        raise typer.Exit(1)

    # Evaluate the coils (this would normally load and evaluate, but for now use placeholder)
    # In a real implementation, you'd load the coils and compute metrics
    results_dict = {
        "chi2_Bn": 0.001,  # Placeholder - would come from actual evaluation
    }
    
    metrics = evaluate_case(case_cfg=case_cfg, results_dict=results_dict)

    # Compute reactor-scale equivalent metrics
    reactor_scale_metrics = _compute_reactor_scale_metrics(metrics, case_cfg)

    # Build submission results
    run_date = datetime.now().isoformat()
    version_info = _get_version_info()
    submission = {
        "metadata": {
            "method_name": metadata.method_name,
            "method_version": metadata.method_version,
            "contact": metadata.contact,
            "hardware": metadata.hardware,
            "notes": metadata.notes,
            "run_date": run_date,
        },
        "version_info": version_info,
        "metrics": metrics,
        "reactor_scale_metrics": reactor_scale_metrics,
    }

    # Write output
    if submission_out is None:
        submission_out = Path("submissions") / metadata.method_name / metadata.method_version / "results.json"
    
    # Ensure output path has .json extension for JSON format
    if not str(submission_out).endswith('.json'):
        submission_out = submission_out.with_suffix('.json')
    
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission_out.write_text(json.dumps(submission, indent=2, cls=NumpyJSONEncoder))


@app.command("post-process")
def post_process(
    coils_json: Path = typer.Argument(
        ...,
        help="Path to coils JSON file (e.g., biot_savart_optimized.json or coils.json).",
    ),
    output_dir: Path = typer.Option(
        Path("post_processing_output"),
        "--output-dir",
        "-o",
        help="Directory where post-processing results will be saved.",
    ),
    case_yaml: Optional[Path] = typer.Option(
        None,
        "--case-yaml",
        help="Path to case.yaml file. If not provided, will search relative to coils JSON.",
    ),
    plasma_surfaces_dir: Optional[Path] = typer.Option(
        None,
        "--plasma-surfaces-dir",
        help="Directory containing plasma surface files. Defaults to 'plasma_surfaces'.",
    ),
    run_vmec: bool = typer.Option(
        False,
        "--run-vmec/--no-vmec",
        help="Whether to run QFM and VMEC equilibrium calculation (expensive, disabled by default).",
    ),
    helicity_m: int = typer.Option(
        1,
        "--helicity-m",
        help="Poloidal mode number for quasisymmetry evaluation.",
    ),
    helicity_n: int = typer.Option(
        0,
        "--helicity-n",
        help="Toroidal mode number for quasisymmetry evaluation.",
    ),
    ns: int = typer.Option(
        50,
        "--ns",
        help="Number of radial surfaces for quasisymmetry evaluation.",
    ),
    plot_boozer: bool = typer.Option(
        True,
        "--plot-boozer/--no-plot-boozer",
        help="Whether to generate Boozer surface plot.",
    ),
    plot_iota: bool = typer.Option(
        True,
        "--plot-iota/--no-plot-iota",
        help="Whether to generate iota profile plot.",
    ),
    plot_qs: bool = typer.Option(
        True,
        "--plot-qs/--no-plot-qs",
        help="Whether to generate quasisymmetry profile plot.",
    ),
    plot_poincare: bool = typer.Option(
        True,
        "--plot-poincare/--no-plot-poincare",
        help="Whether to generate Poincaré plot.",
    ),
    nfieldlines: int = typer.Option(
        20,
        "--nfieldlines",
        help="Number of fieldlines to trace for Poincaré plot.",
    ),
    run_simple: bool = typer.Option(
        False,
        "--run-simple/--no-simple",
        help="Whether to run SIMPLE fast particle tracing (requires --run-vmec, expensive).",
    ),
    plot_finite_build: bool = typer.Option(
        False,
        "--plot-finite-build/--no-plot-finite-build",
        help="Generate finite-build coil geometry (rectangular cross-section swept along centerline) and export to VTK. Output: finite_build_coils.vtk (and finite_build_coils_parastell.vtk if ParaStell available).",
    ),
    finite_build_width: Optional[float] = typer.Option(
        None,
        "--finite-build-width",
        help="Cross-section width [m] for finite-build coils. Default: 5 cm.",
    ),
    finite_build_height: Optional[float] = typer.Option(
        None,
        "--finite-build-height",
        help="Cross-section height [m] for finite-build coils. Default: 5 cm.",
    ),
) -> None:
    """
    Run post-processing on optimized coil results.
    
    This command performs analysis of optimized coil configurations, including:
    - Computing B·n on plasma surface (always)
    - Generating Poincaré plots (by default)
    - Computing QFM surfaces (with --run-vmec)
    - Running VMEC equilibrium calculations (with --run-vmec)
    - Computing quasisymmetry metrics (with --run-vmec)
    - Generating Boozer/iota/quasisymmetry plots (with --run-vmec)
    - Running SIMPLE particle tracing (with --run-vmec --run-simple)
    - Generating finite-build coil VTK (with --plot-finite-build)
    
    Examples:
        stellcoilbench post-process coils.json --output-dir results
        stellcoilbench post-process coils.json --run-vmec --output-dir results
        stellcoilbench post-process coils.json --plot-finite-build --output-dir results
    """
    from .post_processing import run_post_processing
    
    typer.echo(f"Running post-processing on {coils_json}")
    typer.echo(f"Output directory: {output_dir}")
    
    try:
        results = run_post_processing(
            coils_json_path=coils_json,
            output_dir=output_dir,
            case_yaml_path=case_yaml,
            plasma_surfaces_dir=plasma_surfaces_dir,
            run_vmec=run_vmec,
            helicity_m=helicity_m,
            helicity_n=helicity_n,
            ns=ns,
            plot_boozer=plot_boozer,
            plot_poincare=plot_poincare,
            nfieldlines=nfieldlines,
            run_simple=run_simple,
            plot_finite_build=plot_finite_build,
            finite_build_width=finite_build_width,
            finite_build_height=finite_build_height,
        )
        
        typer.echo("\nPost-processing complete!")
        typer.echo(f"Results saved to: {output_dir}")
        
        if 'quasisymmetry_average' in results:
            typer.echo(f"Average quasisymmetry error: {results['quasisymmetry_average']:.2e}")
        
    except Exception as e:
        typer.echo(f"Error during post-processing: {e}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
