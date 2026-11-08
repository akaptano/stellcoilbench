#!/usr/bin/env python3
"""FEM mesh resolution study: h-convergence with lines labeled by (p, q).

Single master function _run_mesh_resolution_study runs FEM at each resolution
for all (p, q) combinations and plots one convergence figure with lines labeled
by polynomial order p and quadrature degree q.

Run from repo root:
  python -m tools.fem_resolution_study [--output-dir DIR] [--coils-json PATH] [--torus]

Requires: simsopt, gmsh, matplotlib, ParaStell, DOLFINx.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import dolfinx  # noqa: F401
    import gmsh  # noqa: F401
except ImportError as e:
    print(f"Required packages missing: {e}")
    sys.exit(1)

from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import create_equally_spaced_curves

from stellcoilbench.coil_optimization._structural_mesh import _generate_torus_mesh_gmsh
from stellcoilbench.coil_optimization._structural_objective import StructuralStressObjective
from stellcoilbench.structural_analysis._common import _compute_coil_frame

PA_TO_GPA = 1e-9
_DEFAULT_COILS_JSON = REPO_ROOT / "structural_correlation_output/run_thr1e08_seed42/biot_savart_optimized.json"
_DEFAULT_CASE_YAML = REPO_ROOT / "cases/LandremanPaulQA_structural_optimization.yaml"
_QA_WIDTH, _QA_HEIGHT = 0.1, 0.1
_TORUS_WIDTH, _TORUS_HEIGHT = 0.1, 0.1


def _load_qa_coils(coils_json: Path, case_yaml: Path) -> tuple[list, list, BiotSavart]:
    from stellcoilbench.post_processing import load_coils_and_surface, get_unique_coils
    bfield, surface = load_coils_and_surface(
        coils_json, case_yaml_path=case_yaml,
        plasma_surfaces_dir=REPO_ROOT / "plasma_surfaces",
    )
    coils = bfield.coils
    nfp = int(surface.nfp)
    stellsym = bool(surface.stellsym)
    unique_coils = get_unique_coils(coils, nfp=nfp, stellsym=stellsym)
    return unique_coils, coils, bfield


def _create_parastell_msh(
    coils: list,
    tmp_path: Path,
    *,
    width: float,
    height: float,
    mesh_size: float,
    suffix: str = "",
) -> Path | None:
    from stellcoilbench.finite_build._parastell import _finite_build_coils_to_msh_parastell
    msh_file = tmp_path / f"qa_parastell{suffix}.msh"
    result = _finite_build_coils_to_msh_parastell(
        coils, msh_file, width=width, height=height, mesh_size=mesh_size,
    )
    if result is None:
        return None
    return result[0]


def _make_test_coils(ncoils: int = 2, order: int = 2, R0: float = 1.0, R1: float = 0.1):
    base_curves = create_equally_spaced_curves(ncoils, 1, stellsym=True, R0=R0, R1=R1, order=order)
    base_currents = [Current(5e7) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, 1, True)
    return coils, BiotSavart(coils)


def _create_torus_mesh(
    coils,
    tmp_path: Path,
    *,
    width: float,
    height: float,
    mesh_size: float,
    suffix: str = "",
) -> Path:
    frame = _compute_coil_frame(coils[0])
    gamma = frame["gamma"]
    xy = gamma[:, :2]
    R_major = float(np.mean(np.linalg.norm(xy, axis=1)))
    if R_major < 1e-6:
        R_major = 1.2
    centroid = np.mean(gamma, axis=0)
    r_minor = max(width, height) / 2
    msh_file = tmp_path / f"coil{suffix}.msh"
    temp_msh = _generate_torus_mesh_gmsh(
        float(centroid[0]), float(centroid[1]), float(centroid[2]),
        R_major, r_minor, mesh_size,
    )
    shutil.copy(temp_msh, msh_file)
    temp_msh.unlink(missing_ok=True)
    return msh_file


def _run_mesh_resolution_study(
    unique_coils: list,
    all_coils: list,
    bs,
    tmp_path: Path,
    resolutions: list[float],
    width: float,
    height: float,
    use_parastell: bool,
    output_dir: Path,
    *,
    p_orders: tuple[int, ...] = (1, 2, 3),
    q_degrees: tuple[int, ...] = (1, 2),
    diagnostics: bool = False,
    use_skfem_for_p1: bool = False,
) -> None:
    """Run FEM at each resolution for all (p, q), plot convergence with lines labeled by (p, q)."""
    from stellcoilbench.structural_analysis import write_structural_vtk
    from stellcoilbench.structural_analysis._common import _compute_jcross_b
    from tests.structural_analysis._integration_helpers import _solve_with_dolfinx

    cross_section_area = width * height
    E, nu = 100e9, 0.3

    def make_body_force_fn(coils, bs, area, w, h):
        def body_force_fn(coords: np.ndarray) -> np.ndarray:
            return _compute_jcross_b(coords, coils, bs, area, width=w, height=h, use_regularized=True)
        return body_force_fn

    body_force_fn = make_body_force_fn(all_coils, bs, cross_section_area, width, height)
    series: dict[tuple[int, int], list[float]] = {(p, q): [] for p in p_orders for q in q_degrees}

    for res in resolutions:
        if use_parastell:
            msh = _create_parastell_msh(
                unique_coils, tmp_path, width=width, height=height, mesh_size=res, suffix=f"_{res}"
            )
            if msh is None:
                raise RuntimeError(f"ParaStell mesh failed at resolution {res} m")
        else:
            msh = _create_torus_mesh(
                all_coils, tmp_path, width=width, height=height, mesh_size=res, suffix=f"_{res}"
            )

        print(f"Resolution {res} m:")
        for p in p_orders:
            for q in q_degrees:
                if p == 1:
                    backend = "skfem" if use_skfem_for_p1 else "dolfinx"
                    obj = StructuralStressObjective(
                        unique_coils=unique_coils,
                        bs=bs,
                        all_coils=all_coils,
                        width=width,
                        height=height,
                        mesh_resolution=res,
                        msh_path=msh,
                        stress_metric="max_von_mises",
                        structural_backend=backend,
                        quadrature_degree=q,
                    )
                    vm_GPa = obj.J()
                    print(f"  p={p} q={q}: max={vm_GPa:.6f} GPa")
                else:
                    u_nodes, vm_cells, mesh = _solve_with_dolfinx(
                        msh, body_force_fn, E=E, nu=nu, degree=p, bc_mode="bottom", q_degree=q,
                        diagnostics=diagnostics,
                    )
                    vm_arr = np.asarray(vm_cells, dtype=float) * PA_TO_GPA
                    vm_max = float(np.max(vm_arr))
                    vm_avg = float(np.mean(vm_arr))
                    print(f"  p={p} q={q}: max={vm_max:.6f} GPa, avg={vm_avg:.6f} GPa")
                    vm_GPa = vm_max
                    if (p, q) == (2, 2):
                        vtk_path = output_dir / f"structural_res_{res:.3f}m.vtk"
                        write_structural_vtk(mesh, u_nodes, vm_cells, vtk_path)
                        print(f"  Saved VTK with stresses: {vtk_path}")
                series[(p, q)].append(vm_GPa)

    # Plot
    res_arr = np.array(resolutions)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Mesh resolution (m)")
    ax.set_ylabel("Max Von Mises stress (GPa)")
    coil_label = "QA (ParaStell)" if use_parastell else "torus"
    ax.set_title(f"FEM convergence ({coil_label})")

    for (p, q), vals in series.items():
        arr = np.clip(np.array(vals), 1e-9, None)
        ax.semilogy(res_arr, arr, "o-", label=f"p={p} q={q}", markersize=5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "fem_resolution_study.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="FEM mesh resolution study")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tests" / "artifacts" / "fem_benchmarks")
    parser.add_argument("--coils-json", type=Path, default=_DEFAULT_COILS_JSON)
    parser.add_argument("--case-yaml", type=Path, default=_DEFAULT_CASE_YAML)
    parser.add_argument("--torus", action="store_true", help="Use torus coils (bypass ParaStell)")
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="*",
        default=None,
        help="Override resolutions [m] (e.g. --resolutions 0.05 0.02 for diagnostics)",
    )
    parser.add_argument("--diagnostics", action="store_true", help="Print RHS norm, displacement max, quadrature layout")
    parser.add_argument("--skfem", action="store_true", help="Use scikit-fem for p=1 (cross-check vs dolfinx)")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import tempfile

    if args.torus:
        all_coils, bs = _make_test_coils(ncoils=2, order=2)
        unique_coils = all_coils[:2]
        width, height = _TORUS_WIDTH, _TORUS_HEIGHT
        use_parastell = False
    else:
        if not args.coils_json.exists():
            print(f"Coils JSON not found: {args.coils_json}, falling back to torus")
            all_coils, bs = _make_test_coils(ncoils=2, order=2)
            unique_coils = all_coils[:2]
            width, height = _TORUS_WIDTH, _TORUS_HEIGHT
            use_parastell = False
        else:
            try:
                unique_coils, all_coils, bs = _load_qa_coils(args.coils_json, args.case_yaml)
                width, height = _QA_WIDTH, _QA_HEIGHT
                with tempfile.TemporaryDirectory() as probe:
                    msh = _create_parastell_msh(
                        unique_coils, Path(probe), width=width, height=height, mesh_size=0.12, suffix="_probe"
                    )
                use_parastell = msh is not None
                if not use_parastell:
                    print("ParaStell unavailable, falling back to torus")
                    all_coils, bs = _make_test_coils(ncoils=2, order=2)
                    unique_coils = all_coils[:2]
                    width, height = _TORUS_WIDTH, _TORUS_HEIGHT
            except Exception as e:
                print(f"QA coils load failed ({e}), falling back to torus")
                all_coils, bs = _make_test_coils(ncoils=2, order=2)
                unique_coils = all_coils[:2]
                width, height = _TORUS_WIDTH, _TORUS_HEIGHT
                use_parastell = False

    tube_radius = max(width, height) / 2
    resolutions = args.resolutions
    if resolutions is None:
        resolutions = [r for r in [0.32, 0.16, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02] if r <= tube_radius]
    if not resolutions:
        resolutions = [tube_radius * 0.8, tube_radius * 0.5, tube_radius * 0.25]

    with tempfile.TemporaryDirectory() as tmp:
        _run_mesh_resolution_study(
            unique_coils, all_coils, bs, Path(tmp),
            resolutions, width, height, use_parastell, out_dir,
            diagnostics=args.diagnostics,
            use_skfem_for_p1=args.skfem,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
