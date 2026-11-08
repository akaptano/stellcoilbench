# dJ() Timing Report: PETSc and JIT Optimizations

**Date:** 2026-03-09  
**Case:** basic_LandremanPaulQA_structural.yaml (max_von_mises, adaptive mesh 0.16→0.08 m)  
**Platform:** Apple M3 Pro, stellcoilbench_vmec conda env

## Summary

| Configuration | DOLFINx dJ() | DOLFINx _evaluate_stress_impl |
|---------------|--------------|-------------------------------|
| **Baseline** (LU, q=2) | 3.62 s | 0.053 s |
| **Optimized** (LU, q=1, JIT -O3 -march=native) | **2.54 s** | ~0.035 s |
| **PETSc iterative** (CG+GAMG) | 5.03 s | 0.068 s |

**Conclusion:** `structural_quadrature_degree: 1` + `dolfinx_jit_options.json` yields **~1.4× faster dJ()** (2.54 s vs 3.62 s). Stick with LU; iterative solvers do not beat it at this scale.

## CG Variants: Jacobi and Tolerance

| Config | Time (s) | Speedup vs LU | J (Pa) |
|--------|----------|---------------|--------|
| LU (baseline) | 0.054 | 1.00× | 18.149 M |
| CG, no PC | 0.068 | 0.79× | 18.149 M |
| CG + Jacobi | 0.066 | 0.81× | 18.149 M |
| CG + Jacobi, rtol=1e-6 | 0.063 | 0.85× | 18.149 M |
| CG + Jacobi, rtol=1e-4 | 0.058 | 0.93× | 18.149 M |

**Findings:**
- Jacobi gives a small speedup over CG without preconditioner (0.066 vs 0.068 s).
- Looser tolerance helps: rtol=1e-4 is ~13% faster than rtol=1e-10 (0.058 vs 0.066 s).
- LU remains fastest; CG+Jacobi at rtol=1e-4 is still ~8% slower than LU.
- J values are nearly identical; rtol=1e-4 yields ~0.0001% relative error.

**Env vars:** `CG_JACOBI=1`, `CG_NOPC=1`, `CG_RTOL=1e-6` (with `ITERATIVE=1`).

## dJ() Timing (full FD gradient — the actual bottleneck)

~1.6k tets, q=2. Run: `python tools/benchmark_petsc_cg.py --dj`.

| Config                  | mesh        | Time (s) | Speedup | J (GPa) |
|-------------------------|-------------|----------|---------|---------|
| LU (baseline)           | ~1.6k tets  | 3.62     | 1.00×   | 0.0181  |
| Cholesky                | ~1.6k tets  | 3.61     | 1.00×   | 0.0181  |
| LU+MUMPS                | ~1.6k tets  | 4.05     | 0.89×   | 0.0181  |
| LU+SuperLU              | ~1.6k tets  | 3.64     | 0.99×   | 0.0181  |
| CG no PC, rtol=1e-2     | ~1.6k tets  | 3.56     | **1.02×** | 0.0182  |
| CG no PC, rtol=1e-1     | ~1.6k tets  | 3.65     | 0.99×   | 0.0182  |
| CG+Jacobi, rtol=1e-2    | ~1.6k tets  | 3.78     | 0.96×   | 0.0181  |
| CG+Jacobi, rtol=1e-1    | ~1.6k tets  | 3.53     | **1.02×** | 0.0178  |
| GMRES no PC, rtol=1e-2  | ~1.6k tets  | 5.20     | 0.70×   | 0.0172  |
| GMRES no PC, rtol=1e-1  | ~1.6k tets  | 3.61     | 1.00×   | **0.0091** (inaccurate) |
| GMRES+Jacobi, rtol=1e-2 | ~1.6k tets  | 4.48     | 0.81×   | 0.0173  |
| GMRES+Jacobi, rtol=1e-1 | ~1.6k tets  | 3.65     | 0.99×   | **0.0097** (inaccurate) |

**Findings:**
- **CG no PC rtol=1e-2** and **CG+Jacobi rtol=1e-1** are ~2% faster than LU for dJ().
- LU, Cholesky, LU+SuperLU are all within 1% — stick with default LU.
- GMRES is 19–30% slower (rtol=1e-2); GMRES rtol=1e-1 matches LU but J is ~50% wrong.
- MUMPS is ~11% slower.

---

## J() Timing: Solver + Quadrature + Mesh (LU, Cholesky, MUMPS, SuperLU, CG, GMRES)

Single mesh resolution (non-adaptive). ~1.6k tets = 0.16 m, ~3k tets = 0.10 m. Baseline: LU q=2 at 1.6k tets.

### ~1.6k tets (1586 elements, res=0.16 m)

| Config                  | q   | Time (s) | Speedup | J (Pa)   |
|-------------------------|-----|----------|---------|----------|
| LU (baseline)           | 2   | 0.053    | 1.00×   | 18.15 M  |
| LU (baseline)           | 1   | 0.034    | 1.53×   | 18.46 M  |
| Cholesky                | 2   | 0.054    | 0.97×   | 18.15 M  |
| Cholesky                | 1   | 0.036    | 1.48×   | 18.46 M  |
| LU+MUMPS                | 2   | 0.080    | 0.66×   | 18.15 M  |
| LU+MUMPS                | 1   | 0.044    | 1.20×   | 18.46 M  |
| LU+SuperLU              | 2   | 0.054    | 0.97×   | 18.15 M  |
| LU+SuperLU              | 1   | 0.035    | 1.49×   | 18.46 M  |
| CG no PC, rtol=1e-2     | 2   | 0.056    | 0.94×   | 18.15 M  |
| CG no PC, rtol=1e-2     | 1   | 0.037    | 1.44×   | 18.47 M  |
| CG no PC, rtol=1e-1     | 2   | 0.061    | 0.86×   | 18.16 M  |
| CG no PC, rtol=1e-1     | 1   | 0.037    | 1.42×   | 18.48 M  |
| CG+Jacobi, rtol=1e-2    | 2   | 0.054    | 0.97×   | 18.07 M  |
| CG+Jacobi, rtol=1e-2    | 1   | 0.036    | 1.44×   | 18.38 M  |
| CG+Jacobi, rtol=1e-1    | 2   | 0.054    | 0.98×   | 17.84 M  |
| CG+Jacobi, rtol=1e-1    | 1   | 0.035    | 1.51×   | 18.16 M  |
| GMRES no PC, rtol=1e-2  | 2   | 0.078    | 0.67×   | 17.21 M  |
| GMRES no PC, rtol=1e-2  | 1   | 0.061    | 0.86×   | 17.55 M  |
| GMRES no PC, rtol=1e-1  | 2   | 0.054    | 0.98×   | **9.12 M** (inaccurate) |
| GMRES no PC, rtol=1e-1  | 1   | 0.044    | 1.20×   | **9.58 M** (inaccurate) |
| GMRES+Jacobi, rtol=1e-2 | 2   | 0.069    | 0.76×   | 17.26 M  |
| GMRES+Jacobi, rtol=1e-2 | 1   | 0.047    | 1.11×   | 17.70 M  |
| GMRES+Jacobi, rtol=1e-1 | 2   | 0.054    | 0.97×   | **9.67 M** (inaccurate) |
| GMRES+Jacobi, rtol=1e-1 | 1   | 0.035    | 1.50×   | **10.45 M** (inaccurate) |

**Findings:**
- **CG slower than LU despite low tolerance:** Even at rtol=1e-2 and rtol=1e-1, CG is ~86–97% of LU. The dominant cost is assembly (J×B, stiffness) and mesh ops — the solve is only ~15–25% of `_evaluate_stress_impl`. Iterative solvers add per-iteration matvec overhead; for small systems (~5k DOFs) direct LU is cheaper.
- **rtol=1e-1:** CG+Jacobi rtol=1e-1 nearly matches LU (0.98×); GMRES rtol=1e-1 is fast but **J is ~50% wrong** — too loose for optimization.
- **GMRES** is slower than CG for SPD systems (no PC: 0.67× vs 0.94×); GMRES+Jacobi rtol=1e-1 can match LU speed but J is inaccurate.
- **q_degree=1** gives ~33–50% speedup; J +1.7% vs q=2.
- **~10k tets:** On conda PETSc builds, larger meshes can trigger Cholesky/SuperLU backend errors; use default LU for production.
- **Env vars:** `KSP_TYPE=gmres`, `CG_RTOL=1e-1` (with `ITERATIVE=1`); `structural_quadrature_degree: 1`.

## Implementation

- **PETSc options:** Configurable via `_get_structural_petsc_options()` in `_dolfinx.py`. Default: `preonly` + `lu`. When `ITERATIVE=1`: `KSP_TYPE=cg|gmres` (default cg), `CG_RTOL`, `CG_NOPC`, `CG_JACOBI`. Direct: `CHOLESKY=1`, `LU_SOLVER=mumps|superlu`. SuperLU NOREFINE; MUMPS icntl_7=4.
- **JIT options:** Create `dolfinx_jit_options.json` in cwd with `cffi_extra_compile_args: ["-O3", "-march=native"]` before first DOLFINx import. Use `python tools/compare_dj_backends.py --optimized`.
- **Benchmark:** `python tools/benchmark_petsc_cg.py` (J); `python tools/benchmark_petsc_cg.py --dj` (dJ); `python tools/benchmark_petsc_cg.py --dj --case-config` (time case YAML as configured); `python tools/compare_dj_backends.py` (baseline), `--petsc-iterative`, or `--optimized`.

## K Caching (use_cached_K)

K caching is supported for the **skfem** backend only. Caches condensed K_ff; dJ() up to ~40% faster when `structural_use_cached_K: true`.
