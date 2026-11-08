# Comprehensive Plan for Reducing J() Cost in Structural Stress Objective

## 1. Mathematical Formulation and Current Pipeline

### 1.1 Weak Form of Linear Elasticity

Solve for displacement \(\mathbf{u} \in H^1(\Omega)^3\):

\[
\int_\Omega \left( \lambda (\nabla\cdot\mathbf{u})(\nabla\cdot\mathbf{v}) + 2\mu \,\varepsilon(\mathbf{u}):\varepsilon(\mathbf{v}) \right) d\mathbf{x}
= \int_\Omega \mathbf{f} \cdot \mathbf{v} \, d\mathbf{x}
\quad \forall \mathbf{v} \in V_0
\]

with \(\varepsilon(\mathbf{u}) = \tfrac{1}{2}(\nabla\mathbf{u} + \nabla\mathbf{u}^T)\), Lamé parameters \(\lambda, \mu\) from \((E, \nu)\), and body force \(\mathbf{f} = \mathbf{J} \times \mathbf{B}\). Homogeneous Dirichlet on fixed support nodes.

### 1.2 Discrete System

After P1 tetrahedral FEM assembly:

\[
\mathbf{K} \mathbf{u} = \mathbf{f}, \qquad \mathbf{K}_{ij} = \sum_e \int_{T_e} \mathcal{B}_i^T \mathbf{D} \mathcal{B}_j \, d\mathbf{x}, \qquad f_i = \sum_e \int_{T_e} N_i \mathbf{f} \, d\mathbf{x}
\]

- \(\mathcal{B}\) maps nodal displacements to strain; `sym_grad` implements this.
- \(\mathbf{D}\) is the elasticity tensor (isotropic).
- Assembly cost: \(O(n_{elem} \times n_{qpt} \times n_{loc}^2)\) per matrix entry.

### 1.3 J() Pipeline (Per Call)

1. **Mesh deformation** — `_deform_mesh()`: map nodes to updated coil geometry via nearest-neighbor + frame.
2. **J×B at quadrature** — `_compute_jcross_b(q_coords, ...)`: B from Landreman internal + BiotSavart(mutual).
3. **Stiffness assembly** — `K = stiffness.assemble(ib)`.
4. **RHS assembly** — `f_vec = body_load.assemble(ib, bf=bf_qp)`.
5. **Condense** — eliminate fixed DOFs.
6. **Solve** — SuperLU: \(\mathbf{K}_{free} \mathbf{u}_{free} = \mathbf{f}_{free}\).
7. **Von Mises** — \(\sigma_{vm} = \sqrt{\tfrac{3}{2} \mathbf{s}:\mathbf{s}}\) from \(\mathbf{u}\).

---

## 2. Current Cost Breakdown (5 J() evals, 0.736 s total)

| Rank | Function | tottime | % | Category |
|------|----------|---------|---|----------|
| 1 | `sym_grad` | 0.181 s | 25% | Stiffness kernel |
| 2 | `c_einsum` | 0.102 s | 14% | Assembly / tensor |
| 3 | `csr_sort_indices` | 0.056 s | 8% | Sparse format |
| 4 | `bilinear_form._kernel` | 0.052 s | 7% | Stiffness form |
| 5 | `_K_curvature_vector` | 0.040 s | 5% | J×B (Bκ) |
| 6 | `superlu.gssv` | 0.032 s | 4% | Linear solve |
| 7 | `ufunc.reduce` | 0.028 s | 4% | Element ops |
| 8 | `stiffness` | 0.026 s | 4% | Bilinear form |
| 9 | `_compute_B0` | 0.021 s | 3% | J×B (B0) |
| 10 | `_G_helper` | 0.018 s | 2% | J×B (aux) |

**Aggregate:** Stiffness ~50%, J×B ~15%, Solve ~4%, Sparse prep ~8%, Von Mises ~1%.

---

## 3. Reduction Strategies (Ordered by Impact and Feasibility)

### 3.1 Stiffness Matrix Caching (High Impact, Medium Effort)

**Observation:** \(\mathbf{K}\) depends on mesh geometry and Lamé parameters. For small coil perturbations, \(\mathbf{K}\) changes by \(O(h)\) (FD step) — negligible for gradient accuracy.

**Idea:** Cache \((\mathbf{K}, \mathtt{ib}, \mathtt{fixed\_dofs})\) and reuse when \(\|\mathbf{x} - \mathbf{x}_{cached}\|_\infty < \tau\).

**Mathematical condition:** Stiffness depends on \(\nabla N_i\) via Jacobian \(\mathbf{J}_e = \partial \mathbf{x}/\partial \boldsymbol{\xi}\). For P1 tets:
\[
\mathbf{K}_e = |T_e| \, \mathcal{B}_e^T \mathbf{D} \mathcal{B}_e
\]
Mesh node movement \(\delta\mathbf{p}\) changes \(|T_e|\) and \(\mathcal{B}_e\). Relative change in \(\mathbf{K}\) is \(O(\delta/h)\); for \(\delta \sim 10^{-5}\), this is \(10^{-3}\) — acceptable for FD.

**Implementation:**
1. Add `_cached_K`, `_cached_ib`, `_cached_fixed_dofs`, `_cached_x` to `StructuralStressObjective`.
2. Tolerance \(\tau\) from `structural_K_cache_tol` (default 0 or 1e-8).
3. In `_evaluate_stress_impl`, if `cached_K` is None or `\|full_x - _cached_x\|_\infty > \tau`: assemble and cache. Else: pass cached into `_solve_elasticity_skfem`.

**Expected gain:** Up to ~50% of J() time when cache hits (typical during optimization when coil moves slowly).

---

### 3.2 Mesh Coarsening (High Impact, Low Effort)

**Observation:** Cost scales as \(O(n_{elem})\) for assembly and roughly \(O(n_{dof}^{1.5})\) for SuperLU on 3D tets.

**Idea:** Use coarser `structural_mesh_resolution` (e.g. 0.08 m vs 0.04 m).

**Scaling:** \(n_{elem} \propto h^{-3}\). Doubling \(h\) → ~8× fewer elements → ~4–6× faster assembly + solve.

**Accuracy:** Von Mises converges as \(O(h)\) for P1. Use Richardson extrapolation or mesh-refinement studies to bound error. For optimization, coarse stress estimate often sufficient to guide descent; finer mesh for final validation.

**Config:** `structural_mesh_resolution: 0.08` or adaptive: coarse early, refine when \(\sigma_{vm}\) approaches threshold.

**Expected gain:** 2–6× depending on coarsening factor.

---

### 3.3 Reduced Quadrature for J×B (Medium Impact, Medium Effort)

**Observation:** J×B is evaluated at all quadrature points: \(n_{elem} \times n_{qpt}\) (P1 tet: typically 1–4 qpts/elem). For 12k elements × 4 coils, that is ~48k B evaluations per J().

**Idea:** Evaluate J×B at a subset of quadrature points and interpolate; or use nodal interpolation of a coarser J×B field.

**Mathematical trade-off:** RHS accuracy vs cost. Body force \(\mathbf{f}\) integrated as \(\int N_i \mathbf{f}\). If \(\mathbf{f}\) is smooth, reduced quadrature (e.g. centroid only) can suffice. Error analysis: \(\|\mathbf{f} - \mathbf{f}_h\|_{L^2} \lesssim h^k\) for smooth \(\mathbf{f}\); integration error \(O(h^{q+1})\) for quadrature order \(q\).

**Options:**
1. **Centroid-only:** Evaluate J×B at element centroids; constant \(\mathbf{f}\) per element. Cuts evaluations by factor of \(n_{qpt}\).
2. **Sparse quadrature:** Every 2nd or 4th point in each direction; interpolate.
3. **Nodal J×B:** Compute at nodes, interpolate to quadrature (add nodal projection step).

**Expected gain:** 30–50% of J×B cost (~5–8% of total J()).

---

### 3.4 Sparse Format Optimization (Low–Medium Impact, Low Effort)

**Observation:** `csr_sort_indices` (8%) runs 5× (once per coil solve). Scikit-fem assembles to COO; conversion to CSR triggers sort.

**Idea:**
1. Assemble directly to CSR if skfem supports it.
2. Cache sorted CSR if sparsity pattern is unchanged (mesh fixed, only RHS changes).
3. Use CSC or other format that avoids per-solve sort when possible.

**Constraint:** Skfem API may not expose format choice. Profile to confirm if sort is in our code path or inside condense/solve.

**Expected gain:** Up to ~8% if eliminated.

---

### 3.5 J×B Computational Kernels (Medium Impact, High Effort)

**Observation:** `_K_curvature_vector`, `_G_helper`, `_compute_B0` dominate J×B. All are pure NumPy.

**Ideas:**
1. **Numba JIT:** Compile hot loops (G_helper, K_curvature, B0, Bκ) with `@njit`. Expected 2–5× speedup for these kernels.
2. **Vectorization audit:** Ensure no Python loops; use `np.einsum` or `np.tensordot` where beneficial.
3. **Fused kernels:** Combine B0 + Bκ + Bb into one pass to reduce memory traffic.

**Mathematical note:** Landreman formulas involve `arctan`, `log`, and `G(u,v)`. These are inherently serial per point; parallelization across points is trivial (NumPy already does this). Numba reduces Python overhead and can improve memory access patterns.

**Expected gain:** 2–4× on J×B (~10–20% of total J()).

---

### 3.6 eval_interval and Lazy Evaluation (Config-Only)

**Observation:** J() is called every optimizer iteration. When stress is far below threshold, precise J is unnecessary.

**Ideas:**
1. **structural_eval_interval:** Evaluate every N iterations; cache and return last value otherwise. Already implemented.
2. **Adaptive interval:** Start with interval 3–5; reduce to 1 when \(\sigma_{vm}\) approaches threshold.
3. **Surrogate:** Use a cheap surrogate (e.g. coil curvature) when far from threshold; switch to FEM when close.

**Expected gain:** Reduces number of J() calls; no per-call speedup.

---

### 3.7 Alternative FEM Backends (Research)

**Observation:** Assembly (sym_grad, einsum, skfem kernel) is ~50%. Scikit-fem uses Python/NumPy; no JIT by default.

**Options:**
1. **DOLFINx:** Already supported in codebase; may have faster assembly (C++/JIT). Benchmark both.
2. **JAX-FEM / Sparse FEM:** JAX XLA compilation could reduce Python overhead.
3. **Custom Cython/Numba assembly:** Replace skfem assembly for the specific linear elasticity form.

**Minimal formulation:** Only need stiffness \(\mathbf{K}\) and RHS \(\mathbf{f}\). A custom assembler for isotropic linear elasticity on P1 tets is a small, self-contained project.

**Expected gain:** 2–3× on assembly if backend is faster; high implementation cost.

---

### 3.8 Matrix-Free / Iterative with Preconditioner (Research)

**Observation:** SuperLU dominates only ~4% of J(); assembly dominates. Iterative solvers (CG, GMRES) do not reduce assembly cost. Matrix-free would avoid storing \(\mathbf{K}\), but assembly is still needed for RHS and for preconditioner. Not recommended unless assembly is first reduced.

---

## 4. Recommended Implementation Order

| Phase | Strategy | Effort | Expected J() Reduction | Risk |
|-------|----------|--------|-------------------------|------|
| 1 | Mesh coarsening (0.08 m) | Low | 2–6× | Stress accuracy |
| 2 | Stiffness caching | Medium | Up to ~50% when cache hits | Stale K if tol too loose |
| 3 | eval_interval tuning | Low | Fewer calls | Convergence |
| 4 | Reduced quadrature for J×B | Medium | ~5–8% | RHS accuracy |
| 5 | Sparse format / csr_sort | Low | ~8% | API constraints |
| 6 | Numba for J×B kernels | High | ~10–20% | Maintenance |
| 7 | Alternative backend | Research | 2–3× assembly | Large refactor |

---

## 5. Validation Requirements

For any change affecting accuracy:

1. **Taylor test:** \(\tfrac{|J(\mathbf{x}+h\mathbf{e}_i) - J(\mathbf{x}) - h \nabla J_i|}{h^2} \to 0\) as \(h \to 0\).
2. **Stress comparison:** Run short optimization with proposed change; compare \(\sigma_{vm}\) VTK to baseline at same iteration.
3. **Convergence:** Ensure optimizer reaches similar or better final \(\sigma_{vm}\) and other objectives.

---

## 6. Summary

The dominant cost is **stiffness assembly** (~50%), followed by **J×B body force** (~15%) and **sparse format prep** (~8%). Highest-impact, feasible reductions:

1. **Stiffness caching** — reuse K when coil perturbation is small.
2. **Mesh coarsening** — fewer elements via larger resolution.
3. **Reduced quadrature for J×B** — fewer B evaluations.
4. **Numba for J×B kernels** — faster execution of existing math.

Combined, these can realistically achieve **2–4×** reduction in J() wall time with low to medium effort.
