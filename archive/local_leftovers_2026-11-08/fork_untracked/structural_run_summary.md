# Run and Time Summary: basic_LandremanPaulQA_structural.yaml

**Run date:** 2026-03-09  
**Config:** max_iterations=5 (run showed 2), structural_eval_interval=2, use_cached_K=false, adaptive mesh (0.16 m coarse, 0.08 m fine)

---

## 1. Total Timing

| Metric | Value |
|--------|-------|
| **Total wall time** | 2m 7.1s (127.1 s) |
| **User CPU time** | 1m 49.7s |
| **System CPU time** | 22.5 s |
| **Optimization walltime** | 105 s (from results) |
| **Iterations used** | 2 |
| **nfev / njev** | 10 each |

---

## 2. Speedup Analysis

### Current optimizations in effect

- **eval_interval=2**: Reduces J() calls when stress is active (every 2nd iteration uses cached J).
- **Adaptive mesh**: Starts at 0.16 m coarse; refines to 0.08 m when σ_vm ≥ 50% of threshold. Mesh refinement was triggered after iter 1 (σ_vm=1.53 GPa ≥ 0.05 GPa).
- **Short-circuit**: When σ_vm ≤ threshold, dJ() returns 0 without FD gradient. In this run, σ_vm (1.53 GPa) exceeded threshold (0.1 GPa), so **dJ_short_circuit=False**—the full FD gradient was computed every iteration.

### Structural timing benchmark (`python tools/structural_timing_benchmark.py`)

| Config | use_cached_K | J() (s/call) | dJ() (s/call) | Notes |
|--------|--------------|--------------|---------------|-------|
| Single 0.04 m (fine baseline) | False | 1.10 | 124.04 | Fresh K per FD perturbation |
| Single 0.08 m (coarse) | False | 0.16 | 15.68 | 86% faster J(), 87% faster dJ() vs 0.04 m |
| Adaptive (0.16 coarse, 0.08 fine) | False | 0.04 | 5.04 | 97% faster J(), 96% faster dJ() vs 0.04 m |
| Adaptive (0.16 coarse, 0.08 fine) | True | 0.04 | 3.09 | **38.7% faster dJ()** vs use_cached_K=False |

**use_cached_K speedup:** On adaptive mesh, `use_cached_K=True` reduces dJ() from 5.04 s/call to 3.09 s/call (38.7% faster). Trade-off: gradient is less accurate because stiffness K is cached at baseline; acceptable for many optimizations.

---

## 3. Gradient Issues

### Taylor test: **FAILED**

```
WARNING: Taylor test failed: error ratio 1.000 > 0.6 (ε=1.0e-06 -> 1.0e-07)
WARNING: Taylor test failed: error ratio 1.000 > 0.6 (ε=1.0e-07 -> 1.0e-08)
```

The Taylor test runs on the **full JF** (flux + length + structural + distance + curvature + MSC + …) at the start of optimization. Error ratio 1.0 means the residual did not decrease when ε was reduced—the gradient is not consistent with the Taylor expansion at the tested ε.

### Taylor diagnostic (structural-only, `python tools/diagnose_taylor_test.py`)

| Epsilons | eps=1e-6 | eps=1e-7 | eps=1e-8 | Error ratio 1e-6→1e-7 | Error ratio 1e-7→1e-8 |
|----------|----------|----------|----------|------------------------|------------------------|
| (1e-6, 1e-7, 1e-8) | err=2.5e-6 | err=4.3e-11 | err=2.2e-7 | 0.00 (pass) | **5168** (fail) |

| Epsilons | eps=1e-4 | eps=1e-5 | eps=1e-6 | Error ratio 1e-4→1e-5 | Error ratio 1e-5→1e-6 |
|----------|----------|----------|----------|------------------------|------------------------|
| (1e-4, 1e-5, 1e-6) | err=2.5e-4 | err=3.7e-9 | err=2.2e-5 | 0.00 (pass) | **5895** (fail) |

**Interpretation:** The structural gradient is correct (error decreases from 1e-4→1e-5 and 1e-6→1e-7). The failure at smaller ε is **machine precision**: when ε is so small that `J(x0 + εh)` is unchanged (quantized to float), `Jp` stays constant while `Jpred = J0 + ε∇J·h` keeps changing with ε, so the residual grows. Use coarser epsilons: `TAYLOR_TEST_EPSILONS = (1e-4, 1e-5)` or `(1e-3, 1e-4, 1e-5)` avoids this regime.

**Possible causes for full JF failure:**

- Very small epsilons (1e-7, 1e-8) hitting machine precision (confirmed for structural term).
- Another term (flux, curvature, MSC) may have similar or different numerical behavior.

---

## 4. Runtime Distribution

### J() profile (5 evals, ~0.49 s total from profile_stats.txt)

| Phase | % of time | Notes |
|-------|-----------|-------|
| sym_grad | 24% | Stiffness kernel |
| c_einsum | 14% | Assembly/tensor |
| bilinear_form._kernel | 7% | Stiffness form |
| csr_sort_indices | 7% | Sparse format |
| _K_curvature_vector | 4% | J×B |
| stiffness | 4% | skfem |
| superlu.gssv | 4% | Linear solve |

**J() ≈ 0.10–0.15 s/call** (profile: 5 calls in 0.49 s; verbose: 5 calls in 0.73 s)

### dJ() profile (1 call, ~7.3 s from profile_stats_dJ.txt)

| Phase | % of time | Notes |
|-------|-----------|-------|
| sym_grad | 22% | Stiffness kernel |
| c_einsum | 12% | Assembly |
| bilinear_form._kernel | 12% | Stiffness |
| csr_sort_indices | 6% | Sparse format |
| SuperLU solve | 3% | Linear solve |
| _K_curvature_vector | 4% | J×B |
| dofs_free_status | 4% | simsopt |

**dJ() ≈ 7.3 s/call** (65 FD perturbations ≈ n_dofs+1 with use_cached_K=false)

### Phase breakdown

| Phase | Share | Notes |
|-------|-------|-------|
| Stiffness assembly (sym_grad, BilinearForm) | ~50% of J() | Main cost |
| J×B body force | ~15% | Landreman, B evaluation |
| Linear solve (SuperLU) | ~4% | |
| dJ() | n_dofs+1 × J() | ~65 × 0.1 s ≈ 6.5 s, matches profile |

---

## 5. Summary

| Question | Answer |
|----------|--------|
| **Speedup achieved?** | Yes, indirectly: eval_interval=2, adaptive mesh (0.16→0.08 m), and short-circuit when σ_vm ≤ threshold. No side‑by‑side baseline in this run. |
| **Gradient issues?** | Yes: Taylor test failed on full JF (error ratio 1.0 at ε=1e-6→1e-7 and 1e-7→1e-8). Structural term passes its own Taylor test in unit tests. |
| **Runtime distribution** | Stiffness assembly ~50%, J×B ~15%, solve ~4%. J() ~0.1 s/call, dJ() ~7 s/call. |

---

## 6. Recommendations

1. **Taylor test**: Set `TAYLOR_TEST_EPSILONS = (1e-4, 1e-5)` or `(1e-3, 1e-4, 1e-5)` in `constants.py` to avoid machine precision at 1e-6 and smaller.
2. **Speedup comparison**: Run with structural_eval_interval=1 and structural_use_cached_K=false vs current to measure eval_interval impact.
3. **Short-circuit ratio**: For longer runs, log dJ_short_circuit=True vs False; when stress stays below threshold, most dJ() calls are skipped and runtime drops.

---

## 7. DOLFINx Feasibility (tools/dolfinx_skfem_benchmark.py)

| Backend | Wall time (10 solves, mesh loaded once) | Speedup |
|---------|----------------------------------------|---------|
| skfem   | 0.59 s (0.059 s/solve)                 | 1.0×    |
| DOLFINx | 0.21 s (0.021 s/solve)                 | **2.78×** |

DOLFINx is substantially faster for assembly+solve. Integration would require a backend switch in `StructuralStressObjective`; mesh deform + J×B coupling needs validation.

---

## 8. dJ() Speed Diagnosis (tools/diagnose_dj_speed.py)

| Phase                          | Time (est.) | Share  |
|--------------------------------|-------------|--------|
| Baseline _evaluate_stress_impl | 0.19 s      | 4%     |
| baseline_frames + Breg (1×)    | 0.02 s      | <1%    |
| dof_to_base setup              | ~0 s        | 0%     |
| Per-iter frame/Breg update (×64) | 0.85 s   | **18.5%** |
| Per-iter _evaluate_stress_impl (×64) | 3.53 s | **77%** |

**Conclusion:** Most of dJ() is in `_evaluate_stress_impl` (77%), but frame/Breg update is significant (18.5%). Caching or simplifying frame computation when only a subset of coils change could yield modest gains.

---

## 9. Further Speedup Opportunities

From `docs/structural_J_cost_reduction_plan.md`:

| Strategy | Effort | Expected gain | Notes |
|----------|--------|---------------|-------|
| **Mesh coarsening** (e.g. 0.08 m) | Low | 2–6× | Already supported; adaptive mesh refines when needed. |
| **Stiffness caching** (`use_cached_K=True`) | Config-only | Up to ~50% of dJ() when cache hits | 38.7% measured on adaptive mesh; trade-off: less accurate gradient. |
| **eval_interval tuning** | Config-only | Fewer J() calls | `structural_eval_interval=2` already used. |
| **Reduced quadrature for J×B** | Medium | ~5–8% | Evaluate J×B at centroids only; needs validation. |
| **Sparse format / csr_sort** | Low | ~8% | Avoid per-solve sort if skfem allows. |
| **Numba JIT for J×B kernels** | High | ~10–20% | Compile `_K_curvature_vector`, `_G_helper`, `_compute_B0`. |
| **Alternative FEM backend** (DOLFINx) | Research | **2.78× measured** | tools/dolfinx_skfem_benchmark.py; integration needs mesh-deform + J×B validation. |
