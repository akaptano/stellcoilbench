# Benchmark Cases

Case definitions in YAML. See `docs/cases.rst` for full schema.

**Required fields:** `description`, `surface_params` (surface, range), `coils_params` (ncoils, order), `optimizer_params` (algorithm, max_iterations, max_iter_subopt).

**Optional:** `coil_objective_terms`, `fourier_continuation`, `virtual_casing`. Case ID comes from directory/metadata, not a `case_id` field.

## Example

```yaml
description: "Basic test"
surface_params:
  surface: "input.LandremanPaul2021_QA"
  range: "half period"
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: "L-BFGS-B"
  max_iterations: 200
  max_iter_subopt: 10
coil_objective_terms:
  total_length: "l2_threshold"
  coil_curvature: "lp_threshold"
  coil_curvature_p: 2
  linking_number: ""
```

## Virtual Casing

Set `virtual_casing: true` with a VMEC wout file. Requires `virtual_casing` Python package.

## Run

```bash
stellcoilbench submit-case cases/my_case.yaml
```
