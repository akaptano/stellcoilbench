# Benchmark Cases

This directory contains benchmark case definitions in `case.yaml` format.

## Structure

Each `case.yaml` file defines:
- `description`: Brief description of the case
- `surface_params`: Plasma surface configuration
  - `surface`: Surface file name (must match a file in `plasma_surfaces/` directory, e.g., `input.LandremanPaul2021_QA`)
  - `range`: Surface range (`"half period"` or `"full torus"`)
  - `virtual_casing`: Boolean to enable virtual casing (default: `false`). Requires a VMEC wout file.
- `coils_params`: Coil optimization parameters
  - `ncoils`: Number of coils (required, must be positive integer)
  - `order`: Fourier order for coil curves (required, must be positive integer)
- `optimizer_params`: Optimizer settings
  - `algorithm`: Optimization algorithm (e.g., `"L-BFGS-B"`, `"BFGS"`, `"SLSQP"`, `"augmented_lagrangian"`)
  - `max_iterations`: Maximum optimization iterations
  - `max_iter_subopt`: Maximum suboptimization iterations (for augmented Lagrangian)
  - `verbose`: Verbose flag (optional, boolean) - controls optimization progress output
  - `algorithm_options`: Optional dictionary of algorithm-specific hyperparameters
- `coil_objective_terms`: Optional dictionary specifying which coil objectives to include
  - Each term can be specified with options like `"l1"`, `"l1_threshold"`, `"l2"`, `"l2_threshold"`, `"lp"`, `"lp_threshold"`
  - Unspecified terms are not included in the objective function
  - `linking_number: ""` (empty string) includes the linking number objective
  - `coil_arclength_variation` is supported with `l1/l1_threshold/l2/l2_threshold`
  - `target_B` is automatically set based on the surface (not specified in case.yaml)

**Note**: Cases are identified by their directory structure and metadata, not by a `case_id` field.

## Example

```yaml
description: "Basic test case"
surface_params:
  surface: "input.LandremanPaul2021_QA"  # Must match file in plasma_surfaces/
  range: "half period"  # or "full torus"
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: "L-BFGS-B"
  max_iterations: 200
  max_iter_subopt: 10
  verbose: False
  algorithm_options:
    ftol: 1e-6
    gtol: 1e-5
coil_objective_terms:
  total_length: "l2_threshold"
  coil_curvature: "lp_threshold"
  coil_curvature_p: 2
  coil_mean_squared_curvature: "l2_threshold"
  coil_arclength_variation: "l2_threshold"
  linking_number: ""
```

## Virtual Casing

Virtual casing allows you to compute the external magnetic field contribution from a VMEC equilibrium.
When enabled, the SquaredFlux objective uses `B_external_normal` as the target instead of zero,
which is useful for cases where you want the coils to produce a specific magnetic field configuration.

**Requirements:**
- Virtual casing requires a VMEC wout file (not input or FOCUS file)
- The `virtual_casing` Python package must be installed

**Example with virtual casing:**

```yaml
description: "Case with virtual casing"
surface_params:
  surface: "wout_schuetthenneberg_nfp2.nc"  # Must be a VMEC wout file
  range: "half period"
  virtual_casing: true  # Enable virtual casing with default parameters
coils_params:
  ncoils: 4
  order: 8
optimizer_params:
  algorithm: "L-BFGS-B"
  max_iterations: 1000
```

## Fastest Way to Run

Add a case under `cases/` and `git push`. CI will run the case and update the leaderboards.

## Running a Case Locally

```bash
stellcoilbench submit-case cases/case.yaml
```
