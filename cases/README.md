# Benchmark Cases

This directory contains benchmark case definitions in `case.yaml` format.

## Structure

Each `case.yaml` file defines:
- `description`: Brief description of the case
- `surface_params`: Plasma surface configuration
  - `surface`: Surface file name (must match a file in `plasma_surfaces/` directory, e.g., `input.LandremanPaul2021_QA`)
  - `range`: Surface range (`"half period"` or `"full torus"`)
- `coils_params`: Coil optimization parameters
  - `ncoils`: Number of coils (required, must be positive integer)
  - `order`: Fourier order for coil curves (required, must be positive integer)
- `optimizer_params`: Optimizer settings
  - `algorithm`: Optimization algorithm (e.g., `"L-BFGS-B"`, `"BFGS"`, `"SLSQP"`, `"augmented_lagrangian"`)
  - `max_iterations`: Maximum optimization iterations
  - `max_iter_subopt`: Maximum suboptimization iterations (for augmented Lagrangian)
  - `verbose`: Verbose flag (optional, boolean) - controls optimization progress output
    - **Note**: `verbose` is only valid in `optimizer_params`, not in `coils_params`
  - `algorithm_options`: Optional dictionary of algorithm-specific hyperparameters (e.g., `{"ftol": 1e-6, "gtol": 1e-5}` for L-BFGS-B)
- `coil_objective_terms`: Optional dictionary specifying which coil objectives to include
  - Each term can be specified with options like `"l1"`, `"l1_threshold"`, `"l2"`, `"l2_threshold"`, `"lp"`, `"lp_threshold"`
  - Unspecified terms are not included in the objective function
  - `linking_number: ""` (empty string) includes the linking number objective
  - `target_B` is automatically set based on the surface (not specified in case.yaml)

**Note**: Cases are identified by their directory structure and metadata, not by a `case_id` field. Each submission includes a copy of the `case.yaml` file used.

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
  algorithm: "L-BFGS-B"  # Options: "BFGS", "L-BFGS-B", "SLSQP", "augmented_lagrangian", etc.
  max_iterations: 200
  max_iter_subopt: 10
  verbose: False  # Optional: controls optimization progress output
  algorithm_options:  # Optional: algorithm-specific hyperparameters
    ftol: 1e-6  # Function tolerance (for L-BFGS-B)
    gtol: 1e-5  # Gradient tolerance (for L-BFGS-B)
coil_objective_terms:  # Optional: specify which objectives to include
  total_length: "l2_threshold"  # Options: "l2", "l2_threshold"
  coil_coil_distance: "l1_threshold"  # Options: "l1", "l1_threshold", "l2", "l2_threshold"
  coil_surface_distance: "l1_threshold"  # Options: "l1", "l1_threshold", "l2", "l2_threshold"
  coil_curvature: "lp_threshold"  # Options: "lp", "lp_threshold"
  coil_curvature_p: 2  # p-value for lp curvature (default: 2)
  coil_mean_squared_curvature: "l2_threshold"  # Options: "l2", "l2_threshold", "l1"
  linking_number: ""  # Empty string includes linking number (l2)
  coil_coil_force: "lp_threshold"  # Options: "lp", "lp_threshold"
  coil_coil_force_p: 2  # p-value for lp force (default: 2)
  coil_coil_torque: "lp_threshold"  # Options: "lp", "lp_threshold"
  coil_coil_torque_p: 2  # p-value for lp torque (default: 2)
```

## Running a Case

To run a case and generate a submission:

```bash
stellcoilbench submit-case cases/case.yaml
```

**Auto-detection**: GitHub username and hardware are automatically detected from your git config and system information. No flags needed!

This will:
1. Run the coil optimization for the case
2. Evaluate the results
3. Create a submission directory `submissions/<surface>/<github_username>/<MM-DD-YYYY_HH-MM>/`
4. Save `results.json`, `coils.json`, `case.yaml` (with `source_case_file` field), and VTK files in the submission directory
5. **Automatically zip** the submission directory into `submissions/<surface>/<github_username>/<MM-DD-YYYY_HH-MM>.zip`
6. Remove the original directory, leaving only the `.zip` file

**Submission identification**: Each submission is uniquely identified by:
- **Date submitted**: Timestamp in zip filename (`MM-DD-YYYY_HH-MM.zip`)
- **GitHub username**: Auto-detected from `git config user.name`
- **Source case file**: Tracked in `case.yaml` via `source_case_file` field (which case file was used)
- **Metadata**: All case parameters stored in `case.yaml` (copied to submission zip file)

## Leaderboard

After submitting, CI automatically:
- Commits your submission
- Generates `docs/leaderboard.json` and per-surface leaderboards in `docs/leaderboards/`
- Commits leaderboard files (`docs/`)

**Important**: 
- All files (`submissions/` and `docs/`) are committed to the repository
- Leaderboard files (`docs/`) are automatically generated and updated by CI

View the leaderboard:
- Browse: `https://github.com/<your-repo>/blob/main/docs/leaderboards/` (all surface leaderboards)
- Or browse individual surface leaderboards: `https://github.com/<your-repo>/blob/main/docs/leaderboards/<surface>.md`

