# Benchmark Cases

This directory contains benchmark case definitions in `case.yaml` format.

## Structure

Each `case.yaml` file defines:
- `description`: Brief description of the case
- `surface_params`: Plasma surface configuration
  - `nphi`, `ntheta`: Surface discretization
  - `surface`: Surface file name (e.g., `input.LandremanPaul2021_QA`)
  - `range`: Surface range (`"half period"` or `"full torus"`)
- `coils_params`: Coil optimization parameters
  - `ncoils`: Number of coils
  - `order`: Fourier order for coil curves
  - `nturns`: Number of turns per coil
  - `target_B`: Target magnetic field strength (Tesla)
  - `coil_radius`: Coil radius for regularization
- `optimizer_params`: Optimizer settings
  - `algorithm`: Optimization algorithm (e.g., `"l-bfgs"`)
  - `max_iterations`: Maximum optimization iterations
  - `max_iter_lag`: Maximum Lagrangian iterations

**Note**: Cases are identified by their directory structure and metadata, not by a `case_id` field. Each submission includes a copy of the `case.yaml` file used.

## Example

```yaml
description: "Basic test case"
surface_params:
  nphi: 16
  ntheta: 16
  surface: "input.LandremanPaul2021_QA"
  range: "half period"
coils_params:
  ncoils: 4
  order: 16
  nturns: 200
  target_B: 1.0
  coil_radius: "0.05"
optimizer_params:
  algorithm: "l-bfgs"
  max_iterations: 200
  max_iter_lag: 10
  verbose: False
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
3. Generate `submissions/<surface>/<github_username>/<MM-DD-YYYY_HH-MM>/results.json`
4. Save `coils.json` (optimized coil geometry) in the submission directory
5. Copy `case.yaml` to the submission directory for reference
6. Save VTK visualization files (*.vtu, *.vts) in the submission directory

**Submission identification**: Each submission is uniquely identified by:
- **Date submitted**: Timestamp in directory name (`MM-DD-YYYY_HH-MM`)
- **GitHub username**: Auto-detected from `git config user.name`
- **Metadata**: All case parameters stored in `case.yaml` (copied to submission directory)

## Leaderboard

After submitting, CI automatically:
- Commits your submission to the `main` branch
- Generates per-surface leaderboards in `docs/surfaces/` on the `main` branch
- Commits leaderboard files (`db/` and `docs/`) to the `main` branch

**Important**: 
- All files (`submissions/`, `db/`, and `docs/`) are committed to the `main` branch
- Leaderboard files (`db/` and `docs/`) are automatically generated and updated by CI on `main`

View the leaderboard:
- Browse: `https://github.com/<your-repo>/blob/main/docs/surfaces.md` (index of all surfaces)
- Or browse individual surface leaderboards: `https://github.com/<your-repo>/blob/main/docs/surfaces/<surface>.md`

