# Benchmark Cases

This directory contains benchmark case definitions in `case.yaml` format.

## Structure

Each `case.yaml` file defines:
- `case_id`: Unique identifier for the case
- `description`: Brief description
- `surface_params`: Plasma surface configuration
- `coils_params`: Coil optimization parameters
- `optimizer_params`: Optimizer settings
- `output`: Output file naming patterns

## Example

```yaml
case_id: "my_benchmark_case"
description: "QA configuration test"
surface_params:
  nphi: 32
  ntheta: 32
  surface: "input.LandremanPaul2021_QA"
  range: "half period"
coils_params:
  ncoils: 4
  order: 16
  nturns: 200
  target_B: 1.0
optimizer_params:
  algorithm: "l-bfgs"
  max_iterations: 200
```

## Running a Case

To run a case and generate a submission:

```bash
stellcoilbench submit-case cases/case.yaml
```

GitHub username and hardware are **auto-detected** automatically!

This will:
1. Run the optimization for the case
2. Evaluate the results
3. Generate `submissions/<surface>/<github_username>/<MM-DD-YYYY_HH-MM>/results.json`
4. Save `coils.json` (optimized coil geometry) in the submission directory
5. Copy `case.yaml` to the submission directory for reference
6. Save VTK visualization files in the submission directory

The directory structure uses your GitHub username and the current date/time (to minute accuracy).

## Leaderboard

After submitting, CI automatically:
- Commits your submission to the `main` branch
- Generates and commits leaderboard files to the `leaderboard` branch
- You don't need to pull leaderboard files - they're on a separate branch

View the leaderboard by browsing the `leaderboard` branch on GitHub.

