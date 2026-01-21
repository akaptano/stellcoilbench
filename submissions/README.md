# Submissions

This directory holds submission results generated from running benchmark cases.

## Workflow

1. Define a case in `cases/`
2. Run and submit:
   ```bash
   stellcoilbench submit-case cases/case.yaml
   ```
3. Results are written to `submissions/<surface>/<github_username>/<MM-DD-YYYY_HH-MM>.zip`
4. PDF plots are saved **next to** the zip file (not inside the archive)

## Directory Structure

```
submissions/
  LandremanPaul2021_QA/
    akaptano/
      11-23-2025_23-03.zip
      bn_error_3d_plot.pdf
      bn_error_3d_plot_initial.pdf
```

Each submission zip file contains:
- `results.json`
- `case.yaml` (with `source_case_file` field)
- `coils.json`
- `biot_savart_optimized.json`
- VTK files (`*.vtu`, `*.vts`)

## Leaderboard

CI scans `submissions/` for `.zip` files and regenerates `docs/leaderboards/` on push.
