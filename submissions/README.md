# Submissions

This directory holds **submission results** generated from running benchmark cases.

## Workflow

1. **Define a case** in `cases/case.yaml` with your benchmark configuration
2. **Run and submit** using:
   ```bash
   stellcoilbench submit-case cases/case.yaml
   ```
   
   GitHub username and hardware are **auto-detected** from your git config and system info.
   You can override with `--contact` and `--hardware` if needed.
   
   Optional: Add `--method-name` and `--version` to store method info in metadata (not used for directory structure).

3. **Results** are automatically written to `submissions/<github_username>/<MM-DD-YYYY_HH-MM>/results.json`

## Directory Structure

```
submissions/
  akaptano/                    # GitHub username
    12-20-2024_14-30/          # Date and time (MM-DD-YYYY_HH-MM)
      results.json              # Contains metadata + case results with metrics
      case.yaml                 # Copy of the case.yaml used to generate this submission
    12-20-2024_15-00/          # Another submission from the same user
      results.json
      case.yaml
  another_user/                # Different GitHub user
    12-20-2024_14-00/
      results.json
      case.yaml
```

Each submission directory is named with the timestamp (to minute accuracy) when it was created.

Each submission directory contains:
- **`results.json`** - Submission results with metadata, case_id, and metrics
- **`case.yaml`** - Copy of the case definition file used to run this submission

## Results.json Format

Each `results.json` file contains:
- `metadata`: Method name, version, contact, hardware, run date
- `cases`: Array of case results with `case_id` and `metrics`

## Case.yaml

The `case.yaml` file is automatically copied from `cases/` when you run `submit-case`. This ensures each submission includes the exact case configuration that was used, making it easy to reproduce or understand what benchmark was run.

The `update-db` command scans this directory for `results.json` files to build the leaderboard.

