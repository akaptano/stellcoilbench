# Submissions

This directory holds **submission results** generated from running benchmark cases.

## Workflow

1. **Define a case** in `cases/case.yaml` with your benchmark configuration
2. **Run and submit** using:
   ```bash
   stellcoilbench submit-case cases/case.yaml \
     --method-name my_method \
     --version v1.0.0 \
     --contact your@email.com \
     --hardware "CPU/GPU description"
   ```
3. **Results** are automatically written to `submissions/<method_name>/<version>/results.json`

## Directory Structure

```
submissions/
  my_method/
    v1.0.0/
      results.json    # Contains metadata + case results with metrics
    v1.1.0/
      results.json
  another_method/
    v1.0.0/
      results.json
```

## Results.json Format

Each `results.json` file contains:
- `metadata`: Method name, version, contact, hardware, run date
- `cases`: Array of case results with `case_id` and `metrics`

The `update-db` command scans this directory to build the leaderboard.

