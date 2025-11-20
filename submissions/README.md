# Submissions

This directory holds **raw submissions** from coil optimization methods.  
Each submission is a folder containing at least:

- `metadata.yaml` – short description of the method/run
- `results.json` – output from `coilbench eval-bundle`

### Recommended layout

```text
submissions/
  MyAwesomeCoilOptimizer/
    v0.3.1/
      metadata.yaml
      results.json
    v0.3.2/
      metadata.yaml
      results.json
  AnotherMethod/
    2025-11-01_baseline/
      metadata.yaml
      results.json

