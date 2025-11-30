# StellCoilBench Workflow Guide

## Overview

This repository uses two main directories:

- **`cases/`** - Benchmark case definitions (input)
- **`submissions/`** - Submission results (output, generated)

## Directory Structure

```
stellcoilbench/
├── cases/                    # Benchmark case definitions
│   ├── basic_LandremanPaulQA.yaml
│   ├── basic_MUSE.yaml
│   ├── basic_tokamak.yaml
│   ├── basic_rotating_ellipse.yaml
│   └── README.md
├── submissions/              # Generated submission results (zipped)
│   ├── LandremanPaul2021_QA/  # Plasma surface name (from case.yaml)
│   │   └── akaptano/         # GitHub username
│   │       └── 11-23-2025_23-03.zip  # Date and time (MM-DD-YYYY_HH-MM).zip
│   │           # Contains: results.json, case.yaml, coils.json, *.vtu, *.vts
│   ├── MUSE.focus/           # Different plasma surface
│   │   └── akaptano/
│   │       └── 11-23-2025_23-03.zip
│   │           # Contains: results.json, case.yaml, coils.json, *.vtu, *.vts
│   └── README.md
└── docs/                     # Generated leaderboards
    └── leaderboards/         # Per-surface leaderboards
        ├── LandremanPaul2021_QA.md
        ├── muse_focus.md
        ├── circular_tokamak.md
        └── rotating_ellipse.md
```

## How to Add a Submission

### Step 1: Define a Case (if needed)

Create or use an existing `case.yaml` file in `cases/`:

```yaml
# cases/my_case.yaml
description: "My optimization test"
surface_params:
  surface: "input.LandremanPaul2021_QA"  # Must match file in plasma_surfaces/
  range: "half period"  # or "full torus"
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: "L-BFGS-B"  # or "BFGS", "SLSQP", "augmented_lagrangian", etc.
  max_iterations: 200
  max_iter_subopt: 10
  verbose: False  # Optional: controls optimization progress output (only in optimizer_params)
  algorithm_options:  # Optional: algorithm-specific hyperparameters
    ftol: 1e-6
    gtol: 1e-5
coil_objective_terms:  # Optional: specify which objectives to include
  total_length: "l2_threshold"
  coil_coil_distance: "l1_threshold"
  coil_surface_distance: "l1_threshold"
  linking_number: ""  # Empty string includes linking number
```

### Step 2: Run and Submit

Run the case to generate a submission:

```bash
stellcoilbench submit-case cases/my_case.yaml
```

**What this does:**
1. Runs the coil optimization for the case
2. Evaluates the results
3. Auto-detects GitHub username from git config (`git config user.name`)
4. Auto-detects hardware (CPU/GPU) from system information
5. Creates a submission directory `submissions/<surface>/<github_username>/<MM-DD-YYYY_HH-MM>/`
   - Surface name is extracted from `surface_params.surface` in case.yaml
   - Directory structure: `submissions/LandremanPaul2021_QA/akaptano/11-23-2025_23-03/`
6. Saves `results.json`, `coils.json`, `case.yaml`, `biot_savart_optimized.json`, and VTK files (*.vtu, *.vts) in the submission directory
7. Copies `case.yaml` to the submission directory with `source_case_file` field added (tracks which case file was used)
8. **Automatically zips** the entire submission directory into `submissions/<surface>/<github_username>/<MM-DD-YYYY_HH-MM>.zip`
9. Removes the original directory, leaving only the `.zip` file

**Directory naming:**
- Username: Auto-detected from git config (`git config user.name`)
- Timestamp: Current date and time in `MM-DD-YYYY_HH-MM` format (e.g., `12-20-2024_14-30`)

**Submission identification:**
Each submission is uniquely identified by:
- **Date submitted**: Timestamp in directory name (`MM-DD-YYYY_HH-MM`)
- **GitHub username**: Auto-detected from git config
- **Metadata**: All case parameters stored in `case.yaml` (copied to submission directory)

**Optional flags:**
- `--method-name` - Name of your optimization method (optional, stored in metadata)
- `--notes` - Add notes about the submission (optional, stored in metadata)

### Step 3: Commit and Push

Commit the generated submission zip file:

```bash
git add submissions/<surface>/<your_username>/<timestamp>.zip
git commit -m "Add submission"
git push
```

Replace `<surface>`, `<your_username>`, and `<timestamp>` with the actual values from your submission zip file.

**Note**: Submissions are automatically zipped. The zip file contains all submission files (results.json, case.yaml, coils.json, VTK files, etc.).

### Step 4: CI Updates Leaderboard

When you push, CI automatically:
1. Commits your submission zip file
2. **Only runs case files that don't have successful submissions yet** (skips cases that already have successful submissions)
3. Scans `submissions/` for all `.zip` files and extracts `results.json` from each
4. Generates `docs/leaderboard.json` and per-surface leaderboards in `docs/leaderboards/`
5. Commits the leaderboard files (`docs/`)

**Important**: 
- All files (`submissions/` and `docs/`) are committed to the repository
- Leaderboard files (`docs/`) are automatically generated and updated by CI

## How It Works

### `cases/` Directory
- **Purpose**: Defines benchmark cases (problem specifications)
- **Format**: YAML files with case configuration
- **Usage**: Input to `submit-case` command
- **Git**: Tracked in git (part of the benchmark definition)

### `submissions/` Directory  
- **Purpose**: Stores submission results (solution outputs)
- **Format**: `.zip` files organized by surface/username/timestamp (e.g., `11-23-2025_23-03.zip`)
- **Content**: Each zip file contains `results.json`, `case.yaml` (with `source_case_file` field), `coils.json`, VTK files, etc.
- **Usage**: Scanned by `update-db` to build leaderboard (extracts `results.json` from zip files)
- **Git**: Tracked in git (submissions are part of the repo)
- **Identification**: Each submission is identified by GitHub username, submission date/time, and `source_case_file` in `case.yaml` (which tracks which case file was used)

### Generated Files (`docs/`)
- **Purpose**: Aggregated database and per-surface leaderboards
- **Format**: JSON (docs/leaderboard.json) and Markdown (docs/leaderboards/)
- **Usage**: Displayed on GitHub
- **Git**: Tracked in git
- **Update**: Automatically generated and updated by CI when submissions are pushed

## Example Workflow

```bash
# 1. Create a case (or use existing)
cp cases/case.yaml cases/my_test_case.yaml
# Edit my_test_case.yaml as needed

# 2. Run and submit
stellcoilbench submit-case cases/my_test_case.yaml

# 3. Check the generated submission (zip file name will be timestamp)
ls submissions/*/$(git config user.name)/
# To view contents of a zip file:
unzip -l submissions/*/$(git config user.name)/*.zip

# 4. Commit and push
git add submissions/
git commit -m "Add submission"
git push

# 5. CI will update the leaderboard automatically
```

## Key Points

- **Cases** (`cases/`) = Problem definitions (what to optimize)
- **Submissions** (`submissions/`) = Results (your solutions, stored as `.zip` files)
- **Leaderboard files** (`docs/`) = Generated per-surface leaderboards and leaderboard.json
- **Put submissions in `submissions/`** - either:
  - Generate them with `submit-case` command (recommended, automatically creates zip files)
  - Or manually create zip files containing `results.json` following the format
- **CI scans `submissions/`** for `.zip` files to build per-surface leaderboards
- **Each submission** = one `.zip` file in `submissions/<surface>/<username>/<MM-DD-YYYY_HH-MM>.zip`
- **CI only runs cases without successful submissions** - cases with existing successful submissions are skipped
- **Leaderboard files are auto-generated** - `docs/` directory is created and updated by CI
- **All files are tracked** - `submissions/` and `docs/` are all committed to the repository

## Viewing the Leaderboard

The leaderboard is automatically updated:
- Browse: `https://github.com/<your-repo>/blob/main/docs/leaderboards/` (all surface leaderboards)
- Or browse individual surface leaderboards: `https://github.com/<your-repo>/blob/main/docs/leaderboards/<surface>.md`

