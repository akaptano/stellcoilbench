# StellCoilBench

A repository for automated benchmarking of stellarator coil optimization techniques.

## Quick Start

1. **Run a benchmark case:**
   ```bash
   stellcoilbench submit-case cases/case.yaml
   ```

2. **Commit and push:**
   ```bash
   git add submissions/
   git commit -m "Add submission"
   git push
   ```

3. **View the leaderboard:**
   - Browse `docs/surfaces.md` on GitHub
   - Or view locally: `cat docs/surfaces.md`

## How It Works

- **Submissions** (`submissions/`) are committed to the `main` branch
- **Leaderboard files** (`docs/`, `db/`) are committed to the `main` branch
- CI automatically updates the leaderboard when you push submissions
- All files are on the `main` branch

## Documentation

- **[Workflow Guide](WORKFLOW.md)** - Complete guide to using StellCoilBench
- **[Submissions](submissions/README.md)** - How to create and submit results
- **[Cases](cases/README.md)** - How to define benchmark cases

## Leaderboard

The leaderboard is automatically generated and available on the `main` branch:
- View online: Browse `docs/surfaces.md` on GitHub
- View locally: `cat docs/surfaces.md` or browse `docs/surfaces/` directory
