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

## Git Configuration

The `db/`, `docs/`, and `submissions/` directories are updated by CI and should prefer remote versions when pulling.

### Quick Solution

**To pull without conflicts, use the git alias (recommended):**
```bash
git pull-safe
```

This will automatically clean untracked files in these directories and pull with the `theirs` strategy (prefers remote versions).

### Alternative Methods

**Option 1: Manual cleanup then pull**
```bash
# Remove untracked files that would conflict
git clean -fd db/ docs/ submissions/

# Pull with strategy that prefers remote
git pull -X theirs
```

**Option 2: Use the helper script**
```bash
./scripts/git-pull-safe.sh
```

**Option 3: Configure git to always prefer remote for these paths**
The `.gitattributes` file is already configured. You can also set:
```bash
git config --local pull.strategy-option theirs
```

### Why This Is Needed

When CI updates `db/`, `docs/`, or `submissions/` on the remote, and you have local untracked files in these directories, git will refuse to pull to avoid overwriting your local files. Since these directories are managed by CI, you typically want to accept the remote version.

### Configuration Files

- **`.gitattributes`** - Configures merge strategies to prefer remote versions
- **`scripts/git-pull-safe.sh`** - Helper script for safe pulling
- **Git alias `pull-safe`** - Convenient alias (configured automatically)

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
