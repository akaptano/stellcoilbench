# Pushing a Large Repo: Manual Steps

Automated LFS migration and filter-repo both hit blockers. Run these steps **manually in your terminal** (they need long runtimes and full repo access).

## Blocker Summary

1. **Git LFS migrate** – Needs 30+ minutes on a 16GB repo; timed out here.
2. **git filter-repo** – Failed with "Blob not found" (missing object `01073bc6...` in `cases/done/`). Likely due to partial/blobless clone or a corrupted local object store.

## Option A: LFS Migration (Recommended)

Run in your repo root. **Allow 30–45 minutes** for the migration:

```bash
cd /Users/aak572/stellcoilbench

# 1. Install and init LFS (if needed)
brew install git-lfs
git lfs install

# 2. Track submissions
git lfs track "submissions/**"

# 3. Stash uncommitted changes
git stash --include-untracked

# 4. Migrate (rewrites history; answer 'y' when prompted)
git lfs migrate import --include="submissions/**" --everything

# 5. Re-add remote (migrate removes it)
git remote add origin https://github.com/akaptano/stellcoilbench.git

# 6. Force push
git push --force origin main

# 7. Restore stash if desired
git stash pop
```

## Option B: Full Clone + filter-repo (If LFS Fails)

If LFS migrate fails or times out, use a **fresh full clone** and then filter-repo:

```bash
cd /tmp
git clone https://github.com/akaptano/stellcoilbench.git stellcoilbench-clean
cd stellcoilbench-clean

# Remove submissions from all history
pip install git-filter-repo
git filter-repo --path submissions/ --invert-paths --force

# Re-add remote and push
git remote add origin https://github.com/akaptano/stellcoilbench.git
git push --force origin main
```

**Warning:** This keeps the **remote** history and removes submissions. Your local 708 extra commits and changes will not be pushed. Only use this if you want to reset the remote to a version without submissions.

## Option C: Remove submissions Going Forward Only (No History Rewrite)

If you only want to stop tracking submissions from now on (history stays large, push will still fail):

```bash
git rm -r --cached submissions/
echo "submissions/" >> .gitignore
git add .gitignore
git commit -m "chore: stop tracking submissions"
# Push will still fail (2GB limit) since history still contains the blobs
```

## Current State

- LFS is installed and initialized.
- `submissions/**` LFS tracking was added then reverted.
- Origin remote is present.
- Stash holds any pre-migration uncommitted changes (see `git stash list`).
