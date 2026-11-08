# Local leftovers from deleted checkouts (2026-11-08)

Files and patches that existed only on the Mac before the local checkouts
`~/stellcoilbench_clone`, `~/old_scb/stellcoilbench`, `~/old_scb/stellcoilbench_fork`
and `~/old_stellcoilbench_submissions` were deleted.

- `clone_untracked/`: untracked files from `stellcoilbench_clone` (main was fully pushed).
  The Zenodo download cache was left out.
- `clone_stash_leaderboard.patch`: stash on `stellcoilbench_clone` (generated leaderboard docs).
- `old_scb_main_unpushed_code.patch`: four 2026-03-11 commits on the old pre-LFS `main`
  with no counterpart on GitHub, restricted to code (no submissions/ or cases/).
- `old_scb_untracked/`: untracked file from the old pre-LFS checkout.
- `fork_untracked/`: untracked notes and scripts from `stellcoilbench_fork`
  (sweep_output/ data and the coverage database were left out).
- `fork_stash_src_docs.patch`: stash on `stellcoilbench_fork`, src/ and docs/ only (no docs/_build).

Not kept: the local branch `backup-pre-lfs-migration` (its src/ and tests/ trees match
GitHub commits of 2026-03-14; the rest is CI autopilot commits rewritten by the LFS migration),
the pre-LFS history of the old checkout, and ignored output data.
