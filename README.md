# StellCoilBench

**Open benchmark suite for stellarator coil optimization algorithms.** Standardized case definitions (YAML), automated optimization via simsopt, post-processing (VMEC, Poincaré plots), and CI-driven leaderboards.

## Quick Start

```bash
# Add a case and push — CI runs it and updates leaderboards
stellcoilbench submit-case cases/my_case.yaml
git add submissions/ && git commit -m "Add submission" && git push
```

**Or:** Add a case under `cases/` and `git push` — CI will run it automatically.

## Repository Layout

| Directory | Purpose |
|-----------|---------|
| `cases/` | Benchmark case definitions (YAML). See `cases/README.md` |
| `cases/pending/` | Autopilot queue (JSON, written by proposer) |
| `cases/done/` | Autopilot results |
| `submissions/<surface>/<user>/<timestamp>/` | Submission zips and PDFs |
| `submissions/<surface>/auto/` | Autopilot submissions |
| `docs/leaderboards/` | Per-surface leaderboards (CI-generated) |
| `policy/proposer_policy.yaml` | Autopilot tuning and guardrails |

## Commands

```bash
stellcoilbench submit-case cases/case.yaml   # Run a case locally
stellcoilbench run-ci-case cases/pending/X.json  # Run autopilot case
stellcoilbench update-db                     # Rebuild leaderboards from submissions
stellcoilbench validate-config cases/X.yaml # Validate case file
```

## Autopilot

Continuous CI loop: propose → run → record. Create `PAUSE_AUTORUN` to halt.

```bash
python tools/propose_batch.py --batch-size 3 --dry-run   # Preview
python tools/build_context.py | python -m json.tool      # Inspect context
```

## Documentation

- **ReadTheDocs**: https://stellcoilbench.readthedocs.io/
- **Cases**: `cases/README.md`, `docs/cases.rst`
- **Autopilot**: `docs/autopilot.rst`
- **Leaderboard**: `docs/leaderboard.rst`, `docs/leaderboard/metric_definitions.rst`
