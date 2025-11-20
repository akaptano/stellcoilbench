
---

## 5. `docs/index.md` and `docs/leaderboard.md`

### `docs/index.md` (simple starter)

```markdown
# CoilBench

CoilBench is an open benchmark suite for stellarator coil optimization algorithms.

- **Benchmark cases** live under `cases/`.
- **Submissions** live under `submissions/`.
- Aggregated results are stored in `db/` and shown on the [Leaderboard](leaderboard.md).

To update the database and leaderboard after new submissions are added, run:

```bash
coilbench update-db

