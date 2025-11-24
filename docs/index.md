# StellCoilBench

StellCoilBench is an open benchmark suite for stellarator coil optimization algorithms.

- **Benchmark cases** live under `cases/`.
- **Submissions** live under `submissions/`.
- Aggregated results are stored in `docs/leaderboard.json` and shown in the [Surface Leaderboards](leaderboards/).

## Leaderboards

- **[LandremanPaul2021_QA](leaderboards/LandremanPaul2021_QA.md)** - QA stellarator surface
- **[MUSE Focus](leaderboards/muse_focus.md)** - MUSE focus configuration
- **[Circular Tokamak](leaderboards/circular_tokamak.md)** - Circular tokamak surface
- **[Rotating Ellipse](leaderboards/rotating_ellipse.md)** - Rotating ellipse surface

## Updating the Leaderboard

The leaderboard is automatically updated by CI when submissions are pushed to the `main` branch.

To update locally after adding submissions, run:

```bash
stellcoilbench update-db

