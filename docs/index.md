# StellCoilBench

StellCoilBench is an open benchmark suite for stellarator coil optimization algorithms.

## Fastest Way to Run

Add a case under `cases/` and `git push`. CI will run the case and update the leaderboards.

## Leaderboards

- **[LandremanPaul2021_QA](leaderboards/LandremanPaul2021_QA.md)**
- **[MUSE Focus](leaderboards/muse_focus.md)**
- **[Circular Tokamak](leaderboards/circular_tokamak.md)**
- **[Rotating Ellipse](leaderboards/rotating_ellipse.md)**

## Updating the Leaderboard

The leaderboard is updated by CI when submissions are pushed. To update locally after adding submissions:

```bash
stellcoilbench update-db
```
