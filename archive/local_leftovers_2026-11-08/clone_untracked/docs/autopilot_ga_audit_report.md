# Autopilot Genetic Algorithm Performance Audit

**Generated:** Run `python -m tools.audit_autopilot_ga --repo-root . --out docs/autopilot_ga_audit.json` to refresh.

**Plots:** Add `--plots-dir docs/` to generate:
- `autopilot_composite_score_vs_date.png` — composite score vs date (scatter)
- `autopilot_exploit_vs_explore.png` — exploit/explore/unknown counts by day
- `autopilot_cumulative_best.png` — running max composite score over time
- `autopilot_median_by_date.png` — daily median and best composite score
- `autopilot_exploit_vs_explore_scores.png` — boxplot of composite by tag (requires tagged runs)
- `autopilot_parent_child_delta.png` — histogram of child−parent composite (requires exploit runs with parent_ids)

## Summary

| Metric | Value |
|--------|-------|
| Total successful runs | 2,722 |
| Total failures (run-time) | 22 |
| Overall fail rate | 0.8% |
| Constraint failures (excluded from leaderboard) | 412 |
| Total runs | 2,744 |

## Composite Score

- **Median:** 1.65
- **Best:** 1.91 (Landreman-Paul QA, 5 coils, FC 4,8,16)
- **Worst:** 0.0 (constraint violations)
- **Quartiles (Q1–Q3):** 1.41, 1.65, 1.73

## Flux (score_primary, lower is better)

- **Best:** 4.77e-08
- **Median:** 2.94e-05
- **Quartiles:** 1.82e-06, 2.94e-05, 0.0029

## N_turns (reactor-scale max per coil)

- **Min / Max:** 1 / 2,020,076
- **Median:** 490
- **Over 300 (soft bound):** 2,526 of 2,721 (93%)

Most designs exceed the N_turns soft bound (300). The top 10 by composite score have N_turns_max between 336 and 415.

## Exploit vs Explore

Autopilot batches mix **exploit** (mutation of top parents) and **explore** (random configs). Tags are stored in submission metadata for new runs. Historical runs show as "unknown".

## Failure Analysis

All 22 run-time failures are **nan_in_objective** ("NaN in objective function"). No other failure classes (timeout, vmec_nonconverged, etc.) appear in the recent failure log.

## By Surface

| Surface | Count | Composite median | Flux best |
|---------|-------|------------------|-----------|
| LandremanPaul2021_QA | 2,578 | 1.67 | 4.77e-08 |
| LandremanPaul2021_QH_reactorScale_lowres | 142 | 0.0* | 3.16e-07 |
| TestSurface | 2 | 1.79 | 2e-07 |

\* Many QH lowres runs fail reactor-scale constraints (coil linking, curvature, N_turns, etc.), so median composite is 0.

## Top 10 by Composite Score

All top 10 use **Landreman-Paul QA**, **Fourier continuation 4,8,16**, and **4–5 coils**:

1. 2026-03-15_031510_17672 — comp=1.91, flux=6.29e-05, N=5, N_turns_max=336
2. 2026-03-16_150116_22450 — comp=1.88, flux=8.75e-05, N=4, N_turns_max=371
3. 2026-03-16_231538_44195 — comp=1.88, flux=1.14e-04, N=4, N_turns_max=390
4. 2026-03-14_075836_80253 — comp=1.87, flux=3.86e-05, N=4, N_turns_max=411
5. 2026-03-17_091532_51032 — comp=1.87, flux=3.96e-05, N=4, N_turns_max=411
6. 2026-03-18_093359_37359 — comp=1.87, flux=4.16e-05, N=4, N_turns_max=412
7. 2026-03-16_003415_82908 — comp=1.87, flux=4.87e-05, N=4, N_turns_max=415
8. 2026-03-17_062506_61075 — comp=1.87, flux=3.58e-05, N=4, N_turns_max=410
9. 2026-03-17_072829_25921 — comp=1.87, flux=2.97e-05, N=4, N_turns_max=406
10. 2026-03-13_192014_46896 — comp=1.87, flux=3.22e-05, N=4, N_turns_max=408

## Progress Assessment

- **First 200 vs last 200 runs:** Median composite improved (last 1.71 vs first 1.70); best improved (last 1.87 vs first 1.84). Pass rate declined (89.5% vs 100%) as more runs hit constraint limits.
- **Exploit vs explore:** Requires tagged runs (`metadata.tags`). Historical runs show as "unknown"; new runs store `exploit` or `explore` for comparison.
- **Parent-child lineage:** Exploit runs with `metadata.parent_ids` enable improvement-rate analysis. Zero comparable pairs when no exploit runs have stored parent_ids.
- **Convergence:** Rolling std (window=50) — early 0.16, recent 0.49. Increasing variance suggests continued exploration rather than convergence.

## Conclusions

- **GA proposer is effective** on Landreman-Paul QA: 2,578 runs, median composite 1.67, best flux 4.77e-08.
- **N_turns** is frequently above the 300 soft bound; top designs cluster around 330–415 turns.
- **Landreman-Paul QH reactor-scale lowres** is challenging: 142 runs, many constraint failures (linking, curvature, N_turns).
- **Run-time stability** is high: only 22 NaN failures in 2,744 runs.
