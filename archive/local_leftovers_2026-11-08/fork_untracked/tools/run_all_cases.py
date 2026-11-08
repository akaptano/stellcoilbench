#!/usr/bin/env python3
"""Run all non-structural cases with default params (no post-processing), then print metrics table."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = REPO_ROOT / "cases"
SUBS_DIR = REPO_ROOT / "submissions"
EXCLUDE = {"basic_LandremanPaulQA_structural.yaml"}

METRICS_OF_INTEREST = [
    "final_squared_flux",
    "final_total_length",
    "final_min_cc_separation",
    "final_min_cs_separation",
    "final_mean_squared_curvature",
    "iterations_used",
]


def main() -> None:
    cases = sorted(p for p in CASES_DIR.glob("*.yaml") if p.name not in EXCLUDE)
    results: list[dict] = []

    for i, case_path in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] Running {case_path.name}...", flush=True)
        r = subprocess.run(
            ["stellcoilbench", "submit-case", str(case_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if r.returncode != 0:
            results.append(
                {
                    "case": case_path.stem,
                    "status": "FAILED",
                    **{k: "-" for k in METRICS_OF_INTEREST},
                }
            )
            err = (r.stderr or r.stdout or "")[:150]
            print(f"  FAILED: {err}", flush=True)
            continue

        # Find most recent results.json for this case (submissions/<surface>/<user>/<case>/<dt>/)
        found: Path | None = None
        stem = case_path.stem
        for rj in SUBS_DIR.rglob("results.json"):
            if not rj.exists():
                continue
            # Path like submissions/X/user/CASENAME/datetime/results.json
            parts = rj.relative_to(SUBS_DIR).parts
            if len(parts) >= 4 and parts[2] == stem:
                if found is None or rj.stat().st_mtime > found.stat().st_mtime:
                    found = rj
        if found is None:
            results.append(
                {
                    "case": case_path.stem,
                    "status": "NO_RESULTS",
                    **{k: "-" for k in METRICS_OF_INTEREST},
                }
            )
            continue

        data = json.loads(found.read_text())
        metrics = data.get("metrics", data)
        row: dict = {"case": case_path.stem, "status": "OK"}
        for m in METRICS_OF_INTEREST:
            if m in metrics:
                v = metrics[m]
                row[m] = f"{v:.4g}" if isinstance(v, float) else v
            else:
                row[m] = "-"
        results.append(row)

    # Print table
    cols = ["case", "status"] + METRICS_OF_INTEREST
    w = 18
    sep = " | "
    header = sep.join(c.ljust(w)[:w] for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for row in results:
        line = sep.join(str(row.get(c, "-")).ljust(w)[:w] for c in cols)
        print(line)
    print("=" * len(header))
    return None


if __name__ == "__main__":
    main()
    sys.exit(0)
