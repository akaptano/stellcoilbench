#!/usr/bin/env python3
"""
Evaluate Kaptanoglu dipole coil solutions from Zenodo 14934092
and add them to StellCoilBench dipole leaderboards.

Zenodo 14934092: "Reactor-scale Stellarator Dipole Array Solutions"
(Kaptanoglu et al., arXiv 2412.13937). These are dipole+TF coils, not filamentary.

Loads each final coil file, computes dipole metrics (flux, B·n, dipole_metrics, tf_metrics),
writes results.json to submissions/, then runs update-db to regenerate leaderboards.

Run from repo root:
    cd /path/to/stellcoilbench
    python knowledge/zenodo/14934092/evaluate_and_submit.py

Requires: stellcoilbench_vmec conda environment, simsopt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Config name -> (surface_file, surface_range, ncoils_dipole)
# ncoils_dipole: number of dipole coils (first N in coil list; rest are TF)
# Parsed from config name: n164 -> 164, n41 -> 41, etc.
CONFIG_TO_SURFACE: dict[str, tuple[str, str, int]] = {
    "QA_continuation_TForder4_n164_p1_50e_00_c1_50e_00_lw1_00e-02_lt1_15e_02_lkw1_00e": (
        "input.LandremanPaul2021_QA",
        "half period",
        164,
    ),
    "QA_henneberg_continuation_TForder16_n64_p1_50e_00_c2_00e_00_lw1_00e-02_lt7_50e_0": (
        "input.LandremanPaul2021_QA",
        "half period",
        64,
    ),
    "QA_henneberg_shape_TForder16_DCorder_0_n18_p1_50e_00_c1_50e_00_lw1_00e-03_lt8_50": (
        "input.LandremanPaul2021_QA",
        "half period",
        18,
    ),
    "QA_minimal_TForder4_n41_p1_50e_00_c1_50e_00_lw1_00e-02_lt1_00e_02_lkw1_00e_03_cc": (
        "input.LandremanPaul2021_QA",
        "half period",
        41,
    ),
    "QH_continuation_fixed_TForder4_n216_lw1_00e-02_lt9_00e_01_lkw1_00e_04_cct8_00e-0": (
        "input.LandremanPaul2021_QH_reactorScale_lowres",
        "half period",
        216,
    ),
    "QH_minimal_TForder4_n27_p1_50e_00_c2_50e_00_lw1_00e-02_lt8_00e_01_lkw1_00e_04_cc": (
        "input.LandremanPaul2021_QH_reactorScale_lowres",
        "half period",
        27,
    ),
}


def _extract_surface_name(surface_file: str) -> str:
    """Extract surface name for leaderboard path (e.g. LandremanPaul2021_QA)."""
    s = surface_file.replace("input.", "").replace(".focus", "").replace("wout.", "")
    return s.split("/")[-1].split(".")[0] if "." in s else s


OLD_USERS_TO_REMOVE = (
    "zenodo_14934092_Gil",  # Old filamentary evaluation
    "zenodo_14934092_Kaptanoglu",  # Remove before writing to avoid stale duplicates on rerun
)


def main() -> int:
    zenodo_dir = Path(__file__).resolve().parent
    repo_root = zenodo_dir.parent.parent.parent
    plasma_surfaces_dir = repo_root / "plasma_surfaces"
    submissions_dir = repo_root / "submissions"

    # Remove old entries before writing dipole entries
    import shutil
    surfaces = {_extract_surface_name(sf) for sf, _, _ in CONFIG_TO_SURFACE.values()}
    for surface_name in surfaces:
        for old_user in OLD_USERS_TO_REMOVE:
            old_dir = submissions_dir / surface_name / old_user
            if old_dir.exists():
                shutil.rmtree(old_dir)
                print(f"Removed old entries: {old_dir}")

    from stellcoilbench.coil_optimization import evaluate_external_dipole_coils

    user = "zenodo_14934092_Kaptanoglu"

    for config_name, (surface_file, surface_range, ncoils_dipole) in CONFIG_TO_SURFACE.items():
        coils_path = zenodo_dir / config_name / "coils.json"
        if not coils_path.exists():
            print(f"Skip: {coils_path} not found")
            continue

        surface_name = _extract_surface_name(surface_file)
        sub_dir = submissions_dir / surface_name / user / config_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        print(f"Evaluating {config_name} -> {surface_name} (ncoils_dipole={ncoils_dipole})...")
        try:
            metrics = evaluate_external_dipole_coils(
                coils_json_path=coils_path,
                surface_file=surface_file,
                ncoils_dipole=ncoils_dipole,
                surface_range=surface_range,
                surface_resolution=32,
                plasma_surfaces_dir=plasma_surfaces_dir,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        submission = {
            "metadata": {
                "contact": "Zenodo 14934092 / Kaptanoglu et al.",
                "method_version": "Kaptanoglu dipole (Zenodo 14934092)",
                "hardware": "Paper results",
                "notes": "Reactor-scale dipole array solutions (arXiv 2412.13937)",
                "run_date": "2025-02-22",
                "dipole_array": True,
            },
            "version_info": {"stellcoilbench": "zenodo_import", "simsopt": "unknown"},
            "metrics": metrics,
        }

        # Remove internal keys from metrics for JSON serialization
        metrics_clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
        submission["metrics"] = metrics_clean

        results_path = sub_dir / "results.json"
        results_path.write_text(json.dumps(submission, indent=2, default=str))
        print(f"  Wrote {results_path}")

        # Copy coils.json so update-db can extract num_coils/coil_order if needed
        coils_dest = sub_dir / "coils.json"
        if not coils_dest.exists():
            import shutil
            shutil.copy(coils_path, coils_dest)

    print("\nRunning update-db to regenerate leaderboards...")
    import subprocess
    result = subprocess.run(
        ["stellcoilbench", "update-db"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"update-db stderr: {result.stderr}")
        print(f"update-db stdout: {result.stdout}")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
