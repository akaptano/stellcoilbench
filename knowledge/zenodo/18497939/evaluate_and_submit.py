#!/usr/bin/env python3
"""
Evaluate Pedro Gil augmented Lagrangian filamentary coil solutions from Zenodo 18497939
and add them to StellCoilBench leaderboards.

Zenodo 18497939: "Coilsets and Scripts from Augmented Lagrangian Methods for Stellarator Coils"
(Gil et al., arXiv 2507.12681). These are filamentary coils for comparison with Gil Table 2 (QA) and Table 3 (QH).

Run from repo root:
    cd /path/to/stellcoilbench
    python knowledge/zenodo/18497939/evaluate_and_submit.py

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

# Config name (subdir) -> (surface_file, surface_range, already_reactor_scaled)
# QA: L18/L20/L24 and output_lengthtarget* are reactor-scaled (paper Table 2) → True
#     Other QA configs (LP_QA_total_length_*) may be 1m → False
# QH: reactor-scale surface and coils (Gil Table 3) → True
CONFIG_TO_SURFACE: dict[str, tuple[str, str, bool]] = {
    # QA (Landreman-Paul) - Gil Table 2, reactor-scaled
    "L18": ("input.LandremanPaul2021_QA", "half period", True),
    "L20": ("input.LandremanPaul2021_QA", "half period", True),
    "L24": ("input.LandremanPaul2021_QA", "half period", True),
    "LandremanPaul_QA": ("input.LandremanPaul2021_QA", "half period", False),
    "QA": ("input.LandremanPaul2021_QA", "half period", False),
    "LandremanPaul2021_QA": ("input.LandremanPaul2021_QA", "half period", False),
    # QH (reactor-scale) - Gil Table 3
    "LP_QH_total_length_4coils_setup_paper": (
        "input.LandremanPaul2021_QH_reactorScale_lowres", "half period", True
    ),
    "LP_QH_total_length_5coils_setup_paper": (
        "input.LandremanPaul2021_QH_reactorScale_lowres", "half period", True
    ),
    "coils_wiedman": (
        "input.LandremanPaul2021_QH_reactorScale_lowres", "half period", True
    ),
    "LandremanPaul_QH": ("input.LandremanPaul2021_QH_reactorScale_lowres", "half period", True),
    "QH": ("input.LandremanPaul2021_QH_reactorScale_lowres", "half period", True),
    "LandremanPaul2021_QH": ("input.LandremanPaul2021_QH_reactorScale_lowres", "half period", True),
}


def _extract_surface_name(surface_file: str) -> str:
    """Extract surface name for leaderboard path."""
    s = surface_file.replace("input.", "").replace(".focus", "").replace("wout.", "")
    return s.split("/")[-1].split(".")[0] if "." in s else s


def _infer_surface_from_config(config_name: str) -> tuple[str, str, bool]:
    """Infer surface and scaling from config name when not in explicit map."""
    name_lower = config_name.lower()
    if "qa" in name_lower or "quasiaxisymmetric" in name_lower or "lp_qa" in name_lower:
        return ("input.LandremanPaul2021_QA", "half period", False)
    if "qh" in name_lower or "quasihelical" in name_lower or "lp_qh" in name_lower or "wiedman" in name_lower:
        return ("input.LandremanPaul2021_QH_reactorScale_lowres", "half period", True)
    # Default to QA (1m scale)
    return ("input.LandremanPaul2021_QA", "half period", False)


OLD_USER_TO_REMOVE = "zenodo_18497939_Gil"


def main() -> int:
    zenodo_dir = Path(__file__).resolve().parent
    repo_root = zenodo_dir.parent.parent.parent
    plasma_surfaces_dir = repo_root / "plasma_surfaces"
    submissions_dir = repo_root / "submissions"

    # Remove previous leaderboard entries before writing new ones
    import shutil
    for surface_dir in submissions_dir.iterdir():
        if surface_dir.is_dir():
            old_dir = surface_dir / OLD_USER_TO_REMOVE
            if old_dir.exists():
                shutil.rmtree(old_dir)
                print(f"Removed old entries: {old_dir}")

    from stellcoilbench.coil_optimization import evaluate_external_coils
    from stellcoilbench.cli import _compute_reactor_scale_metrics
    from stellcoilbench.config_scheme import CaseConfig

    def _case_cfg(surface_file: str) -> CaseConfig:
        return CaseConfig.from_dict({
            "surface_params": {"surface": surface_file, "range": "half period"},
            "coils_params": {"ncoils": 4, "order": 16},
        })

    user = "zenodo_18497939_Gil"
    method_name = "Gil augmented Lagrangian (Zenodo 18497939)"

    config_dirs = [d for d in zenodo_dir.iterdir() if d.is_dir() and (d / "coils.json").exists()]
    if not config_dirs:
        print("No coil configs found. Run fetch_zenodo_coils.py --record-ids 18497939 first.")
        return 1

    for config_dir in sorted(config_dirs):
        config_name = config_dir.name
        coils_path = config_dir / "coils.json"
        cfg = CONFIG_TO_SURFACE.get(config_name, _infer_surface_from_config(config_name))
        surface_file, surface_range, already_reactor_scaled = cfg

        surface_name = _extract_surface_name(surface_file)
        sub_dir = submissions_dir / surface_name / user / config_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        print(f"Evaluating {config_name} -> {surface_name} (reactor_scaled={already_reactor_scaled})...")
        try:
            metrics = evaluate_external_coils(
                coils_json_path=coils_path,
                surface_file=surface_file,
                surface_range=surface_range,
                surface_resolution=32,
                plasma_surfaces_dir=plasma_surfaces_dir,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        reactor_scale_metrics = _compute_reactor_scale_metrics(
            metrics, _case_cfg(surface_file), already_reactor_scaled=already_reactor_scaled
        )

        submission = {
            "metadata": {
                "method_name": method_name,
                "contact": "Zenodo 18497939 / Gil et al.",
                "hardware": "Paper results",
                "notes": "Augmented Lagrangian filamentary coils (arXiv 2507.12681)",
                "run_date": "2025-02-22",
                "dipole_array": False,
            },
            "version_info": {"stellcoilbench": "zenodo_import", "simsopt": "unknown"},
            "metrics": metrics,
            "reactor_scale_metrics": reactor_scale_metrics,
        }

        metrics_clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
        submission["metrics"] = metrics_clean

        results_path = sub_dir / "results.json"
        results_path.write_text(json.dumps(submission, indent=2, default=str))
        print(f"  Wrote {results_path}")

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
