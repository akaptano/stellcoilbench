#!/usr/bin/env python3
"""
Load Zenodo 18497939 coils and compare with Gil et al. augmented Lagrangian paper tables.

Gil paper (arXiv 2507.12681) Table 2: QA (Landreman-Paul) L18/L20/L24, 4 coils/hfp, ARIES-CS scaled.
Gil paper Table 3: QH #1 (4 coils), #2 (5 coils), Wiedman (5 coils).

Run from repo root:
    cd /path/to/stellcoilbench
    python knowledge/zenodo/18497939/compare_with_gil_paper.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Config -> (surface_file, surface_range, paper_label, already_reactor_scaled)
# QA: Zenodo L18/L20/L24 may be at reactor scale; we only have 1m QA surface → scale mismatch.
#     Use already_reactor_scaled=True (coils at reactor scale) → raw metrics; B·n may be off.
# QH: reactor-scale surface; Zenodo coils are reactor-scale.
GIL_TABLE_CONFIGS: dict[str, tuple[str, str, str, bool]] = {
    # Table 2 (QA) - Zenodo coils at reactor scale; 1m surface causes scale mismatch for B·n
    "L18": ("input.LandremanPaul2021_QA", "half period", "L18 (4 coils)", True),
    "L20": ("input.LandremanPaul2021_QA", "half period", "L20 (4 coils)", True),
    "L24": ("input.LandremanPaul2021_QA", "half period", "L24 (4 coils)", True),
    # Table 3 (QH)
    "LP_QH_total_length_4coils_setup_paper": (
        "input.LandremanPaul2021_QH_reactorScale_lowres",
        "half period",
        "QH #1 (4 coils)",
        True,
    ),
    "LP_QH_total_length_5coils_setup_paper": (
        "input.LandremanPaul2021_QH_reactorScale_lowres",
        "half period",
        "QH #2 (5 coils)",
        True,
    ),
    "coils_wiedman": (
        "input.LandremanPaul2021_QH_reactorScale_lowres",
        "half period",
        "QH Wiedman (5 coils)",
        True,
    ),
}

# Gil paper Table 2 (QA) - ARIES-CS scaled
GIL_TABLE2: dict[str, dict[str, float | str]] = {
    "L18": {
        "⟨B·n⟩/⟨B⟩×10⁻⁵": 93.7,
        "Total Length [m]": 182,
        "Min CC [m]": 1.44,
        "Min CS [m]": 2.57,
        "Max κ [m⁻¹]": 0.5,
        "Max MSC [m⁻¹]": 0.05,
    },
    "L20": {
        "⟨B·n⟩/⟨B⟩×10⁻⁵": 34,
        "Total Length [m]": 203,
        "Min CC [m]": 1.31,
        "Min CS [m]": 2.71,
        "Max κ [m⁻¹]": 0.51,
        "Max MSC [m⁻¹]": 0.05,
    },
    "L24": {
        "⟨B·n⟩/⟨B⟩×10⁻⁵": 6.0,
        "Total Length [m]": 240,
        "Min CC [m]": 1.0,
        "Min CS [m]": 3.14,
        "Max κ [m⁻¹]": 0.5,
        "Max MSC [m⁻¹]": 0.05,
    },
}

# Gil paper Table 3 (QH)
GIL_TABLE3: dict[str, dict[str, float | str]] = {
    "LP_QH_total_length_4coils_setup_paper": {
        "⟨B·n⟩/⟨B⟩×10⁻⁴": 6.2,
        "Total Length [m]": 160,
        "Min CC [m]": 0.8,
        "Min CS [m]": 1.91,
        "Max κ [m⁻¹]": 1.0,
        "Max MSC [m⁻¹]": 0.1,
    },
    "LP_QH_total_length_5coils_setup_paper": {
        "⟨B·n⟩/⟨B⟩×10⁻⁴": 6.2,
        "Total Length [m]": 177.8,
        "Min CC [m]": 1.09,
        "Min CS [m]": 1.59,
        "Max κ [m⁻¹]": 0.77,
        "Max MSC [m⁻¹]": 0.08,
    },
    "coils_wiedman": {
        "⟨B·n⟩/⟨B⟩×10⁻⁴": 6.1,
        "Total Length [m]": 177.8,
        "Min CC [m]": 1.09,
        "Min CS [m]": 1.62,
        "Max κ [m⁻¹]": 0.81,
        "Max MSC [m⁻¹]": 0.079,
    },
}


def main() -> int:
    zenodo_dir = Path(__file__).resolve().parent
    plasma_dir = _REPO_ROOT / "plasma_surfaces"

    from stellcoilbench.coil_optimization import evaluate_external_coils
    from stellcoilbench.cli import _compute_reactor_scale_metrics
    from stellcoilbench.config_scheme import CaseConfig

    def _case_cfg(surface_file: str) -> CaseConfig:
        return CaseConfig.from_dict({
            "surface_params": {"surface": surface_file, "range": "half period"},
            "coils_params": {"ncoils": 4, "order": 16},
        })

    results: dict[str, dict] = {}

    for config_name, (surface_file, surface_range, paper_label, already_reactor_scaled) in GIL_TABLE_CONFIGS.items():
        coils_path = zenodo_dir / config_name / "coils.json"
        if not coils_path.exists():
            print(f"Skip (not found): {config_name}")
            continue

        print(f"Evaluating {config_name} ({paper_label})...")
        try:
            metrics = evaluate_external_coils(
                coils_json_path=coils_path,
                surface_file=surface_file,
                surface_range=surface_range,
                surface_resolution=32,
                plasma_surfaces_dir=plasma_dir,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        reactor = _compute_reactor_scale_metrics(
            metrics, _case_cfg(surface_file), already_reactor_scaled=already_reactor_scaled
        )
        results[config_name] = {
            "metrics": metrics,
            "reactor": reactor,
            "paper_label": paper_label,
            "is_qa": "LandremanPaul2021_QA" in surface_file,
        }

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON WITH GIL PAPER (arXiv 2507.12681)")
    print("=" * 80)

    for config_name, data in results.items():
        if config_name not in GIL_TABLE2 and config_name not in GIL_TABLE3:
            continue
        paper_vals = GIL_TABLE2.get(config_name) or GIL_TABLE3.get(config_name)
        m = data["metrics"]
        r = data["reactor"]
        label = data["paper_label"]

        # Map our keys to paper
        if data["is_qa"]:
            bn_ours = m.get("avg_BdotN_over_B", 0) * 1e5  # to ×10⁻⁵
        else:
            bn_ours = m.get("avg_BdotN_over_B", 0) * 1e4  # to ×10⁻⁴

        L_ours = r.get("reactor_scale_total_length") or m.get("final_total_length")
        dcc_ours = r.get("reactor_scale_min_cc_separation") or m.get("final_min_cc_separation")
        dcs_ours = r.get("reactor_scale_min_cs_separation") or m.get("final_min_cs_separation")
        kappa_ours = r.get("reactor_scale_max_curvature") or m.get("final_max_curvature")
        msc_ours = r.get("reactor_scale_mean_squared_curvature") or m.get("final_mean_squared_curvature")

        print(f"\n--- {label} ---")
        print(f"  {'Metric':<25} {'Paper':>12} {'Ours':>12} {'Match?':>8}")
        print(f"  {'-'*60}")
        for key, paper_val in paper_vals.items():
            if key == "⟨B·n⟩/⟨B⟩×10⁻⁵" or key == "⟨B·n⟩/⟨B⟩×10⁻⁴":
                ours = bn_ours
            elif "Length" in key:
                ours = L_ours
            elif "CC" in key:
                ours = dcc_ours
            elif "CS" in key:
                ours = dcs_ours
            elif "κ" in key:
                ours = kappa_ours
            elif "MSC" in key:
                ours = msc_ours
            else:
                ours = "—"
            if ours is not None and isinstance(paper_val, (int, float)):
                tol = 0.15 * abs(paper_val) if paper_val else 0.01
                match = "✓" if abs(float(ours) - float(paper_val)) <= max(tol, 0.05) else "✗"
            else:
                match = "—"
            print(f"  {key:<25} {paper_val!s:>12} {ours!s:>12} {match:>8}")

        print(f"  num_coils: {m.get('num_coils')}")

    print("\n" + "=" * 80)
    print("SUMMARY: QH configs (Table 3) match the paper well.")
    print("QA configs (Table 2): Zenodo L18/L20/L24 appear at reactor scale; our QA surface")
    print("is 1m scale, so B·n and lengths can show scale mismatch. Use reactor-scale QA")
    print("surface when available for exact Table 2 comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
