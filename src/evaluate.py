from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from .config_schema import CaseConfig, SubmissionMetadata
from . import biotsavart, geometry, metrics


@dataclass
class CaseResult:
    case_id: str
    metrics: Dict[str, float]
    scores: Dict[str, float]


@dataclass
class SubmissionResults:
    metadata: SubmissionMetadata
    cases: List[CaseResult]


def load_case_config(case_dir: Path) -> CaseConfig:
    """
    Load a case.yaml file into a CaseConfig dataclass.
    """
    cfg_path = case_dir / "case.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Expected case.yaml in {case_dir}")
    data = yaml.safe_load(cfg_path.read_text())
    return CaseConfig.from_dict(data)


def evaluate_case(case_dir: Path, coils_path: Path) -> Dict[str, Any]:
    """
    Evaluate a single case + coils.

    This function orchestrates:
      - computing B·n on the plasma surface for this case
      - computing normal-field error and coil complexity metrics
      - combining those into one or more scores

    Heavy physics-specific pieces live in biotsavart.py and geometry.py.
    """
    case_cfg = load_case_config(case_dir)

    # Compute B·n on the plasma surface for this case.
    Bn = biotsavart.compute_Bn_on_plasma_surface(case_dir=case_dir, coils_path=coils_path)

    # Normal-field error metric.
    chi2_Bn = metrics.normal_field_error(Bn, B0=case_cfg.normalization_B0)

    # Coil complexity / engineering metrics (curvature, length, spacing, ...).
    complexity = geometry.coil_complexity_metrics(case_dir=case_dir, coils_path=coils_path)

    metric_dict: Dict[str, float] = {"chi2_Bn": float(chi2_Bn), **complexity}

    # Composite scores (e.g. normalized 0–1 primary score + feasibility flag).
    score_dict = metrics.composite_scores(metric_dict, case_cfg)

    case_result = CaseResult(
        case_id=case_cfg.case_id,
        metrics=metric_dict,
        scores=score_dict,
    )
    return {
        "case_id": case_result.case_id,
        "metrics": case_result.metrics,
        "scores": case_result.scores,
    }


def _iter_case_dirs(cases_root: Path) -> Iterable[Path]:
    """
    Yield all immediate child directories of cases_root that contain a case.yaml.
    """
    for path in sorted(cases_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "case.yaml").is_file():
            yield path


def evaluate_bundle(
    cases_root: Path,
    coils_root: Path,
    metadata: SubmissionMetadata | None = None,
) -> Dict[str, Any]:
    """
    Evaluate a bundle of coil files for all cases under cases_root.

    Assumes each case directory under cases_root has a `case.yaml`, and that
    coils_root contains `<case_id>.h5` (or whatever extension you use) for each
    such case.

    The mapping from case_id -> coil filename is defined here; for now we use a
    simple `<case_id>.h5` convention that you can adapt.
    """
    cases: List[CaseResult] = []

    for case_dir in _iter_case_dirs(cases_root):
        case_cfg = load_case_config(case_dir)
        coils_path = coils_root / f"{case_cfg.case_id}.h5"
        if not coils_path.is_file():
            raise FileNotFoundError(
                f"Expected coils file at {coils_path} for case_id={case_cfg.case_id}. "
                "You can change this convention in evaluate_bundle."
            )

        result_dict = evaluate_case(case_dir, coils_path)
        cases.append(
            CaseResult(
                case_id=result_dict["case_id"],
                metrics=result_dict["metrics"],
                scores=result_dict["scores"],
            )
        )

    if metadata is None:
        # Minimal metadata stub; in practice you will construct this from a YAML file.
        metadata = SubmissionMetadata(
            method_name="UNKNOWN_METHOD",
            method_version="0.0.0",
            contact="unknown@example.com",
            hardware="unknown",
            notes="",
        )

    submission = SubmissionResults(metadata=metadata, cases=cases)
    return {
        "metadata": asdict(submission.metadata),
        "cases": [asdict(c) for c in submission.cases],
    }


def build_leaderboard(
    submissions: Iterable[Tuple[Path, Dict[str, Any]]],
    primary_score_key: str = "score_primary",
) -> Dict[str, Any]:
    """
    Build a simple leaderboard from multiple submission result dicts.

    Parameters
    ----------
    submissions:
        Iterable of (path, data-dict) pairs, as loaded from results.json.
    primary_score_key:
        Name of the score inside each case's `scores` dict to use as the
        primary scalar objective.

    Returns
    -------
    dict with keys:
        - entries: list of leaderboard rows (sorted best-to-worst)
    """
    entries = []

    for path, data in submissions:
        meta = data.get("metadata") or {}
        cases = data.get("cases") or []
        if not cases:
            continue

        scores: List[float] = []
        for case in cases:
            s = case.get("scores", {}).get(primary_score_key, None)
            if isinstance(s, (int, float)):
                scores.append(float(s))
        if not scores:
            continue

        entries.append(
            {
                "method_name": meta.get("method_name", "UNKNOWN"),
                "method_version": meta.get("method_version", "UNKNOWN"),
                "source": str(path),
                "mean_score_primary": float(mean(scores)),
                "num_cases": len(scores),
            }
        )

    # Sort descending by mean primary score.
    entries.sort(key=lambda e: e["mean_score_primary"], reverse=True)

    # Assign ranks (1-based).
    for i, entry in enumerate(entries, start=1):
        entry["rank"] = i

    return {"entries": entries}

