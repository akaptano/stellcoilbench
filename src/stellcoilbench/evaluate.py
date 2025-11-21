from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from .config_scheme import CaseConfig, SubmissionMetadata


@dataclass
class CaseResult:
    case_id: str
    metrics: Dict[str, float]


@dataclass
class SubmissionResults:
    metadata: SubmissionMetadata
    cases: List[CaseResult]


def load_case_config(case_dir: Path) -> CaseConfig:
    """
    Load a case.yaml file into a CaseConfig dataclass.
    
    Accepts either:
    - A directory path containing case.yaml
    - A direct path to case.yaml file
    """
    # If it's a file, use it directly
    if case_dir.is_file():
        cfg_path = case_dir
    # If it's a directory, look for case.yaml inside
    elif case_dir.is_dir():
        cfg_path = case_dir / "case.yaml"
    else:
        # Try treating it as a file path
        cfg_path = case_dir
    
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Expected case.yaml file at {case_dir} or {case_dir}/case.yaml")
    
    data = yaml.safe_load(cfg_path.read_text())
    return CaseConfig.from_dict(data)


def evaluate_case(case_cfg: CaseConfig, results_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single case + coils.

    This function orchestrates:
      - computing B·n on the plasma surface for this case
      - computing normal-field error and coil complexity metrics
      - combining those into one or more scores

    Parameters
    ----------
    case_cfg:
        CaseConfig object containing the case configuration.
    results_dict:
        Dictionary containing the results of the optimization.

    Returns
    -------
    Dictionary containing the evaluation results.
    The dictionary contains the case ID and the metrics.
    """
    case_result = CaseResult(
        case_id=case_cfg.case_id,
        metrics=results_dict,
    )
    return {
        "case_id": case_result.case_id,
        "metrics": case_result.metrics,
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

