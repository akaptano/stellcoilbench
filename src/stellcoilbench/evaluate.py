"""
Evaluation and leaderboard utilities for StellCoilBench.

This module provides functions to load case configurations, evaluate optimization
results against scoring criteria, and build leaderboards from multiple submissions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from .config_scheme import CaseConfig, SubmissionMetadata


@dataclass
class SubmissionResults:
    """
    Aggregated results for a single submission.

    Attributes
    ----------
    metadata : SubmissionMetadata
        Method name, version, contact, hardware.
    metrics : dict[str, Any]
        Evaluation metrics (scores, coil metrics, reactor-scale quantities).
    """

    metadata: SubmissionMetadata
    metrics: Dict[str, Any]


def load_case_config(case_dir: Path) -> CaseConfig:
    """
    Load a case.yaml file into a CaseConfig dataclass.

    Accepts either a directory path containing case.yaml or a direct path to
    the case.yaml file. Validates the configuration before returning.

    Parameters
    ----------
    case_dir : Path
        Directory containing case.yaml, or path to case.yaml itself.

    Returns
    -------
    CaseConfig
        Parsed and validated case configuration.

    Raises
    ------
    FileNotFoundError
        If case.yaml is not found at the expected location.
    ValueError
        If validation fails (see validate_case_config).
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
    
    # Validate the configuration
    from .validate_config import validate_case_config
    errors = validate_case_config(data, cfg_path)
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    return CaseConfig.from_dict(data)


def evaluate_case(
    case_cfg: CaseConfig,
    results_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate a single case + coils.

    This function orchestrates computing B·n on the plasma surface, normal-field
    error, coil complexity metrics, and combining them into scores. Currently
    returns the results_dict as-is; full evaluation is performed in the
    optimization pipeline.

    Parameters
    ----------
    case_cfg : CaseConfig
        Case configuration (surface, coils).
    results_dict : dict[str, Any]
        Optimization results (coils, metrics, scores).

    Returns
    -------
    dict[str, Any]
        Evaluation metrics (scores, coil metrics, reactor-scale quantities).
    """
    return results_dict


def build_leaderboard(
    submissions: Iterable[Tuple[Path, Dict[str, Any]]],
    primary_score_key: str = "score_primary",
) -> Dict[str, Any]:
    """
    Build a simple leaderboard from multiple submission result dicts.

    Parameters
    ----------
    submissions : Iterable[tuple[Path, dict[str, Any]]]
        Iterable of (path, data-dict) pairs, as loaded from results.json.
    primary_score_key : str, default="score_primary"
        Name of the score inside each case's ``scores`` dict to use as the
        primary scalar objective.

    Returns
    -------
    dict[str, Any]
        Leaderboard with keys:
        - entries : list[dict]
            Leaderboard rows sorted best-to-worst (descending by mean primary score).
            Each row has contact, method_version, source, mean_score_primary,
            num_cases, rank.
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
                "contact": meta.get("contact", "UNKNOWN"),
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

