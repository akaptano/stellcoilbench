# src/coilbench/cli.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import evaluate
from .update_db import update_database

app = typer.Typer(help="CoilBench: benchmarking framework for stellarator coil optimization.")


@app.command("eval")
def eval_case(
    case_dir: Path = typer.Argument(..., help="Path to a single case directory (containing case.yaml)."),
    coils: Path = typer.Argument(..., help="Path to a coils file produced by your method (e.g. .h5)."),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Where to write the per-case results JSON. If omitted, prints to stdout.",
    ),
) -> None:
    """
    Evaluate a single case + coil set and emit metrics/scores.
    """
    result = evaluate.evaluate_case(case_dir=case_dir, coils_path=coils)

    if out is None:
        typer.echo(json.dumps(result, indent=2))
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        typer.echo(f"Wrote results to {out}")


@app.command("eval-bundle")
def eval_bundle(
    cases_root: Path = typer.Argument(
        ...,
        help="Root directory containing multiple case directories (each with case.yaml).",
    ),
    coils_root: Path = typer.Argument(
        ...,
        help="Directory holding one coils file per case. "
        "By default we look for '<case_id>.h5' inside this directory.",
    ),
    out: Path = typer.Option(
        Path("submission_results.json"),
        "--out",
        "-o",
        help="Where to write the aggregated submission results JSON.",
    ),
) -> None:
    """
    Evaluate a bundle of coil files across all cases under cases_root.

    Assumes:
      - each case directory under cases_root contains a case.yaml
      - the coils_root directory contains '<case_id>.h5' for each case
    """
    submission = evaluate.evaluate_bundle(
        cases_root=cases_root,
        coils_root=coils_root,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(submission, indent=2))
    typer.echo(f"Wrote submission results to {out}")


@app.command("update-db")
def update_db_cmd(
    submissions_dir: Path = typer.Argument(
        Path("submissions"),
        help="Directory containing per-method submissions (results.json files).",
    ),
    db_dir: Path = typer.Option(
        Path("db"),
        "--db-dir",
        help="Directory where database JSON files (methods.json, cases.json, leaderboard.json) are stored.",
    ),
    docs_dir: Path = typer.Option(
        Path("docs"),
        "--docs-dir",
        help="Directory where docs/leaderboard.md is written.",
    ),
) -> None:
    """
    Rebuild the on-repo 'database' of submissions and leaderboards.

    This scans submissions_dir for results.json files produced by `coilbench eval-bundle`,
    aggregates them into db/*.json, and writes docs/leaderboard.md.
    """
    repo_root = Path.cwd()
    update_database(
        repo_root=repo_root,
        submissions_root=submissions_dir,
        db_dir=db_dir,
        docs_dir=docs_dir,
    )
    typer.echo(f"Updated database in {db_dir} and leaderboard in {docs_dir / 'leaderboard.md'}")


@app.command("run-case")
def run_case(
    case_dir: Path = typer.Argument(
        ...,
        help="Path to a single case directory (containing case.yaml, geometry, etc.).",
    ),
    coils_config_path: Path = typer.Argument(
        ...,
        help="Path to coils.yaml (or similar) that configures the optimizer.",
    ),
    coils_out_dir: Path = typer.Option(
        Path("coils_runs"),
        "--coils-out-dir",
        help="Directory where the optimized coils file will be written.",
    ),
    results_out: Optional[Path] = typer.Option(
        None,
        "--results-out",
        "-o",
        help="Where to write the per-case results JSON (default: <coils_out_dir>/<case_id>_results.json).",
    ),
) -> None:
    """
    Run a coil optimization for one case using parameters from coils.yaml,
    then evaluate the resulting coil set.
    """
    from .coil_optimization import load_coils_config, optimize_coils
    from .evaluate import load_case_config
    case_cfg = load_case_config(case_dir)
    coils_config = load_coils_config(coils_config_path)

    coils_out_dir.mkdir(parents=True, exist_ok=True)

    # Decide coils filename, using pattern from config if provided.
    pattern = (
        coils_config.get("output", {})
        .get("coils_filename_pattern", "{case_id}.h5")
    )
    coils_filename = pattern.format(case_id=case_cfg.case_id)
    coils_out_path = coils_out_dir / coils_filename

    # 1) Run the optimizer, writing coils_out_path.
    typer.echo(f"Running optimizer '{coils_config.get('optimizer', 'dummy')}' for case {case_cfg.case_id}...")
    optimize_coils(case_dir=case_dir, coils_config=coils_config, coils_out_path=coils_out_path)
    typer.echo(f"Wrote optimized coils to {coils_out_path}")

    # 2) Evaluate the resulting coils.
    result = evaluate.evaluate_case(case_dir=case_dir, coils_path=coils_out_path)

    # Decide results filename.
    if results_out is None:
        results_out = coils_out_dir / f"{case_cfg.case_id}_results.json"

    results_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.write_text(json.dumps(result, indent=2))
    typer.echo(f"Wrote evaluation results to {results_out}")

def main() -> None:
    app()


if __name__ == "__main__":
    main()
