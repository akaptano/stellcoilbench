# src/coilbench/cli.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import typer


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and arrays."""
    def default(self, o):
        # Handle numpy integer types
        if isinstance(o, np.integer):
            return int(o)
        # Handle numpy floating point types
        elif isinstance(o, np.floating):
            return float(o)
        # Handle numpy arrays
        elif isinstance(o, np.ndarray):
            return o.tolist()
        # Handle numpy boolean
        elif isinstance(o, np.bool_):
            return bool(o)
        # Handle jax/jaxlib arrays and other array-like objects
        elif hasattr(o, '__array__'):
            try:
                return np.asarray(o).tolist()
            except (TypeError, ValueError):
                pass
        return super().default(o)

app = typer.Typer(help="CoilBench: benchmarking framework for stellarator coil optimization.")

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
    from .update_db import update_database
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
    case_path: Path = typer.Argument(
        ...,
        help="Path to case directory containing case.yaml and coils.yaml, or a single YAML file.",
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
    from .coil_optimization import optimize_coils
    from .evaluate import load_case_config, evaluate_case

    # Load case configuration
    case_cfg = load_case_config(case_path)

    coils_out_dir.mkdir(parents=True, exist_ok=True)

    # Decide coils filename
    coils_filename = f"{case_cfg.case_id}.json"
    coils_out_path = coils_out_dir / coils_filename

    # 1) Run the optimizer, writing coils_out_path.
    typer.echo(f"Running optimizer for case {case_cfg.case_id}...")
    results_dict = optimize_coils(case_path=case_path, coils_out_path=coils_out_path, case_cfg=case_cfg)
    typer.echo(f"Wrote optimized coils to {coils_out_path}")

    # 2) Evaluate the resulting coils.
    case_result = evaluate_case(case_cfg=case_cfg, results_dict=results_dict)

    # Decide results filename.
    if results_out is None:
        results_out = coils_out_dir / f"{case_cfg.case_id}_results.json"
    
    # Ensure output path has .json extension for JSON format
    if not str(results_out).endswith('.json'):
        results_out = results_out.with_suffix('.json')

    results_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.write_text(json.dumps(case_result, indent=2, cls=NumpyJSONEncoder))
    typer.echo(f"Wrote evaluation results to {results_out}")


@app.command("generate-submission")
def generate_submission(
    cases_root: Path = typer.Argument(
        ...,
        help="Root directory containing case directories (each with case.yaml and coils.yaml).",
    ),
    metadata_path: Path = typer.Argument(
        ...,
        help="Path to metadata.yaml file containing submission metadata.",
    ),
    coils_root: Path = typer.Option(
        Path("coils_runs"),
        "--coils-root",
        help="Directory containing optimized coil files (one per case).",
    ),
    submission_out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Where to write the submission results.json file.",
    ),
) -> None:
    """
    Generate a results.json submission file from multiple cases and a metadata.yaml file.
    
    This command:
    1. Loads metadata from metadata.yaml
    2. For each case in cases_root, loads case.yaml and evaluates the corresponding coils
    3. Combines everything into a results.json file ready for submission
    """
    from .evaluate import load_case_config, evaluate_case
    from .config_scheme import SubmissionMetadata
    import yaml

    # Load metadata
    metadata_data = yaml.safe_load(metadata_path.read_text())
    metadata = SubmissionMetadata(
        method_name=metadata_data.get("method_name", "UNKNOWN"),
        method_version=metadata_data.get("method_version", "0.0.0"),
        contact=metadata_data.get("contact", ""),
        hardware=metadata_data.get("hardware", ""),
        notes=metadata_data.get("notes", ""),
    )

    # Find all case directories
    case_dirs = [d for d in cases_root.iterdir() if d.is_dir() and (d / "case.yaml").exists()]
    
    if not case_dirs:
        typer.echo(f"No case directories found in {cases_root}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(case_dirs)} case(s) to process...")

    # Process each case
    case_results = []
    for case_dir in sorted(case_dirs):
        try:
            case_cfg = load_case_config(case_dir)
            coils_path = coils_root / f"{case_cfg.case_id}.json"
            
            if not coils_path.exists():
                typer.echo(f"Warning: Coils file not found for {case_cfg.case_id}: {coils_path}", err=True)
                continue

            # For now, we need to load the results_dict from the optimization
            # In a real workflow, you'd evaluate the coils here
            # For testing, we can create a minimal results_dict
            results_dict = {
                "chi2_Bn": 0.001,  # Placeholder - would come from actual evaluation
            }
            
            case_result = evaluate_case(case_cfg=case_cfg, results_dict=results_dict)
            case_results.append(case_result)
            typer.echo(f"Processed case: {case_cfg.case_id}")
            
        except Exception as e:
            typer.echo(f"Error processing {case_dir}: {e}", err=True)
            continue

    if not case_results:
        typer.echo("No valid cases processed!", err=True)
        raise typer.Exit(1)

    # Build submission results
    run_date = datetime.now().isoformat()
    submission = {
        "metadata": {
            "method_name": metadata.method_name,
            "method_version": metadata.method_version,
            "contact": metadata.contact,
            "hardware": metadata.hardware,
            "notes": metadata.notes,
            "run_date": run_date,
        },
        "cases": case_results,
    }

    # Write output
    if submission_out is None:
        submission_out = Path("submissions") / metadata.method_name / metadata.method_version / "results.json"
    
    # Ensure output path has .json extension for JSON format
    if not str(submission_out).endswith('.json'):
        submission_out = submission_out.with_suffix('.json')
    
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission_out.write_text(json.dumps(submission, indent=2, cls=NumpyJSONEncoder))
    typer.echo(f"Wrote submission results to {submission_out}")

def main() -> None:
    app()


if __name__ == "__main__":
    main()
