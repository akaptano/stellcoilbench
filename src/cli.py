from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import evaluate

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


@app.command("make-leaderboard")
def make_leaderboard(
    results_dir: Path = typer.Argument(
        ...,
        help="Directory containing multiple submission 'results.json' files (recursively).",
    ),
    out_json: Path = typer.Option(
        Path("leaderboard.json"),
        "--out-json",
        help="Where to write the leaderboard JSON.",
    ),
    out_md: Optional[Path] = typer.Option(
        Path("leaderboard.md"),
        "--out-md",
        help="Where to write a simple markdown leaderboard table.",
    ),
) -> None:
    """
    Merge multiple submission result files into a simple leaderboard.
    """
    submissions = []
    for path in results_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Skipping {path}: could not parse JSON ({exc})", err=True)
            continue
        submissions.append((path, data))

    leaderboard = evaluate.build_leaderboard(submissions)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(leaderboard, indent=2))

    if out_md is not None:
        lines = ["# CoilBench Leaderboard", ""]
        if not leaderboard["entries"]:
            lines.append("_No valid submissions found._")
        else:
            lines.append("| Rank | Method | Version | Mean primary score | Num cases |")
            lines.append("|------|--------|---------|--------------------|-----------|")
            for entry in leaderboard["entries"]:
                lines.append(
                    f"| {entry['rank']} | {entry['method_name']} | {entry['method_version']} | "
                    f"{entry['mean_score_primary']:.3f} | {entry['num_cases']} |"
                )
        out_md.write_text("\n".join(lines))

    typer.echo(f"Wrote leaderboard JSON to {out_json}")
    if out_md is not None:
        typer.echo(f"Wrote markdown leaderboard to {out_md}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

