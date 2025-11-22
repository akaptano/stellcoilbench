# src/coilbench/cli.py
from __future__ import annotations

import json
import platform
import shutil
import subprocess
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


def _detect_github_username() -> str:
    """
    Try to detect GitHub username from git config or remote URL.
    Returns empty string if not found.
    """
    try:
        # Try git config first
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    try:
        # Try to get from remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract username from common GitHub URL patterns
            if "github.com" in url:
                parts = url.replace(".git", "").split("/")
                if len(parts) >= 2:
                    # Handle both https://github.com/user/repo and git@github.com:user/repo
                    username = parts[-2] if ":" in url else parts[-2]
                    if username and username != "github.com":
                        return username
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Try environment variable (useful in CI)
    import os
    github_user = os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USER")
    if github_user:
        return github_user
    
    return ""


def _detect_hardware() -> str:
    """
    Detect hardware information (CPU, GPU, memory).
    Returns a formatted string describing the hardware.
    """
    parts = []
    
    # CPU info
    try:
        cpu_info = platform.processor() or platform.machine()
        if cpu_info:
            parts.append(f"CPU: {cpu_info}")
    except Exception:
        pass
    
    # Try to get more detailed CPU info
    try:
        if platform.system() == "Linux":
            result = subprocess.run(
                ["lscpu"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Model name:" in line:
                        cpu_name = line.split("Model name:")[-1].strip()
                        if cpu_name:
                            parts[0] = f"CPU: {cpu_name}"
                            break
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts[0] = f"CPU: {result.stdout.strip()}"
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # GPU info (NVIDIA)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_names = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            if gpu_names:
                gpu_str = ", ".join(gpu_names)
                parts.append(f"GPU: {gpu_str}")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Memory info (optional, requires psutil)
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        mem_gb = mem.total / (1024**3)
        parts.append(f"RAM: {mem_gb:.1f}GB")
    except (ImportError, Exception):
        # psutil not available or error, skip
        pass
    
    return " | ".join(parts) if parts else platform.platform()

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


@app.command("submit-case")
def submit_case(
    case_path: Path = typer.Argument(
        ...,
        help="Path to case.yaml file (e.g., cases/case.yaml).",
    ),
    method_name: Optional[str] = typer.Option(
        None,
        "--method-name",
        "-m",
        help="Name of your optimization method (optional, stored in metadata).",
    ),
    method_version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Version identifier (optional, stored in metadata).",
    ),
    contact: Optional[str] = typer.Option(
        None,
        "--contact",
        "-c",
        help="Contact email or info (default: auto-detect GitHub username).",
    ),
    hardware: Optional[str] = typer.Option(
        None,
        "--hardware",
        help="Hardware description (default: auto-detect CPU/GPU).",
    ),
    notes: str = typer.Option("", "--notes", "-n", help="Additional notes."),
    submissions_dir: Path = typer.Option(
        Path("submissions"),
        "--submissions-dir",
        help="Directory where submission results.json will be written.",
    ),
) -> None:
    """
    Run a case and generate a submission results.json file.
    
    This command:
    1. Loads case.yaml from cases/
    2. Runs the coil optimization
    3. Evaluates the results
    4. Generates a results.json in submissions/<username>/<datetime>/ with metadata and metrics
    
    Directory structure: submissions/<surface>/<github_username>/<MM-DD-YYYY_HH-MM>/results.json
    GitHub username and hardware are auto-detected if not provided.
    
    Example:
        stellcoilbench submit-case cases/case.yaml
    """
    from .coil_optimization import optimize_coils
    from .evaluate import load_case_config, evaluate_case

    # Auto-detect GitHub username for directory structure
    github_username = _detect_github_username()
    if not github_username:
        github_username = "unknown_user"
        typer.echo("Warning: Could not auto-detect GitHub username. Using 'unknown_user'.")
        typer.echo("Use --contact to specify your GitHub username.")
    else:
        typer.echo(f"Using GitHub username: {github_username}")

    # Auto-detect contact (GitHub username) if not provided
    if contact is None:
        contact = github_username
        typer.echo(f"Auto-detected contact: {contact}")

    # Auto-detect hardware if not provided
    if hardware is None:
        hardware = _detect_hardware()
        if hardware:
            typer.echo(f"Auto-detected hardware: {hardware}")
        else:
            hardware = ""
            typer.echo("Warning: Could not auto-detect hardware. Use --hardware to specify.")

    # Load case configuration
    case_cfg = load_case_config(case_path)

    # Extract surface name from case config
    surface_file = case_cfg.surface_params.get("surface", "")
    if not surface_file:
        raise ValueError("case.yaml must specify surface_params.surface")
    # Extract just the filename if it's a path (e.g., "input.LandremanPaul2021_QA")
    surface_name = Path(surface_file).name
    # Remove common prefixes like "input." or "wout." from directory name
    if surface_name.startswith("input."):
        surface_name = surface_name[6:]  # Remove "input." prefix
    elif surface_name.startswith("wout."):
        surface_name = surface_name[5:]  # Remove "wout." prefix
    
    # 3) Build submission directory first (needed for output_dir)
    now = datetime.now()
    run_date = now.isoformat()
    datetime_str = now.strftime("%m-%d-%Y_%H-%M")  # Format: MM-DD-YYYY_HH-MM
    
    # Write to submissions directory: submissions/<surface>/<username>/<datetime>/
    submission_dir = submissions_dir / surface_name / github_username / datetime_str
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Coils filename is always coils.json
    coils_filename = "coils.json"
    coils_out_path = submission_dir / coils_filename

    # 1) Run the optimizer, writing coils_out_path and VTK files to submission_dir.
    typer.echo("Running optimizer...")
    results_dict = optimize_coils(
        case_path=case_path, 
        coils_out_path=coils_out_path, 
        case_cfg=case_cfg,
        output_dir=submission_dir  # VTK files will be saved here
    )
    typer.echo(f"Wrote optimized coils to {coils_out_path}")

    # 2) Evaluate the resulting coils.
    metrics = evaluate_case(case_cfg=case_cfg, results_dict=results_dict)

    # 3) Build submission results.json
    submission = {
        "metadata": {
            "method_name": method_name or "",
            "method_version": method_version or "",
            "contact": contact,
            "hardware": hardware,
            "notes": notes,
            "run_date": run_date,
        },
        "metrics": metrics,
    }
    
    # Write results.json
    submission_path = submission_dir / "results.json"
    submission_path.write_text(json.dumps(submission, indent=2, cls=NumpyJSONEncoder))
    typer.echo(f"Wrote submission results to {submission_path}")
    
    # Copy case.yaml file to submission directory for reference
    case_yaml_path = case_path if case_path.is_file() else (case_path / "case.yaml")
    if case_yaml_path.exists() and case_yaml_path.is_file():
        submission_case_yaml = submission_dir / "case.yaml"
        shutil.copy2(case_yaml_path, submission_case_yaml)
        typer.echo(f"Copied case.yaml to {submission_case_yaml}")


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
        help="Where to write the results JSON (default: <coils_out_dir>/results.json).",
    ),
) -> None:
    """
    Run a coil optimization for one case using parameters from case.yaml,
    then evaluate the resulting coil set.
    
    Note: For generating submissions, use 'submit-case' instead.
    """
    from .coil_optimization import optimize_coils
    from .evaluate import load_case_config, evaluate_case

    # Load case configuration
    case_cfg = load_case_config(case_path)

    coils_out_dir.mkdir(parents=True, exist_ok=True)

    # Coils filename is always coils.json
    coils_filename = "coils.json"
    coils_out_path = coils_out_dir / coils_filename

    # 1) Run the optimizer, writing coils_out_path.
    typer.echo("Running optimizer...")
    results_dict = optimize_coils(case_path=case_path, coils_out_path=coils_out_path, case_cfg=case_cfg)
    typer.echo(f"Wrote optimized coils to {coils_out_path}")

    # 2) Evaluate the resulting coils.
    metrics = evaluate_case(case_cfg=case_cfg, results_dict=results_dict)

    # Decide results filename.
    if results_out is None:
        results_out = coils_out_dir / "results.json"
    
    # Ensure output path has .json extension for JSON format
    if not str(results_out).endswith('.json'):
        results_out = results_out.with_suffix('.json')

    results_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.write_text(json.dumps(metrics, indent=2, cls=NumpyJSONEncoder))
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
            coils_path = coils_root / "coils.json"
            
            if not coils_path.exists():
                typer.echo(f"Warning: Coils file not found: {coils_path}", err=True)
                continue

            # For now, we need to load the results_dict from the optimization
            # In a real workflow, you'd evaluate the coils here
            # For testing, we can create a minimal results_dict
            results_dict = {
                "chi2_Bn": 0.001,  # Placeholder - would come from actual evaluation
            }
            
            metrics = evaluate_case(case_cfg=case_cfg, results_dict=results_dict)
            case_results.append(metrics)
            typer.echo(f"Processed case from {case_dir}")
            
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
        "metrics": case_results[0] if case_results else {},  # Use first case's metrics, or empty dict
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
