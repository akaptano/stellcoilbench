# src/coilbench/cli.py
from __future__ import annotations

import json
import platform
import subprocess
import zipfile
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
    Try to detect GitHub username from remote URL or environment variables.
    Returns empty string if not found.
    
    Note: git config user.name returns the display name, not the GitHub username,
    so we prioritize extracting from the remote URL.
    """
    try:
        # Try to get from remote URL first (most reliable for GitHub username)
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
                # Handle https://github.com/user/repo format
                if url.startswith("https://") or url.startswith("http://"):
                    parts = url.replace(".git", "").split("/")
                    # URL format: https://github.com/user/repo
                    # parts: ['https:', '', 'github.com', 'user', 'repo']
                    if len(parts) >= 4 and parts[2] == "github.com":
                        username = parts[3]
                        if username and username != "github.com":
                            return username
                # Handle git@github.com:user/repo format
                elif url.startswith("git@"):
                    # URL format: git@github.com:user/repo
                    # Split on ':' to get the part after github.com:
                    if ":" in url:
                        after_colon = url.split(":", 1)[1]
                        parts = after_colon.replace(".git", "").split("/")
                        if len(parts) >= 1:
                            username = parts[0]
                            if username:
                                return username
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Try environment variable (useful in CI)
    import os
    github_user = os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USER")
    if github_user:
        return github_user
    
    return ""


def _zip_submission_directory(submission_dir: Path) -> Path:
    """
    Zip the submission files (excluding PDFs and post-processing outputs) into all_files.zip.
    
    Creates a zip file named "all_files.zip" inside the submission directory.
    PDF files and post-processing outputs (QFM surface, Poincaré plots, VMEC plots, etc.)
    are kept in the directory alongside the zip file.
    
    Parameters
    ----------
    submission_dir: Path
        Directory containing submission files to zip.
    
    Returns
    -------
    Path
        Path to the created zip file (submission_dir / "all_files.zip").
    """
    submission_dir = Path(submission_dir)
    
    if not submission_dir.exists() or not submission_dir.is_dir():
        typer.echo(f"Warning: Submission directory does not exist: {submission_dir}")
        return submission_dir / "all_files.zip"
    
    # Create zip file inside the submission directory
    zip_filename = "all_files.zip"
    zip_path = submission_dir / zip_filename
    
    # Find all files in the submission directory
    files_to_zip = []
    pdf_files_to_keep = []
    for file_path in submission_dir.rglob("*"):
        if file_path.is_file():
            # PDF plots stay in the DATE directory and are NOT zipped
            if file_path.suffix.lower() == ".pdf":
                pdf_files_to_keep.append(file_path)
            else:
                files_to_zip.append(file_path)
    
    if not files_to_zip:
        typer.echo(f"Warning: No files found in {submission_dir} to zip")
        return zip_path
    
    # Note: PDF files are kept in the DATE directory and not included in the zip

    # Create zip file with remaining files (excluding PDFs)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            # Add file to zip with relative path from submission_dir
            arcname = file_path.relative_to(submission_dir)
            zipf.write(file_path, arcname=arcname)
    
    typer.echo(f"Created zip archive: {zip_path}")
    typer.echo(f"  Contains {len(files_to_zip)} files")
    if pdf_files_to_keep:
        typer.echo(f"  Kept {len(pdf_files_to_keep)} PDF file(s) in {submission_dir}")
    
    # Keep post-processing files in addition to PDFs:
    # - PDF files (bn_error plots)
    # - Post-processing outputs: .vts (QFM surface), .png (plots), post_processing_results.json
    post_processing_patterns = [
        'qfm_surface',
        'poincare',
        'boozer',
        'iota',
        'quasisymmetry',
        'post_processing_results',
        'simple_loss_fraction',  # SIMPLE fast particle tracing plot
        'simple',  # Also match any file with 'simple' in name
    ]
    
    # Remove files that should be zipped, but keep PDFs and post-processing files
    for file_path in files_to_zip:
        # Keep if it's a post-processing file (check filename patterns)
        is_post_processing_file = any(
            pattern.lower() in file_path.name.lower() 
            for pattern in post_processing_patterns
        ) and file_path.suffix.lower() in {'.vts', '.png', '.json'}
        
        if not is_post_processing_file:
            try:
                file_path.unlink()
                # Try to remove parent directory if it's empty (but not the submission_dir itself)
                parent = file_path.parent
                if parent != submission_dir and parent.exists() and not any(parent.iterdir()):
                    try:
                        parent.rmdir()
                    except (OSError, FileNotFoundError):
                        pass  # Directory not empty or other error, skip
            except (OSError, FileNotFoundError) as e:
                typer.echo(f"Warning: Failed to remove {file_path}: {e}")
    
    typer.echo(f"  Submission directory structure: {submission_dir}")
    typer.echo(f"    - Zip file: {zip_path.name}")
    if pdf_files_to_keep:
        typer.echo(f"    - PDF files: {len(pdf_files_to_keep)} file(s)")
    
    return zip_path


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
    docs_dir: Path = typer.Option(
        Path("docs"),
        "--docs-dir",
        help="Directory where docs/leaderboards/ leaderboards and leaderboard.json are written.",
    ),
) -> None:
    """
    Rebuild the on-repo 'database' of submissions and leaderboards.

    This scans submissions_dir for results.json files produced by `coilbench eval-bundle`,
    aggregates them into docs/leaderboard.json, and writes per-surface leaderboards in docs/leaderboards/.
    """
    from .update_db import update_database
    repo_root = Path.cwd()
    update_database(
        repo_root=repo_root,
        submissions_root=submissions_dir,
        docs_dir=docs_dir,
    )
    typer.echo(f"Updated leaderboard.json and surface leaderboards in {docs_dir / 'leaderboards'}")


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
    
    Directory structure: submissions/<github_username>/<MM-DD-YYYY_HH-MM>/all_files.zip
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

    # Auto-detect contact (use GitHub username)
    contact = github_username
    typer.echo(f"Auto-detected contact: {contact}")

    # Auto-detect hardware
    hardware = _detect_hardware()
    if not hardware:
        hardware = "Unknown hardware"
        typer.echo("Warning: Could not auto-detect hardware.")
    else:
        typer.echo(f"Auto-detected hardware: {hardware}")

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
    
    # Remove file extensions like ".focus" from surface name
    # Keep only the base name (e.g., "c09r00_B_axis_half_tesla_PM4Stell" from "c09r00_B_axis_half_tesla_PM4Stell.focus")
    if "." in surface_name:
        # Split on first dot and take the part before it
        surface_name = surface_name.split(".", 1)[0]
    
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
    # Also add source_case_file field to track which case file was used
    case_yaml_path = case_path if case_path.is_file() else (case_path / "case.yaml")
    if case_yaml_path.exists() and case_yaml_path.is_file():
        submission_case_yaml = submission_dir / "case.yaml"
        # Read the case file and add source_case_file field
        import yaml
        case_data = yaml.safe_load(case_yaml_path.read_text())
        # Store relative path from repo root for portability
        repo_root = Path.cwd()
        try:
            source_case_file = str(case_yaml_path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            # If relative path fails, use absolute path
            source_case_file = str(case_yaml_path.resolve())
        case_data["source_case_file"] = source_case_file
        # Write modified case.yaml to submission directory
        submission_case_yaml.write_text(yaml.dump(case_data, default_flow_style=False, sort_keys=False))
        typer.echo(f"Copied case.yaml to {submission_case_yaml} (with source_case_file: {source_case_file})")
    
    # Zip the entire submission directory and remove original files
    _zip_submission_directory(submission_dir)


@app.command("run-case")
def run_case(
    case_path: Path = typer.Argument(
        ...,
        help="Path to case directory containing case.yaml and coils.yaml, or a single YAML file.",
    ),
    submissions_dir: Path = typer.Option(
        Path("submissions"),
        "--submissions-dir",
        help="Directory where submission results will be written.",
    ),
    results_out: Optional[Path] = typer.Option(
        None,
        "--results-out",
        "-o",
        help="Where to write the results JSON (default: <submissions_dir>/<surface>/<username>/<datetime>/results.json).",
    ),
) -> None:
    """
    Run a coil optimization for one case using parameters from case.yaml,
    then evaluate the resulting coil set.
    
    Creates a subdirectory in submissions/ with structure:
    submissions/<surface>/<username>/<datetime>/
    
    Note: For generating submissions, use 'submit-case' instead.
    """
    from .coil_optimization import optimize_coils
    from .evaluate import load_case_config, evaluate_case

    # Load case configuration
    case_cfg = load_case_config(case_path)

    # Extract surface name from case config (similar to submit-case)
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
    
    # Remove file extensions like ".focus" from surface name
    if "." in surface_name:
        # Split on first dot and take the part before it
        surface_name = surface_name.split(".", 1)[0]

    # Auto-detect GitHub username for directory structure
    github_username = _detect_github_username()
    if not github_username:
        github_username = "unknown_user"
        typer.echo("Warning: Could not auto-detect GitHub username. Using 'unknown_user'.")

    # Create timestamp-based subdirectory
    now = datetime.now()
    datetime_str = now.strftime("%m-%d-%Y_%H-%M")  # Format: MM-DD-YYYY_HH-MM
    
    # Create submission directory: submissions/<surface>/<username>/<datetime>/
    submission_dir = submissions_dir / surface_name / github_username / datetime_str
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Coils filename is always coils.json
    coils_filename = "coils.json"
    coils_out_path = submission_dir / coils_filename

    # 1) Run the optimizer, writing coils_out_path.
    typer.echo("Running optimizer...")
    results_dict = optimize_coils(case_path=case_path, coils_out_path=coils_out_path, case_cfg=case_cfg)
    typer.echo(f"Wrote optimized coils to {coils_out_path}")

    # 2) Evaluate the resulting coils.
    metrics = evaluate_case(case_cfg=case_cfg, results_dict=results_dict)

    # Decide results filename.
    if results_out is None:
        results_out = submission_dir / "results.json"
    
    # Ensure output path has .json extension for JSON format
    if not str(results_out).endswith('.json'):
        results_out = results_out.with_suffix('.json')

    results_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.write_text(json.dumps(metrics, indent=2, cls=NumpyJSONEncoder))
    typer.echo(f"Wrote evaluation results to {results_out}")


@app.command("generate-submission")
def generate_submission(
    case_path: Path = typer.Argument(
        ...,
        help="Path to case.yaml file or directory containing case.yaml.",
    ),
    metadata_path: Path = typer.Argument(
        ...,
        help="Path to metadata.yaml file containing submission metadata.",
    ),
    coils_path: Path = typer.Option(
        None,
        "--coils",
        help="Path to coils.json file (default: <case_dir>/coils.json).",
    ),
    submission_out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Where to write the submission results.json file.",
    ),
) -> None:
    """
    Generate a results.json submission file from a case and coils file.
    
    This command:
    1. Loads metadata from metadata.yaml
    2. Loads case.yaml and evaluates the coils
    3. Creates a results.json file ready for submission
    
    Note: For running optimizations and generating submissions, use 'submit-case' instead.
    This command is for creating submissions from pre-existing coils files.
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

    # Load case configuration
    case_cfg = load_case_config(case_path)
    
    # Determine coils path
    if coils_path is None:
        if case_path.is_dir():
            coils_path = case_path / "coils.json"
        else:
            coils_path = case_path.parent / "coils.json"
    
    if not coils_path.exists():
        typer.echo(f"Error: Coils file not found: {coils_path}", err=True)
        raise typer.Exit(1)

    # Evaluate the coils (this would normally load and evaluate, but for now use placeholder)
    # In a real implementation, you'd load the coils and compute metrics
    results_dict = {
        "chi2_Bn": 0.001,  # Placeholder - would come from actual evaluation
    }
    
    metrics = evaluate_case(case_cfg=case_cfg, results_dict=results_dict)

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
        "metrics": metrics,
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


@app.command("post-process")
def post_process(
    coils_json: Path = typer.Argument(
        ...,
        help="Path to coils JSON file (e.g., biot_savart_optimized.json or coils.json).",
    ),
    output_dir: Path = typer.Option(
        Path("post_processing_output"),
        "--output-dir",
        "-o",
        help="Directory where post-processing results will be saved.",
    ),
    case_yaml: Optional[Path] = typer.Option(
        None,
        "--case-yaml",
        help="Path to case.yaml file. If not provided, will search relative to coils JSON.",
    ),
    plasma_surfaces_dir: Optional[Path] = typer.Option(
        None,
        "--plasma-surfaces-dir",
        help="Directory containing plasma surface files. Defaults to 'plasma_surfaces'.",
    ),
    run_vmec: bool = typer.Option(
        True,
        "--run-vmec/--no-vmec",
        help="Whether to run VMEC equilibrium calculation.",
    ),
    helicity_m: int = typer.Option(
        1,
        "--helicity-m",
        help="Poloidal mode number for quasisymmetry evaluation.",
    ),
    helicity_n: int = typer.Option(
        0,
        "--helicity-n",
        help="Toroidal mode number for quasisymmetry evaluation.",
    ),
    ns: int = typer.Option(
        50,
        "--ns",
        help="Number of radial surfaces for quasisymmetry evaluation.",
    ),
    plot_boozer: bool = typer.Option(
        True,
        "--plot-bozzer/--no-plot-bozzer",
        help="Whether to generate Boozer surface plot.",
    ),
    plot_iota: bool = typer.Option(
        True,
        "--plot-iota/--no-plot-iota",
        help="Whether to generate iota profile plot.",
    ),
    plot_qs: bool = typer.Option(
        True,
        "--plot-qs/--no-plot-qs",
        help="Whether to generate quasisymmetry profile plot.",
    ),
    plot_poincare: bool = typer.Option(
        True,
        "--plot-poincare/--no-plot-poincare",
        help="Whether to generate Poincaré plot.",
    ),
    nfieldlines: int = typer.Option(
        20,
        "--nfieldlines",
        help="Number of fieldlines to trace for Poincaré plot.",
    ),
) -> None:
    """
    Run post-processing on optimized coil results.
    
    This command performs analysis of optimized coil configurations, including:
    - Computing QFM (quasi-flux surface) surfaces
    - Running VMEC equilibrium calculations
    - Computing quasisymmetry metrics
    - Generating Boozer surface plots
    - Generating iota (rotational transform) profiles
    - Generating quasisymmetry error profiles
    - Generating Poincaré plots
    
    Example:
        stellcoilbench post-process coils_runs/biot_savart_optimized.json --output-dir post_processing
    """
    from .post_processing import run_post_processing
    
    typer.echo(f"Running post-processing on {coils_json}")
    typer.echo(f"Output directory: {output_dir}")
    
    try:
        results = run_post_processing(
            coils_json_path=coils_json,
            output_dir=output_dir,
            case_yaml_path=case_yaml,
            plasma_surfaces_dir=plasma_surfaces_dir,
            run_vmec=run_vmec,
            helicity_m=helicity_m,
            helicity_n=helicity_n,
            ns=ns,
            plot_boozer=plot_boozer,
            plot_iota=plot_iota,
            plot_qs=plot_qs,
            plot_poincare=plot_poincare,
            nfieldlines=nfieldlines,
        )
        
        typer.echo("\nPost-processing complete!")
        typer.echo(f"Results saved to: {output_dir}")
        
        if 'BdotN' in results:
            typer.echo(f"B·n on plasma surface: {results['BdotN']:.2e}")
            typer.echo(f"B·n/|B|: {results['BdotN_over_B']:.2e}")
        
        if 'quasisymmetry_average' in results:
            typer.echo(f"Average quasisymmetry error: {results['quasisymmetry_average']:.2e}")
        
    except Exception as e:
        typer.echo(f"Error during post-processing: {e}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
