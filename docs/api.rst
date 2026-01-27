API Reference
==============

This section provides detailed documentation for the StellCoilBench Python API.
The API is organized into several modules, each handling a specific aspect of the
benchmarking framework.

Module Overview
---------------

- **``stellcoilbench.cli``**: Command-line interface implementation
- **``stellcoilbench.coil_optimization``**: Core coil optimization logic
- **``stellcoilbench.config_scheme``**: Configuration data structures
- **``stellcoilbench.evaluate``**: Case evaluation and metric computation
- **``stellcoilbench.update_db``**: Leaderboard generation and management
- **``stellcoilbench.validate_config``**: Configuration validation

Configuration Module
--------------------

.. automodule:: stellcoilbench.config_scheme
   :members:
   :undoc-members:
   :show-inheritance:

The ``config_scheme`` module defines data structures for case configurations and
submission metadata.

**CaseConfig**
   Represents a complete case configuration loaded from a YAML file. Contains:
   
   - ``description``: Case description
   - ``surface_params``: Plasma surface configuration
   - ``coils_params``: Coil geometry parameters
   - ``optimizer_params``: Optimization algorithm settings
   - ``coil_objective_terms``: Objective function terms
   
   Methods:
   
   - ``from_dict(data: Dict[str, Any]) -> CaseConfig``: Create from dictionary
   - ``to_dict() -> Dict[str, Any]``: Convert to dictionary

**SubmissionMetadata**
   Metadata for submissions, including method information, contact details,
   hardware information, and timestamps.

Evaluation Module
-----------------

.. automodule:: stellcoilbench.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

The ``evaluate`` module handles case loading, metric computation, and leaderboard
building.

**load_case_config(case_dir: Path) -> CaseConfig**
   Load a case configuration from a directory containing ``case.yaml``.
   
   Parameters:
   
   - ``case_dir``: Path to case directory
   
   Returns:
   
   - ``CaseConfig``: Loaded configuration
   
   Raises:
   
   - ``FileNotFoundError``: If ``case.yaml`` not found
   - ``ValueError``: If configuration is invalid

**evaluate_case(case_cfg: CaseConfig, results_dict: Dict[str, Any]) -> Dict[str, Any]**
   Evaluate a case and compute all metrics.
   
   Parameters:
   
   - ``case_cfg``: Case configuration
   - ``results_dict``: Dictionary containing optimization results
   
   Returns:
   
   - ``Dict[str, Any]``: Updated results dictionary with computed metrics
   
   Computed metrics include:
   
   - Normalized squared flux error (primary score)
   - :math:`B_N/|B|` error (average and maximum)
   - Coil geometry metrics (curvature, length, separations)
   - Force and torque metrics
   - Linking number
   - Optimization time

**build_leaderboard(submissions: Iterable[Tuple[Path, Dict[str, Any]]], primary_score_key: str = "score_primary") -> Dict[str, Any]**
   Build a leaderboard from submission results.
   
   Parameters:
   
   - ``submissions``: Iterable of (path, results_dict) tuples
   - ``primary_score_key``: Key to use for ranking (default: "score_primary")
   
   Returns:
   
   - ``Dict[str, Any]``: Leaderboard dictionary with ranked entries

Coil Optimization Module
-------------------------

.. automodule:: stellcoilbench.coil_optimization
   :members:
   :undoc-members:
   :show-inheritance:

The ``coil_optimization`` module contains the core optimization logic.

**optimize_coils(case_path: Path, coils_out_path: Path, case_cfg: CaseConfig | None = None, output_dir: Path | None = None, surface_resolution: int = 16) -> Dict[str, Any]**
   Run coil optimization for a given case.
   
   This is the main entry point for optimization. It:
   
   1. Loads the case configuration
   2. Loads the plasma surface
   3. Initializes coils
   4. Builds the objective function
   5. Runs the optimization
   6. Saves results
   
   Parameters:
   
   - ``case_path``: Path to case directory or case YAML file
   - ``coils_out_path``: Where to write optimized coil geometry (JSON)
   - ``case_cfg``: Optional pre-loaded CaseConfig (if None, loads from case_path)
   - ``output_dir``: Directory for VTK and other outputs (default: coils_out_path parent)
   - ``surface_resolution``: Resolution of plasma surface (nphi=ntheta) for evaluation (default: 32).
     Lower values speed up optimization but reduce accuracy. Use 8 for faster unit tests.
   
   Returns:
   
   - ``Dict[str, Any]``: Optimization results dictionary
   
   The results dictionary contains:
   
   - ``coils``: Optimized coil objects
   - ``bs``: BiotSavart field calculator
   - ``surface``: Plasma surface
   - ``metrics``: Computed metrics
   - ``optimization_info``: Optimization algorithm information

**initialize_coils_loop(s: SurfaceRZFourier, out_dir: Path | str = '', target_B: float = 5.7, ncoils: int = 4, order: int = 16, coil_width: float = 0.4, regularization: Callable = regularization_circ)**
   Initialize coils around a plasma surface.
   
   Creates equally-spaced coils with Fourier representation and adjusts total
   current to produce a target B-field on-axis.
   
   Parameters:
   
   - ``s``: Plasma boundary surface
   - ``out_dir``: Output directory for saved files
   - ``target_B``: Target magnetic field strength in Tesla
   - ``ncoils``: Number of base coils
   - ``order``: Fourier order for coil curves
   - ``coil_width``: Coil width in meters
   - ``regularization``: Regularization function
   
   Returns:
   
   - ``coils``: List of initialized coil objects

**optimize_coils_loop(s: SurfaceRZFourier, target_B: float = 5.7, out_dir: Path | str = '', max_iterations: int = 30, ncoils: int = 4, order: int = 16, verbose: bool = False, regularization: Callable = regularization_circ, coil_objective_terms: Dict[str, Any] | None = None, initial_coils: list | None = None, surface_resolution: int = 32, skip_post_processing: bool = False, case_path: Path | None = None, **kwargs)**
   Complete coil optimization including initialization and optimization.
   
   This function combines initialization and optimization into a single call.
   It initializes coils with the target B-field and then optimizes them using
   the specified algorithm.
   
   Parameters:
   
   - ``s``: Plasma boundary surface
   - ``target_B``: Target magnetic field strength
   - ``out_dir``: Output directory
   - ``max_iterations``: Maximum optimization iterations
   - ``ncoils``: Number of base coils
   - ``order``: Fourier order
   - ``verbose``: Print progress
   - ``regularization``: Regularization function
   - ``coil_objective_terms``: Objective function terms dictionary
   - ``initial_coils``: Optional pre-initialized coils (for Fourier continuation)
   - ``surface_resolution``: Resolution of plasma surface (nphi=ntheta) for evaluation (default: 32).
     Lower values speed up optimization but reduce accuracy. Use 8 for faster unit tests.
   - ``skip_post_processing``: Skip post-processing steps (default: False)
   - ``case_path``: Path to case directory containing case.yaml (for post-processing)
   - ``**kwargs``: Additional constraint thresholds
   
   Returns:
   
   - ``coils``: Optimized coil objects
   - ``results``: Optimization results dictionary

**optimize_coils_with_fourier_continuation(s: SurfaceRZFourier, fourier_orders: list[int], target_B: float = 5.7, out_dir: Path | str = '', max_iterations: int = 30, ncoils: int = 4, verbose: bool = False, regularization: Callable | None = regularization_circ, coil_objective_terms: Dict[str, Any] | None = None, surface_resolution: int = 32, case_path: Path | None = None, **kwargs) -> tuple[list, Dict[str, Any]]**
   Perform coil optimization with Fourier continuation.
   
   This function solves a sequence of coil optimizations, starting with a low
   Fourier order and progressively increasing to higher orders. Each step uses
   the previous solution as an initial condition, helping achieve convergence
   for complex problems.
   
   Parameters:
   
   - ``s``: Plasma boundary surface
   - ``fourier_orders``: List of Fourier orders to use in sequence (must be
     positive integers in ascending order)
   - ``target_B``: Target magnetic field strength
   - ``out_dir``: Output directory (subdirectories ``order_N/`` are created
     for each step)
   - ``max_iterations``: Maximum optimization iterations per order
   - ``ncoils``: Number of base coils
   - ``verbose``: Print progress
   - ``regularization``: Regularization function (can be None)
   - ``coil_objective_terms``: Objective function terms dictionary
   - ``surface_resolution``: Resolution of plasma surface (nphi=ntheta) for evaluation (default: 32).
     Lower values speed up optimization but reduce accuracy. Use 8 for faster unit tests.
   - ``case_path``: Path to case directory containing case.yaml (for post-processing)
   - ``**kwargs``: Additional constraint thresholds and algorithm options
   
   Returns:
   
   - ``coils``: Final optimized coil objects (highest order)
   - ``results``: Combined results dictionary containing:
     
     - ``fourier_continuation``: ``True`` flag
     - ``fourier_orders``: List of orders used
     - ``final_order``: Final Fourier order
     - ``continuation_results``: List of results dictionaries for each step
     - Final step results at top level
   
   Each step saves results to ``out_dir/order_N/`` including VTK files and
   B_N error PDFs.

**\_extend_coils_to_higher_order(coils: list, new_order: int, s: SurfaceRZFourier, ncoils: int, regularization: Callable | None = None, coil_width: float = 0.4) -> list**
   Extend coils from a lower Fourier order to a higher order.
   
   This helper function takes coils optimized at a lower order and extends them
   to a higher order by copying existing Fourier coefficients and padding new
   modes with zeros.
   
   Parameters:
   
   - ``coils``: List of Coil objects from previous optimization (lower order)
   - ``new_order``: Target Fourier order for the extended coils
   - ``s``: Plasma surface (needed for creating new curves)
   - ``ncoils``: Number of base coils
   - ``regularization``: Regularization function (can be None)
   - ``coil_width``: Coil width parameter
   
   Returns:
   
   - ``list``: New list of Coil objects with extended Fourier order
   
   The function preserves the geometry of the low-order coils while providing
   additional degrees of freedom for refinement at higher order.

**regularization_circ(coil_width: float) -> Callable**
   Create a circular regularization function for coils.
   
   Parameters:
   
   - ``coil_width``: Target coil width
   
   Returns:
   
   - Regularization function

**augmented_lagrangian_method(objective: Callable, constraints: List[Callable], x0: np.ndarray, max_iterations: int = 100, max_iter_subopt: int = 10, verbose: bool = False, **kwargs) -> Dict[str, Any]**
   Augmented Lagrangian optimization method.
   
   Implements the augmented Lagrangian method for constrained optimization.
   Handles both equality and inequality constraints.
   
   Parameters:
   
   - ``objective``: Objective function
   - ``constraints``: List of constraint functions
   - ``x0``: Initial guess
   - ``max_iterations``: Maximum outer iterations
   - ``max_iter_subopt``: Maximum sub-optimization iterations
   - ``verbose``: Print progress
   - ``**kwargs``: Additional optimizer options
   
   Returns:
   
   - ``Dict[str, Any]``: Optimization results

**LinearPenalty**
   Class for linear penalty terms in the objective function.
   
   Supports L1, L2, and Lp norms, with optional thresholding. Can be combined
   with ``Weight`` objects for scaling.
   
   Methods:
   
   - ``J(x: np.ndarray) -> float``: Compute penalty value
   - ``dJ(x: np.ndarray) -> np.ndarray``: Compute penalty gradient
   - ``__add__``, ``__radd__``: Support for ``sum()``
   - ``__mul__``, ``__rmul__``: Support for ``Weight`` scaling

**load_coils_config(config_path: Path) -> Dict[str, Any]**
   Load coil configuration from JSON file.
   
   Parameters:
   
   - ``config_path``: Path to coils.json file
   
   Returns:
   
   - ``Dict[str, Any]``: Coil configuration dictionary

**save_coils_config(coils: List, config_path: Path) -> None**
   Save coil configuration to JSON file.
   
   Parameters:
   
   - ``coils``: List of coil objects
   - ``config_path``: Path to save coils.json

**coils_to_vtk(coils: List, filename: Path) -> None**
   Export coils to VTK format for visualization.
   
   Parameters:
   
   - ``coils``: List of coil objects
   - ``filename``: Output VTK file path

**plot_bn_error_3d(surface, bs, coils, out_dir: Path, filename: str = "bn_error_3d_plot.pdf", title: str = "B_N/|B| Error on Plasma Surface with Optimized Coils", plot_upsample: int = 3) -> None**
   Create 3D visualization of B_N error on plasma surface.
   
   Generates a high-resolution PDF plot showing:
   
   - Plasma surface colored by :math:`B_N/|B|` error magnitude
   - Coils colored by current magnitude
   - Colorbars for both
   
   Parameters:
   
   - ``surface``: Plasma surface object
   - ``bs``: BiotSavart field calculator
   - ``coils``: List of coil objects
   - ``out_dir``: Output directory
   - ``filename``: Output PDF filename
   - ``title``: Plot title
   - ``plot_upsample``: Surface upsampling factor for higher resolution

Update Database Module
----------------------

.. automodule:: stellcoilbench.update_db
   :members:
   :undoc-members:
   :show-inheritance:

The ``update_db`` module handles leaderboard generation and management.

**update_database(repo_root: Path, submissions_root: Path | None = None, docs_dir: Path | None = None, cases_root: Path | None = None, plasma_surfaces_dir: Path | None = None) -> None**
   Main function to update leaderboards from submissions.
   
   Scans submissions directory, loads results, computes rankings, and generates
   leaderboard files.
   
   Parameters:
   
   - ``repo_root``: Repository root directory
   - ``submissions_root``: Submissions directory (default: repo_root / "submissions")
   - ``docs_dir``: Documentation directory (default: repo_root / "docs")
   - ``cases_root``: Cases directory (default: repo_root / "cases")
   - ``plasma_surfaces_dir``: Plasma surfaces directory (default: repo_root / "plasma_surfaces")

**build_surface_leaderboards(leaderboard: Dict[str, Any], submissions_root: Path, plasma_surfaces_dir: Path) -> Dict[str, Dict[str, Any]]**
   Group leaderboard entries by plasma surface.
   
   Parameters:
   
   - ``leaderboard``: Overall leaderboard dictionary
   - ``submissions_root``: Submissions directory
   - ``plasma_surfaces_dir``: Plasma surfaces directory
   
   Returns:
   
   - ``Dict[str, Dict[str, Any]]``: Dictionary mapping surface names to leaderboard entries

**write_rst_leaderboard(leaderboard: Dict[str, Any], out_rst: Path, surface_leaderboards: Dict[str, Dict[str, Any]]) -> None**
   Write ReadTheDocs-formatted leaderboard.
   
   Generates a comprehensive RST file with embedded tables for all surfaces.
   
   Parameters:
   
   - ``leaderboard``: Overall leaderboard dictionary
   - ``out_rst``: Output RST file path
   - ``surface_leaderboards``: Per-surface leaderboards

**write_markdown_leaderboard(leaderboard: Dict[str, Any], out_md: Path) -> None**
   Write markdown-formatted leaderboard.
   
   Parameters:
   
   - ``leaderboard``: Leaderboard dictionary
   - ``out_md``: Output markdown file path

**write_surface_leaderboards(surface_leaderboards: Dict[str, Dict[str, Any]], docs_dir: Path, repo_root: Path) -> list[str]**
   Write per-surface markdown leaderboards.
   
   Parameters:
   
   - ``surface_leaderboards``: Per-surface leaderboards
   - ``docs_dir``: Documentation directory
   - ``repo_root``: Repository root
   
   Returns:
   
   - ``list[str]``: List of generated surface names

**write_surface_leaderboard_index(surface_names: list[str], docs_dir: Path) -> None**
   Write index file for surface leaderboards.
   
   Parameters:
   
   - ``surface_names``: List of surface names
   - ``docs_dir``: Documentation directory

**load_submissions(submissions_root: Path) -> Iterable[Tuple[str, Path, Dict[str, Any]]]**
   Load all submissions from directory.
   
   Handles both regular directories and zip files. Extracts results.json from
   zips as needed.
   
   Parameters:
   
   - ``submissions_root``: Submissions directory
   
   Yields:
   
   - ``(method_key, path, data)``: Method key, submission path, and results data

**metric_shorthand(metric_name: str) -> str**
   Convert metric names to compact shorthand for display.
   
   Parameters:
   
   - ``metric_name``: Full metric name
   
   Returns:
   
   - Shorthand/acronym (e.g., "f_B" for "final_normalized_squared_flux")

**metric_definition(metric_name: str) -> str**
   Get detailed mathematical definition for a metric.
   
   Parameters:
   
   - ``metric_name``: Metric name
   
   Returns:
   
   - LaTeX-formatted mathematical definition

Post-Processing Module
----------------------

.. automodule:: stellcoilbench.post_processing
   :members:
   :undoc-members:
   :show-inheritance:

The ``post_processing`` module handles post-optimization analysis including VMEC
equilibrium calculations, Poincaré plots, quasisymmetry analysis, and Boozer surface plots.

**run_post_processing(coils_json_path: Path, output_dir: Path, case_yaml_path: Optional[Path] = None, plasma_surfaces_dir: Optional[Path] = None, run_vmec: bool = True, helicity_m: int = 1, helicity_n: int = 0, ns: int = 50, plot_boozer: bool = True, plot_poincare: bool = True, nfieldlines: int = 20, mpi: Optional[Any] = None) -> Dict[str, Any]**
   Run complete post-processing pipeline.
   
   This function:
   
   1. Loads coils and plasma surface
   2. Generates Poincaré plot (if requested)
   3. Computes QFM surface
   4. Optionally runs VMEC equilibrium
   5. Computes quasisymmetry metrics
   6. Generates VMEC-dependent plots (Boozer, iota, quasisymmetry)
   
   Parameters:
   
   - ``coils_json_path``: Path to coils JSON file
   - ``output_dir``: Directory where output files will be saved
   - ``case_yaml_path``: Path to case.yaml file (optional)
   - ``plasma_surfaces_dir``: Directory containing plasma surface files (optional)
   - ``run_vmec``: Whether to run VMEC equilibrium calculation (default: True)
   - ``helicity_m``: Poloidal mode number for quasisymmetry (default: 1)
   - ``helicity_n``: Toroidal mode number for quasisymmetry (default: 0)
   - ``ns``: Number of radial surfaces for quasisymmetry evaluation (default: 50)
   - ``plot_boozer``: Whether to generate Boozer surface plot (default: True)
   - ``plot_poincare``: Whether to generate Poincaré plot (default: True)
   - ``nfieldlines``: Number of fieldlines to trace for Poincaré plot (default: 20)
   - ``mpi``: MPI partition for parallel execution (optional)
   
   Returns:
   
   - ``Dict[str, Any]``: Dictionary containing post-processing results:
     - ``qfm_surface``: QFM surface object
     - ``quasisymmetry_average``: Average quasisymmetry error
     - ``quasisymmetry_profile``: Radial quasisymmetry profile
     - ``vmec``: VMEC equilibrium object (if run_vmec=True)

Validate Config Module
----------------------

.. automodule:: stellcoilbench.validate_config
   :members:
   :undoc-members:
   :show-inheritance:

The ``validate_config`` module provides configuration validation.

**validate_case_config(data: Dict[str, Any], file_path: Path | None = None) -> List[str]**
   Validate a case configuration dictionary.
   
   Checks for:
   
   - Required fields
   - Valid surface names
   - Valid algorithm names
   - Valid objective term options
   - Algorithm-specific option compatibility
   - Type correctness
   
   Parameters:
   
   - ``data``: Configuration dictionary
   - ``file_path``: Optional file path for error messages
   
   Returns:
   
   - ``List[str]``: List of error messages (empty if valid)

**validate_algorithm_options(algorithm: str, options: Dict[str, Any]) -> List[str]**
   Validate algorithm-specific options.
   
   Parameters:
   
   - ``algorithm``: Algorithm name
   - ``options``: Options dictionary
   
   Returns:
   
   - ``List[str]``: List of error messages (empty if valid)

CLI Module
----------

.. automodule:: stellcoilbench.cli
   :members:
   :undoc-members:
   :show-inheritance:

The ``cli`` module implements the command-line interface.

**app**
   Typer application instance. Commands are registered as functions decorated
   with ``@app.command()``.

**NumpyJSONEncoder**
   Custom JSON encoder that handles numpy types and arrays. Used for serializing
   results to JSON.

**submit_case(case_file: Path) -> None**
   CLI command: Run a case and create a submission.
   
   See :doc:`cli` for usage details.

**run_case(case_file: Path, coils_out_dir: Path = "coils_runs/") -> None**
   CLI command: Run a case without creating a submission.
   
   See :doc:`cli` for usage details.

**generate_submission(case_file: Path, coils_file: Path, results_dir: Path | None = None) -> None**
   CLI command: Create a submission from existing results.
   
   See :doc:`cli` for usage details.

**update_db() -> None**
   CLI command: Regenerate leaderboards.
   
   See :doc:`cli` for usage details.

**validate_config(case_file: Path) -> None**
   CLI command: Validate a case configuration.
   
   See :doc:`cli` for usage details.

Usage Examples
--------------

**Running Optimization Programmatically**
   
   .. code-block:: python
   
      from pathlib import Path
      from stellcoilbench.coil_optimization import optimize_coils
      from stellcoilbench.evaluate import load_case_config, evaluate_case
      
      # Load case
      case_path = Path("cases/basic_LandremanPaulQA.yaml")
      case_cfg = load_case_config(case_path)
      
      # Run optimization
      results = optimize_coils(
          case_path=case_path,
          coils_out_path=Path("output/coils.json"),
          case_cfg=case_cfg,
          output_dir=Path("output/")
      )
      
      # Evaluate metrics
      metrics = evaluate_case(case_cfg, results)
      print(f"Primary score: {metrics['metrics']['score_primary']}")

**Creating Custom Objective Terms**
   
   .. code-block:: python
   
      from stellcoilbench.coil_optimization import LinearPenalty
      from simsopt.objectives import Weight
      
      # Create a custom penalty term
      penalty = LinearPenalty(
          func=lambda x: compute_something(x),
          threshold=1.0,
          penalty_type="l2_threshold"
      )
      
      # Scale with weight
      weighted_penalty = Weight(1e-3) * penalty
      
      # Add to objective
      objective = flux_term + weighted_penalty

**Updating Leaderboards Programmatically**
   
   .. code-block:: python
   
      from pathlib import Path
      from stellcoilbench.update_db import update_database
      
      # Update leaderboards
      update_database(
          repo_root=Path("."),
          submissions_root=Path("submissions/"),
          docs_dir=Path("docs/")
      )

Next Steps
----------

- **Getting Started**: See :doc:`getting_started` for a tutorial
- **Cases**: Learn about case files in :doc:`cases`
- **CLI Reference**: See :doc:`cli` for command-line usage
- **Leaderboard**: View results in :doc:`leaderboard`