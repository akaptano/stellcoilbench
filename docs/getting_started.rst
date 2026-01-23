Getting Started
===============

This guide will help you get started with StellCoilBench, from installation to submitting
your first optimization run.

Prerequisites
-------------

**Required Software**
   - Python 3.8 or higher
   - Conda (recommended) or pip for package management
   - Git for version control
   - Access to the StellCoilBench repository

**Required Python Packages**
   StellCoilBench depends on several scientific Python packages:
   
   - ``simsopt``: Stellarator optimization library (provides coil geometry, Biot-Savart, etc.)
   - ``numpy``: Numerical computing
   - ``scipy``: Optimization algorithms
   - ``matplotlib``: Plotting and visualization
   - ``pyyaml``: YAML configuration parsing
   - ``typer``: Command-line interface
   - ``vtk``: Visualization output

Installation
------------

**Using Pip (Recommended)**
   
   Install StellCoilBench and all required dependencies:
   
   .. code-block:: bash
   
      pip install stellcoilbench
   
   The installation will automatically install all required dependencies, including
   the correct version of simsopt from the specified repository.
   
   For development, install in editable mode:
   
   .. code-block:: bash
   
      git clone <repository-url>
      cd stellcoilbench
      pip install -e .
   
   **Optional: Install Documentation Dependencies**
   
   To build the documentation locally, install the optional DOCS dependencies:
   
   .. code-block:: bash
   
      pip install stellcoilbench[DOCS]
   
   Or for development mode:
   
   .. code-block:: bash
   
      pip install -e ".[DOCS]"

**Using Conda**
   
   You can also use conda to create an environment, but pip will still handle
   dependency installation:
   
   .. code-block:: bash
   
      conda create -n stellcoilbench python=3.10
      conda activate stellcoilbench
      pip install stellcoilbench
   
   Or for development:
   
   .. code-block:: bash
   
      conda create -n stellcoilbench python=3.10
      conda activate stellcoilbench
      git clone <repository-url>
      cd stellcoilbench
      pip install -e .

Verify Installation
-------------------

Check that the CLI is available:
   
.. code-block:: bash

   stellcoilbench --help

You should see a list of available commands.

Fastest Path: CI-Driven Workflow
----------------------------------

The fastest way to run a case is through the CI workflow:

1. **Add a Case File**
   
   Create a new YAML file in ``cases/``. For example, ``cases/my_test_case.yaml``:
   
   .. code-block:: yaml
   
      description: "My test case"
      surface_params:
        surface: "input.LandremanPaul2021_QA"
        range: "half period"
      coils_params:
        ncoils: 4
        order: 4
      optimizer_params:
        algorithm: "L-BFGS-B"
        max_iterations: 200
        max_iter_subopt: 10
        verbose: False
      coil_objective_terms:
        total_length: "l2_threshold"
        coil_coil_distance: "l1_threshold"
        coil_surface_distance: "l1_threshold"
        coil_curvature: "lp_threshold"
        coil_curvature_p: 2

2. **Commit and Push**
   
   .. code-block:: bash
   
      git add cases/my_test_case.yaml
      git commit -m "Add my test case"
      git push

3. **CI Runs Automatically**
   
   CI will detect the new case, run it, and update the leaderboards. Check the CI logs
   to see the progress.

4. **View Results**
   
   After CI completes, check:
   
   - ``submissions/<surface>/<your-username>/<timestamp>/`` for your submission zip and PDF plots
     (e.g., ``submissions/LandremanPaul2021_QA/akaptano/01-23-2026_00-45/``)
   - ``docs/leaderboard.rst`` for updated leaderboards

Local Development Workflow
---------------------------

For local development and testing, you can run cases directly:

**Step 1: Run a Case**
   
   .. code-block:: bash
   
      stellcoilbench submit-case cases/basic_LandremanPaulQA.yaml
   
   This will:
   
   - Load the case configuration
   - Initialize coils around the plasma surface
   - Run the optimization
   - Evaluate metrics
   - Create a submission directory under ``submissions/``
   - Zip the submission and move PDF plots next to the zip

**Step 2: Check Outputs**
   
   List your submissions:
   
   .. code-block:: bash
   
      ls submissions/*/$(git config user.name)/
   
   You should see timestamped zip files and PDF plots.

**Step 3: Inspect Results**
   
   Open the PDF plots to visualize:
   
   - B_N error on the plasma surface (colored by error magnitude)
   - Coils colored by current magnitude
   - Separate plots for initial and optimized coils

**Step 4: Update Leaderboards Locally**
   
   To regenerate leaderboards from local submissions:
   
   .. code-block:: bash
   
      stellcoilbench update-db
   
   This updates ``docs/leaderboard.json`` and ``docs/leaderboard.rst``.

Understanding Case Files
------------------------

Case files define the optimization problem. See :doc:`cases` for complete documentation.
Here's a quick overview:

**Surface Parameters**
   - ``surface``: Name of the plasma surface file (without extension)
   - ``range``: Surface range to use ("half period", "full period", etc.)

**Coil Parameters**
   - ``ncoils``: Number of base coils (before applying stellarator symmetry)
   - ``order``: Fourier order for coil representation (final order if Fourier continuation is used)
   
   **Fourier Continuation** (optional):
   
   - ``fourier_continuation.enabled``: Enable Fourier continuation (see :ref:`fourier-continuation`)
   - ``fourier_continuation.orders``: List of Fourier orders to use in sequence (e.g., ``[4, 6, 8]``)

**Optimizer Parameters**
   - ``algorithm``: Optimization algorithm ("L-BFGS-B", "augmented_lagrangian", etc.)
   - ``max_iterations``: Maximum optimization iterations
   - ``max_iter_subopt``: Maximum iterations for sub-optimization (for augmented Lagrangian)
   - ``verbose``: Print optimization progress
   - ``algorithm_options``: Algorithm-specific options (e.g., ``ftol``, ``gtol``)
     - For L-BFGS-B, defaults are ``ftol: 1e-12``, ``gtol: 1e-12``, ``tol: 1e-12``

**Objective Terms**
   Each term can use different penalty types:
   
   - ``l1``, ``l1_threshold``: L1 norm or thresholded L1
   - ``l2``, ``l2_threshold``: L2 norm or thresholded L2
   - ``lp``, ``lp_threshold``: Lp norm or thresholded Lp (requires ``*_p`` parameter)
   
   Available terms:
   
   - ``total_length``: Penalize total coil length
   - ``coil_coil_distance``: Penalize coil-to-coil distances
   - ``coil_surface_distance``: Penalize coil-to-surface distances
   - ``coil_curvature``: Penalize coil curvature (requires ``coil_curvature_p``)
   - ``coil_mean_squared_curvature``: Penalize mean squared curvature
   - ``coil_arclength_variation``: Penalize arclength variation
   - ``linking_number``: Include linking number constraint (use empty string ``""``)
   - ``coil_coil_force``: Penalize coil forces (requires ``coil_coil_force_p``)
   - ``coil_coil_torque``: Penalize coil torques (requires ``coil_coil_torque_p``)

Creating Your First Case
------------------------

Let's create a simple case step by step:

1. **Choose a Plasma Surface**
   
   Check ``plasma_surfaces/`` for available surfaces. Common choices:
   
   - ``input.LandremanPaul2021_QA``: Quasi-axisymmetric configuration
   - ``input.circular_tokamak``: Simple tokamak for testing
   - ``input.W7-X_without_coil_ripple_beta0p05_d23p4_tm``: W7-X configuration

2. **Create the Case File**
   
   Create ``cases/my_first_case.yaml``:
   
   .. code-block:: yaml
   
      description: "My first optimization case"
      surface_params:
        surface: "input.LandremanPaul2021_QA"
        range: "half period"
      coils_params:
        ncoils: 4
        order: 4
      optimizer_params:
        algorithm: "L-BFGS-B"
        max_iterations: 100
        max_iter_subopt: 10
        verbose: True
        algorithm_options:
          ftol: 1e-12
          gtol: 1e-12
      coil_objective_terms:
        total_length: "l2_threshold"
        coil_coil_distance: "l1_threshold"
        coil_surface_distance: "l1_threshold"
        coil_curvature: "lp_threshold"
        coil_curvature_p: 2

3. **Validate the Case**
   
   .. code-block:: bash
   
      stellcoilbench validate-config cases/my_first_case.yaml
   
   This checks for errors in the configuration.

4. **Run the Case**
   
   .. code-block:: bash
   
      stellcoilbench submit-case cases/my_first_case.yaml
   
   This may take several minutes depending on the case complexity.

5. **Check Results**
   
   After completion, check:
   
   - ``submissions/LandremanPaul2021_QA/<your-username>/<timestamp>/`` for the submission
   - PDF plots next to the zip file for visualizations
   - Console output for optimization progress and final metrics

Common Workflows
----------------

**Testing a New Optimization Method**
   
   1. Create a case file with your desired configuration
   2. Modify ``coil_optimization.py`` to implement your method (or use algorithm options)
   3. Run the case locally: ``stellcoilbench submit-case cases/my_case.yaml``
   4. Compare results with existing leaderboard entries
   5. Commit and push to trigger CI

**Comparing Algorithms**
   
   1. Create multiple case files with different ``algorithm`` settings
   2. Run all cases: ``stellcoilbench submit-case cases/case1.yaml``, etc.
   3. Compare results in the leaderboard
   4. Use ``stellcoilbench update-db`` to regenerate leaderboards locally

**Benchmarking on Multiple Surfaces**
   
   1. Create case files for different surfaces
   2. Run all cases (locally or via CI)
   3. Compare performance across surfaces in the leaderboard

Troubleshooting
---------------

**Import Errors**
   
   If you see import errors, ensure StellCoilBench is properly installed:
   
   .. code-block:: bash
   
      pip install stellcoilbench
   
   Or if you're in development mode:
   
   .. code-block:: bash
   
      pip install -e .

**Case Validation Errors**
   
   Run ``stellcoilbench validate-config cases/your_case.yaml`` to check for configuration
   errors. Common issues:
   
   - Missing required fields
   - Invalid algorithm options
   - Invalid objective term options

**Optimization Failures**
   
   If optimization fails:
   
   - Check that the plasma surface file exists in ``plasma_surfaces/``
   - Verify coil parameters are reasonable (not too many coils or too high order)
   - Try reducing ``max_iterations`` for faster debugging
   - Enable ``verbose: True`` to see optimization progress

**Leaderboard Not Updating**
   
   - Ensure submissions exist in ``submissions/``
   - Run ``stellcoilbench update-db`` manually
   - Check that ``results.json`` files are valid JSON
