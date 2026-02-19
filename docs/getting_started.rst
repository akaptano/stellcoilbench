Getting Started
===============

This guide will help you get started with StellCoilBench, from installation to submitting
your first optimization run.

Prerequisites
-------------

**Required Software**
   - Python 3.12 or higher (recommended)
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

**Optional Dependencies for Post-Processing**
   For advanced post-processing features (VMEC equilibrium calculations, Boozer plots, quasisymmetry analysis), you can optionally install:
   
   - **VMEC**: Required for VMEC equilibrium calculations and quasisymmetry metrics
   - **booz_xform**: Required for Boozer surface plots
   
   These are optional - StellCoilBench will work without them, but post-processing features that require them will be skipped.

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

**Optional: Install VMEC and booz_xform for Post-Processing**

   For full post-processing capabilities (VMEC equilibrium calculations, Boozer plots, quasisymmetry analysis), install the optional dependencies:
   
   **VMEC Installation**
   
   VMEC installation instructions vary by platform. See the `simsopt wiki <https://github.com/hiddenSymmetries/simsopt/wiki>`_ for platform-specific installation instructions, including:
   
   - Mac installation (including M1/M2)
   - Linux/Ubuntu installation
   - Various cluster systems (NERSC, PPPL, Princeton, etc.)
   - Common troubleshooting tips
   
   The wiki contains detailed, tested instructions for many computing environments.
   
   **Install booz_xform**
   
   booz_xform is required for performing calculations in Boozer coordinates, e.g. making Boozer surface plots.
   
   .. code-block:: bash
   
      pip install -v git+https://github.com/hiddenSymmetries/booz_xform

**Optional: ParaStell and Coreform Cubit**

   ParaStell enables tetrahedral mesh generation for finite-build coils (when available).
   Coreform Cubit is optional and enables DAGMC export and Cubit-based meshing.
   ParaStell requires PyMOAB (not on PyPI); use the conda environment and setup guide:
   
   - Create env: ``conda env create -f environment-parastell.yml``
   - Install ParaStell: ``pip install --no-deps "parastell @ git+https://github.com/svalinn/parastell.git"``
   - Cubit: Install from `Coreform <https://coreform.com/products/downloads/>`_, add ``bin/`` to ``PYTHONPATH``
   
   See :doc:`parastell_cubit_setup` for full instructions.
   
**Using Conda**
   
   You can also use conda to create an environment, but pip will still handle
   dependency installation:
   
   .. code-block:: bash
   
      conda create -n stellcoilbench python=3.12
      conda activate stellcoilbench
      pip install stellcoilbench
   
   Or for development:
   
   .. code-block:: bash
   
      conda create -n stellcoilbench python=3.12
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
   - Run post-processing (if VMEC/booz_xform installed):
     * Generate Poincaré plots
     * Compute QFM surface
     * Run VMEC equilibrium calculations
     * Generate iota and quasisymmetry profile plots
     * Create Boozer surface plots
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

See :doc:`cases` for the full schema. Quick reference: ``surface``, ``range``, ``ncoils``, ``order``,
``algorithm``, ``max_iterations``, ``max_iter_subopt``, and optional ``coil_objective_terms``,
``fourier_continuation``, ``algorithm_options``.

Creating Your First Case
------------------------

1. Pick a surface from ``plasma_surfaces/`` (e.g. ``input.LandremanPaul2021_QA``)
2. Create ``cases/my_first_case.yaml`` with required fields (see :doc:`cases`)
3. Run ``stellcoilbench validate-config cases/my_first_case.yaml``
4. Run ``stellcoilbench submit-case cases/my_first_case.yaml``
5. Check ``submissions/<surface>/<user>/<timestamp>/`` for zips and PDFs

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
