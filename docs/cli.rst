CLI Reference
=============

StellCoilBench provides a command-line interface (CLI) for running cases, creating
submissions, and managing leaderboards. All commands are accessed via the ``stellcoilbench``
executable.

Command Overview
----------------

The CLI provides the following commands:

- ``submit-case``: Run a case and create a submission
- ``run-case``: Run a case without creating a submission
- ``generate-submission``: Create a submission from existing results
- ``update-db``: Regenerate leaderboards from submissions
- ``validate-config``: Validate a case configuration file

Getting Help
------------

View all available commands:

.. code-block:: bash

   stellcoilbench --help

Get help for a specific command:

.. code-block:: bash

   stellcoilbench submit-case --help

submit-case
-----------

Run a case and create a submission zip file.

**Usage**
   
   .. code-block:: bash
   
      stellcoilbench submit-case <case_file>
   
   **Arguments**
   
   - ``case_file``: Path to the case YAML file (required)
   
   **Description**
   
   This is the primary command for running optimization cases. It:
   
   1. Loads and validates the case configuration
   2. Initializes coils around the plasma surface
   3. Runs the optimization loop
   4. Evaluates metrics
   5. Generates visualization outputs (VTK, PDF)
   6. Creates a submission directory
   7. Zips the submission
   8. Moves PDF plots next to the zip file
   
   The submission is created in:
   
   .. code-block::
   
      submissions/<username>/<timestamp>/all_files.zip
   
   PDF plots are placed next to the zip:
   
   .. code-block::
   
      submissions/<username>/<timestamp>/
      ├── all_files.zip
      ├── bn_error_3d_plot.pdf
      └── bn_error_3d_plot_initial.pdf
   
   **Examples**
   
   Run a basic case:
   
   .. code-block:: bash
   
      stellcoilbench submit-case cases/basic_LandremanPaulQA.yaml
   
   Run an expert case:
   
   .. code-block:: bash
   
      stellcoilbench submit-case cases/expert_LandremanPaulQA.yaml
   
   **Output**
   
   The command prints:
   
   - Case loading progress
   - Optimization progress (if ``verbose: True``)
   - Final metrics summary
   - Submission location
   
   **Exit Codes**
   
   - ``0``: Success
   - ``1``: Error (case file not found, validation error, optimization failure, etc.)

run-case
--------

Run a case and write results to a specified output directory (without creating a submission).

**Usage**
   
   .. code-block:: bash
   
      stellcoilbench run-case <case_file> [OPTIONS]
   
   **Arguments**
   
   - ``case_file``: Path to the case YAML file (required)
   
   **Options**
   
   - ``--coils-out-dir <path>``: Output directory for results (default: ``coils_runs/``)
   
   **Description**
   
   Similar to ``submit-case`` but writes results to a custom directory instead of
   creating a submission. Useful for:
   
   - Testing optimization code changes
   - Running cases without creating submissions
   - Custom output organization
   
   **Examples**
   
   Run a case to a custom directory:
   
   .. code-block:: bash
   
      stellcoilbench run-case cases/my_case.yaml --coils-out-dir my_results/
   
   **Output Structure**
   
   Results are written to:
   
   .. code-block::
   
      <coils-out-dir>/
      ├── results.json
      ├── case.yaml
      ├── coils.json
      ├── biot_savart_optimized.json
      ├── coils_optimized.vtu
      ├── surface_optimized.vts
      ├── bn_error_3d_plot.pdf
      └── bn_error_3d_plot_initial.pdf

generate-submission
-------------------

Create a submission zip from existing results.

**Usage**
   
   .. code-block:: bash
   
      stellcoilbench generate-submission <case_file> <coils_file> [OPTIONS]
   
   **Arguments**
   
   - ``case_file``: Path to the case YAML file (required)
   - ``coils_file``: Path to the coils JSON file (required)
   
   **Options**
   
   - ``--results-dir <path>``: Directory containing results.json (optional)
   
   **Description**
   
   Creates a submission zip from pre-existing optimization results. Useful when:
   
   - You have results from a previous run
   - You want to submit results from external optimization code
   - You need to recreate a submission with updated metadata
   
   **Examples**
   
   Create submission from existing results:
   
   .. code-block:: bash
   
      stellcoilbench generate-submission \\
         cases/my_case.yaml \\
         coils_runs/coils.json \\
         --results-dir coils_runs/
   
   **Requirements**
   
   The results directory must contain:
   
   - ``results.json``: Evaluation metrics
   - ``case.yaml``: Case configuration (or will be generated)
   - ``coils.json``: Coil geometry (or use ``<coils_file>``)
   - ``biot_savart_optimized.json``: Field data (optional)

update-db
---------

Regenerate leaderboards from existing submissions.

**Usage**
   
   .. code-block:: bash
   
      stellcoilbench update-db
   
   **Description**
   
   Scans the ``submissions/`` directory for all zip files, extracts results,
   computes rankings, and regenerates:
   
   - ``docs/leaderboard.json``: Machine-readable leaderboard
   - ``docs/leaderboard.rst``: ReadTheDocs-formatted leaderboard
   - ``docs/leaderboards/*.md``: Per-surface markdown leaderboards
   
   **When to Use**
   
   - After creating submissions locally
   - After manually adding submission zip files
   - To refresh leaderboards without running CI
   - For local development and testing
   
   **Examples**
   
   Regenerate all leaderboards:
   
   .. code-block:: bash
   
      stellcoilbench update-db
   
   **Output**
   
   Prints:
   
   - Number of submissions found
   - Number of surfaces detected
   - Number of leaderboard files generated
   - Any errors encountered
   
   **Exit Codes**
   
   - ``0``: Success
   - ``1``: Error (invalid submissions, file system errors, etc.)

validate-config
---------------

Validate a case configuration file.

**Usage**
   
   .. code-block:: bash
   
      stellcoilbench validate-config <case_file>
   
   **Arguments**
   
   - ``case_file``: Path to the case YAML file (required)
   
   **Description**
   
   Validates a case configuration file and reports any errors. Checks for:
   
   - Required fields (description, surface_params, coils_params, etc.)
   - Valid surface names
   - Valid algorithm names
   - Valid objective term options
   - Algorithm-specific option compatibility
   - Type correctness (numbers, strings, booleans)
   
   **Examples**
   
   Validate a case file:
   
   .. code-block:: bash
   
      stellcoilbench validate-config cases/my_case.yaml
   
   **Output**
   
   Prints validation errors if any, or confirms validity:
   
   .. code-block::
   
      ✓ Configuration is valid
   
   Or:
   
   .. code-block::
   
      ✗ Validation errors:
      - cases/my_case.yaml: optimizer_params.algorithm must be one of ['L-BFGS-B', 'augmented_lagrangian', ...]
   
   **Exit Codes**
   
   - ``0``: Valid configuration
   - ``1``: Validation errors found

Common Workflows
----------------

**Running a Case Locally**
   
   .. code-block:: bash
   
      # Validate first
      stellcoilbench validate-config cases/my_case.yaml
      
      # Run the case
      stellcoilbench submit-case cases/my_case.yaml
      
      # Check results
      ls submissions/*/$(git config user.name)/
   
   **Testing Optimization Changes**
   
   .. code-block:: bash
   
      # Run to custom directory (no submission)
      stellcoilbench run-case cases/test_case.yaml --coils-out-dir test_output/
      
      # Inspect results
      cat test_output/results.json | jq .
      
      # If satisfied, create submission
      stellcoilbench generate-submission \\
         cases/test_case.yaml \\
         test_output/coils.json \\
         --results-dir test_output/
   
   **Updating Leaderboards**
   
   .. code-block:: bash
   
      # After creating submissions
      stellcoilbench update-db
      
      # View updated leaderboard
      cat docs/leaderboard.json | jq .
   
   **Batch Processing**
   
   Run multiple cases:
   
   .. code-block:: bash
   
      for case in cases/*.yaml; do
         stellcoilbench submit-case "$case"
      done
      
      stellcoilbench update-db

Environment Variables
---------------------

**STELLCOILBENCH_SUBMISSIONS_DIR**
   Override the default submissions directory (default: ``submissions/``)
   
   .. code-block:: bash
   
      export STELLCOILBENCH_SUBMISSIONS_DIR=/path/to/submissions
      stellcoilbench submit-case cases/my_case.yaml

**STELLCOILBENCH_DOCS_DIR**
   Override the default docs directory (default: ``docs/``)
   
   .. code-block:: bash
   
      export STELLCOILBENCH_DOCS_DIR=/path/to/docs
      stellcoilbench update-db

Troubleshooting
---------------

**Command Not Found**
   
   Ensure StellCoilBench is installed:
   
   .. code-block:: bash
   
      pip install -e .
   
   Or activate the conda environment:
   
   .. code-block:: bash
   
      conda activate stellcoilbench

**Import Errors**
   
   Check that all dependencies are installed:
   
   .. code-block:: bash
   
      python -c "import simsopt; import numpy; import scipy; import matplotlib"
   
   Install missing packages:
   
   .. code-block:: bash
   
      conda install -c conda-forge simsopt numpy scipy matplotlib pyyaml vtk
      pip install typer

**Case File Not Found**
   
   Use absolute or relative paths:
   
   .. code-block:: bash
   
      stellcoilbench submit-case ./cases/my_case.yaml
      stellcoilbench submit-case /absolute/path/to/case.yaml

**Permission Errors**
   
   Ensure write permissions for:
   
   - ``submissions/`` directory
   - ``docs/`` directory (for ``update-db``)
   - Output directories

**Optimization Failures**
   
   Check:
   
   - Case file is valid (use ``validate-config``)
   - Plasma surface file exists
   - Coil parameters are reasonable
   - Sufficient disk space
   - Sufficient memory
