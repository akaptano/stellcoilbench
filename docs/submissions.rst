Submissions
===========

Submissions are the output of running a StellCoilBench case. Each submission contains
the optimized coil geometry, evaluation metrics, case metadata, and visualization
outputs. Submissions are stored as zip files and automatically processed by CI to
update leaderboards.

Submission Structure
--------------------

Submissions are organized in the repository as follows:

.. code-block::

   submissions/
   └── <surface_name>/
       └── <username>/
           ├── <timestamp>.zip          # Submission archive
           ├── bn_error_3d_plot.pdf     # Optimized coils visualization
           └── bn_error_3d_plot_initial.pdf  # Initial coils visualization

The ``<timestamp>`` format is ``YYYY-MM-DD_HH-MM-SS`` (e.g., ``2025-12-01_01-53-19``).

Submission Zip Contents
-----------------------

Each submission zip file contains the following files:

**results.json**
   Complete evaluation results and metadata. This is the primary file used for
   leaderboard generation. Structure:
   
   .. code-block:: json
   
      {
        "metadata": {
          "method_name": "",
          "method_version": "2025-12-01_01-53-19",
          "contact": "username",
          "hardware": "CPU: ... | RAM: ...",
          "run_date": "2025-12-01T01:53:19.368321"
        },
        "metrics": {
          "final_normalized_squared_flux": 0.0,
          "avg_BdotN_over_B": 1.93e-16,
          "max_BdotN_over_B": 0.244,
          "optimization_time": 2.16,
          "final_min_cs_separation": 1.79,
          "final_min_cc_separation": 1.69,
          "final_total_length": 95.25,
          "final_max_curvature": 0.264,
          "final_average_curvature": 0.264,
          "final_mean_squared_curvature": 0.0696,
          "final_linking_number": 0.0,
          "final_max_max_coil_force": 4.62e6,
          "final_avg_max_coil_force": 4.62e6,
          "final_max_max_coil_torque": 1.81e-7,
          "final_avg_max_coil_torque": 1.55e-7,
          "coil_order": 4.0,
          "num_coils": 4.0,
          "score_primary": 0.0
        }
      }

**case.yaml**
   The case configuration used for this submission. Includes a ``source_case_file``
   field indicating the original case file path. This ensures reproducibility.

**coils.json**
   Optimized coil geometry in JSON format. Contains Fourier coefficients for each
   coil, allowing others to reproduce the exact coil shapes.

**biot_savart_optimized.json**
   Biot-Savart field data computed from the optimized coils. Contains field values
   at evaluation points on the plasma surface.

**Visualization Files**
   VTK format files for 3D visualization:
   
   - ``coils_optimized.vtu``: Coil geometry
   - ``surface_optimized.vts``: Plasma surface with field data

PDF Plots
---------

PDF plots are stored **next to** the zip file (not inside it) for easy access:

**bn_error_3d_plot.pdf**
   High-resolution 3D visualization showing:
   
   - Plasma surface colored by :math:`B_N/|B|` error magnitude
   - Optimized coils colored by current magnitude
   - Colorbars for both surface error and coil currents
   - Publication-quality resolution (300 DPI)

**bn_error_3d_plot_initial.pdf**
   Same visualization for initial (pre-optimization) coils. Useful for comparing
   before and after optimization.

Creating Submissions
--------------------

**Using the CLI**
   
   The primary way to create submissions is via the CLI:
   
   .. code-block:: bash
   
      stellcoilbench submit-case cases/my_case.yaml
   
   This will:
   
   1. Load and validate the case
   2. Initialize coils around the plasma surface
   3. Run the optimization
   4. Evaluate metrics
   5. Generate visualization outputs
   6. Create the submission directory
   7. Zip the submission
   8. Move PDF plots next to the zip

**Submission Directory Structure**
   
   During creation, files are organized as:
   
   .. code-block::
   
      submissions/<surface>/<user>/<timestamp>/
      ├── results.json
      ├── case.yaml
      ├── coils.json
      ├── biot_savart_optimized.json
      ├── coils_optimized.vtu
      └── surface_optimized.vts
   
   After zipping, the directory is removed and replaced with:
   
   .. code-block::
   
      submissions/<surface>/<user>/
      ├── <timestamp>.zip
      ├── bn_error_3d_plot.pdf
      └── bn_error_3d_plot_initial.pdf

**Manual Submission Creation**
   
   For advanced users, you can create submissions manually:
   
   1. Create the submission directory structure
   2. Generate ``results.json`` with required fields
   3. Include ``case.yaml``, ``coils.json``, and visualization files
   4. Zip the directory
   5. Place PDF plots next to the zip
   
   However, using the CLI is strongly recommended to ensure consistency.

Submission Metadata
-------------------

The ``metadata`` section in ``results.json`` contains:

**method_name**
   Optional name for the optimization method. Leave empty for default methods.

**method_version**
   Version identifier (typically the timestamp).

**contact**
   Username or contact information. Extracted from ``git config user.name``.

**hardware**
   Hardware information (CPU, RAM) for reproducibility.

**run_date**
   ISO 8601 timestamp of when the submission was created.

Evaluation Metrics
------------------

The ``metrics`` section contains all computed evaluation metrics:

**Primary Score**
   - ``score_primary``: Normalized squared flux error (used for ranking)
   - ``final_normalized_squared_flux``: Same value (for clarity)

**Field Quality**
   - ``avg_BdotN_over_B``: Average normalized normal field component
   - ``max_BdotN_over_B``: Maximum normalized normal field component
   
   Lower values indicate better field quality (field is more tangent to surface).

**Coil Geometry**
   - ``final_total_length``: Total length of all coils
   - ``final_max_curvature``: Maximum curvature across all coils
   - ``final_average_curvature``: Average curvature
   - ``final_mean_squared_curvature``: Mean squared curvature
   - ``coil_order``: Fourier order used
   - ``num_coils``: Number of base coils

**Separations**
   - ``final_min_cs_separation``: Minimum coil-to-surface distance
   - ``final_min_cc_separation``: Minimum coil-to-coil distance

**Forces and Torques**
   - ``final_max_max_coil_force``: Maximum force magnitude
   - ``final_avg_max_coil_force``: Average of maximum forces per coil
   - ``final_max_max_coil_torque``: Maximum torque magnitude
   - ``final_avg_max_coil_torque``: Average of maximum torques per coil

**Topology**
   - ``final_linking_number``: Linking number between coils

**Performance**
   - ``optimization_time``: Wall-clock time for optimization (seconds)

**Configuration Thresholds**
   These are included for reference but not used in ranking:
   
   - ``flux_threshold``
   - ``cc_threshold`` (coil-coil distance threshold)
   - ``cs_threshold`` (coil-surface distance threshold)
   - ``msc_threshold`` (mean squared curvature threshold)
   - ``curvature_threshold``
   - ``force_threshold``
   - ``torque_threshold``

Leaderboard Processing
----------------------

CI automatically processes submissions:

1. **Scan Submissions**: CI scans ``submissions/`` for all ``*.zip`` files

2. **Extract Results**: For each zip, extracts and parses ``results.json``

3. **Compute Rankings**: Sorts submissions by ``score_primary`` (lower is better)

4. **Group by Surface**: Creates separate leaderboards for each plasma surface

5. **Generate Documentation**: Updates:
   
   - ``docs/leaderboard.json``: Machine-readable leaderboard
   - ``docs/leaderboard.rst``: ReadTheDocs-formatted leaderboard
   - ``docs/leaderboards/*.md``: Per-surface markdown leaderboards

6. **Commit Changes**: Commits updated leaderboards to the repository

Viewing Submissions
-------------------

**List Your Submissions**
   
   .. code-block:: bash
   
      ls submissions/*/$(git config user.name)/
   
   Shows all your submissions across all surfaces.

**View Submission Contents**
   
   Extract a submission zip:
   
   .. code-block:: bash
   
      cd /tmp
      unzip submissions/<surface>/<user>/<timestamp>.zip
      cat results.json | jq .
   
   (Requires ``jq`` for JSON pretty-printing)

**View PDF Plots**
   
   Open PDF plots directly:
   
   .. code-block:: bash
   
      open submissions/<surface>/<user>/bn_error_3d_plot.pdf
   
   Or on Linux:
   
   .. code-block:: bash
   
      xdg-open submissions/<surface>/<user>/bn_error_3d_plot.pdf

**Regenerate Leaderboards Locally**
   
   After creating submissions locally, regenerate leaderboards:
   
   .. code-block:: bash
   
      stellcoilbench update-db
   
   This updates ``docs/leaderboard.json`` and ``docs/leaderboard.rst``.

Submission Best Practices
-------------------------

1. **Use Descriptive Case Names**: Choose case file names that clearly indicate
   the purpose (e.g., ``expert_LandremanPaulQA.yaml`` vs ``test.yaml``).

2. **Include Complete Metadata**: Ensure ``results.json`` includes all required
   metadata fields for reproducibility.

3. **Verify Results**: Check that metrics are reasonable before submitting.
   Unusually high or low values may indicate errors.

4. **Check Visualizations**: Inspect PDF plots to ensure coils are reasonable
   and field quality is acceptable.

5. **Document Changes**: If modifying optimization code, document changes in
   commit messages or case descriptions.

6. **Test Locally First**: Run cases locally before pushing to CI to catch
   errors early.

Troubleshooting
---------------

**Submission Not Appearing in Leaderboard**
   
   - Check that the zip file exists and is valid
   - Verify ``results.json`` is valid JSON
   - Ensure ``score_primary`` is present in metrics
   - Run ``stellcoilbench update-db`` manually

**Invalid JSON Errors**
   
   - Check that ``results.json`` is valid JSON (use ``jq`` or online validator)
   - Ensure all numeric values are valid numbers (not NaN or Inf)
   - Verify all required fields are present

**Missing PDF Plots**
   
   - PDF plots are generated during submission creation
   - If missing, re-run ``stellcoilbench submit-case``
   - Check that matplotlib is installed and working

**Large Submission Files**
   
   - VTK files can be large for high-resolution surfaces
   - Consider reducing surface resolution for testing
   - Production runs should use full resolution
