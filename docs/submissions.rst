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
   └── <surface>/
       └── <username>/
           └── <timestamp>/
               ├── all_files.zip            # Submission archive
               ├── bn_error_3d_plot.pdf     # Optimized coils visualization
               └── bn_error_3d_plot_initial.pdf  # Initial coils visualization

The ``<surface>`` is the plasma surface name (e.g., ``LandremanPaul2021_QA``).
The ``<timestamp>`` format is ``MM-DD-YYYY_HH-MM`` (e.g., ``01-23-2026_00-45``).

Example: ``submissions/LandremanPaul2021_QA/akaptano/01-23-2026_00-45/``

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

Run ``stellcoilbench submit-case cases/my_case.yaml``. The CLI loads the case,
runs optimization, evaluates metrics, and writes to ``submissions/<surface>/<user>/<timestamp>/``.
After zipping, only ``all_files.zip`` and PDF plots remain in that directory.

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

- List: ``ls submissions/<surface>/$(git config user.name)/``
- Extract: ``unzip submissions/<surface>/<user>/<timestamp>/all_files.zip``
- View PDFs: ``open submissions/<surface>/<user>/<timestamp>/bn_error_3d_plot.pdf``
- Regenerate leaderboards: ``stellcoilbench update-db``

Submission Best Practices
-------------------------

Use descriptive case names, verify metrics and PDF plots before submitting, and run locally first.

Troubleshooting
---------------

Submissions not appearing: ensure the zip exists, ``results.json`` is valid, and ``score_primary`` is present.
Run ``stellcoilbench update-db``. Missing PDFs: re-run ``stellcoilbench submit-case``.
