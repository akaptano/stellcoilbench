CLI Reference
=============

StellCoilBench provides a command-line interface for running cases, creating submissions, and managing leaderboards.

Get help for any command:

.. code-block:: bash

   stellcoilbench --help
   stellcoilbench <command> --help

submit-case
-----------

Run a case and create a submission zip file. Runs optimization and creates a submission in ``submissions/<surface>/<username>/<timestamp>/``.

.. code-block:: bash

   stellcoilbench submit-case <case_file>
   stellcoilbench submit-case cases/basic_LandremanPaulQA.yaml

run-case
--------

Run a case without creating a submission. Writes results to a custom directory (default: ``coils_runs/``). Useful for testing without creating submissions.

.. code-block:: bash

   stellcoilbench run-case <case_file> [--coils-out-dir <path>]
   stellcoilbench run-case cases/my_case.yaml --coils-out-dir my_results/

generate-submission
-------------------

Create a submission zip from existing results. Requires ``results.json`` in the results directory.

.. code-block:: bash

   stellcoilbench generate-submission <case_file> <coils_file> [--results-dir <path>]
   stellcoilbench generate-submission cases/my_case.yaml coils_runs/coils.json --results-dir coils_runs/

update-db
---------

Regenerate leaderboards from submissions. Scans ``submissions/`` and regenerates leaderboard files in ``docs/``. Run after creating submissions locally.

.. code-block:: bash

   stellcoilbench update-db

validate-config
---------------

Validate a case configuration file and report errors.

.. code-block:: bash

   stellcoilbench validate-config <case_file>
   stellcoilbench validate-config cases/my_case.yaml

post-process
------------

Run post-processing on optimized coil results. Generates Poincaré plots, QFM surfaces, VMEC equilibria, and quasisymmetry analysis.

.. code-block:: bash

   stellcoilbench post-process <coils_json> [--output-dir <path>] [--case-yaml <path>] [--plasma-surfaces-dir <path>] [--no-vmec] [--helicity-m <m>] [--helicity-n <n>] [--ns <ns>] [--no-plot-bozzer] [--no-plot-iota] [--no-plot-qs] [--no-plot-poincare] [--nfieldlines <n>]

   stellcoilbench post-process coils_runs/biot_savart_optimized.json --output-dir post_processing

Options:

- ``--output-dir``, ``-o``: Directory where post-processing results will be saved (default: ``post_processing_output``)
- ``--case-yaml``: Path to case.yaml file (if not provided, searches relative to coils JSON)
- ``--plasma-surfaces-dir``: Directory containing plasma surface files (default: ``plasma_surfaces``)
- ``--run-vmec/--no-vmec``: Whether to run VMEC equilibrium calculation (default: enabled)
- ``--helicity-m``: Poloidal mode number for quasisymmetry evaluation (default: 1)
- ``--helicity-n``: Toroidal mode number for quasisymmetry evaluation (default: 0)
- ``--ns``: Number of radial surfaces for quasisymmetry evaluation (default: 50)
- ``--plot-bozzer/--no-plot-bozzer``: Whether to generate Boozer surface plot (default: enabled)
- ``--plot-iota/--no-plot-iota``: Whether to generate iota profile plot (default: enabled)
- ``--plot-qs/--no-plot-qs``: Whether to generate quasisymmetry profile plot (default: enabled)
- ``--plot-poincare/--no-plot-poincare``: Whether to generate Poincaré plot (default: enabled)
- ``--nfieldlines``: Number of fieldlines to trace for Poincaré plot (default: 20)
