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
