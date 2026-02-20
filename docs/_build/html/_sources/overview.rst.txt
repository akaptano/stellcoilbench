Overview
========

StellCoilBench is an open benchmark suite for stellarator coil optimization. It standardizes case definitions, runs optimizations, and maintains leaderboards so methods can be compared consistently.

Installation
------------

.. code-block:: bash

   pip install stellcoilbench

For development: ``pip install -e .``

Optional: VMEC and booz_xform enable post-processing (Poincaré plots, quasisymmetry). See the `simsopt wiki <https://github.com/hiddenSymmetries/simsopt/wiki>`_ for installation.

Quick Start
-----------

**CI workflow (fastest):** Add a case file under ``cases/`` and push. CI runs it and updates the leaderboards.

**Local run:**

.. code-block:: bash

   stellcoilbench submit-case cases/basic_LandremanPaulQA.yaml

This creates a submission in ``submissions/<surface>/<user>/<timestamp>/`` with a zip and PDF plots. Regenerate leaderboards locally with ``stellcoilbench update-db``.

Repository Layout
-----------------

- **``cases/``** — YAML case definitions (surface, coils, optimizer). See :doc:`cases`.
- **``submissions/``** — Results organized as ``submissions/<surface>/<user>/<timestamp>/all_files.zip``
- **``plasma_surfaces/``** — VMEC (``input.*``) and FOCUS (``*.focus``) surface files
- **``docs/``** — Leaderboards and generated documentation
