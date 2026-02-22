"""
StellCoilBench: open benchmark suite for stellarator coil optimization algorithms.

This package provides standardized case definitions (YAML), automated optimization
via simsopt, post-processing (VMEC, Poincaré plots, quasisymmetry), and CI-driven
leaderboard generation.

Primary workflow
----------------
1. Define or add a case under ``cases/``
2. Run locally via ``stellcoilbench submit-case`` or push to run on CI
3. CI aggregates results into ``docs/leaderboards/``

Key modules
-----------
- cli : Typer CLI entry point (submit-case, run-case, update-db, etc.)
- coil_optimization : Core optimization loop (scipy, augmented Lagrangian)
- post_processing : VMEC, Poincaré, quasisymmetry, VTK export
- update_db : Leaderboard generation from submissions
- validate_config : Case YAML and CI case JSON validation
"""

__all__ = [
    "cli",
    "coil_optimization",
    "config_scheme",
    "evaluate",
    "finite_build",
    "mpi_utils",
    "path_utils",
    "post_processing",
    "update_db",
    "validate_config",
]
