"""
StellCoilBench: benchmarking framework for stellarator coil optimization algorithms.

Primary workflow:
- Define or add a case under `cases/`
- Run locally via `stellcoilbench submit-case` or push to run on CI
- CI aggregates results into `docs/leaderboards/`
"""

__all__ = [
    "cli",
    "coil_optimization",
    "evaluate",
    "update_db",
    "validate_config",
    "config_scheme",
]
