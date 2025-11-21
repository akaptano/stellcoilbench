"""
CoilBench: benchmarking framework for stellarator coil optimization algorithms.

This package provides:

- Data structures for benchmark cases and submissions.
- An evaluation pipeline that takes (case_dir, coils_path) pairs and produces
  metrics + scores.
- Command-line tools (via `coilbench` entrypoint) for running evaluations and
  building leaderboards.

The heavy physics-specific details (Biot–Savart, coil geometry metrics) live in
`biotsavart.py` and `geometry.py` and are intentionally thin wrappers so you
can integrate whichever codes you want.
"""

__all__ = [
    "evaluate",
    "metrics",
    "geometry",
    "biotsavart",
]

