#!/usr/bin/env python3
"""
Generate a deterministic postmortem from a failed CI run summary.

Rule-based suggestions for what to try next. LLM can refine later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def make_postmortem(summary: dict) -> str:
    """Generate postmortem text for a failed run (or empty for success)."""
    if summary.get("success", True):
        return ""

    lines: list[str] = []
    fc = summary.get("failure_class", "unknown")
    fr = summary.get("failure_reason", "")[:200]

    lines.append(f"Failure class: {fc}")
    lines.append(f"Reason: {fr}")

    suggestions: list[str] = []
    if fc == "min_sep_violation":
        suggestions.extend([
            "Suggest: increase coil-coil or coil-surface separation penalty",
            "Suggest: reduce step scale or trust region",
            "Suggest: consider restart from different initial coils",
        ])
    elif fc == "line_search_fail":
        suggestions.extend([
            "Suggest: reduce trust region or step scale",
            "Suggest: try different line search (e.g. Armijo vs Wolfe)",
            "Suggest: reduce max_iter_subopt or restart",
        ])
    elif fc == "timeout":
        suggestions.extend([
            "Suggest: increase timeout_minutes or reduce max_iterations",
            "Suggest: use simpler surface or lower Fourier order for faster eval",
            "Suggest: reduce surface_resolution for speed",
        ])
    elif fc == "nan_in_objective":
        suggestions.extend([
            "Suggest: check for degenerate geometry or bad initial coils",
            "Suggest: reduce step scale, add regularization",
            "Suggest: try different random_seed",
        ])
    elif fc == "vmec_nonconverged":
        suggestions.extend([
            "Suggest: VMEC may need different ns or convergence params",
            "Suggest: check if coils produce reasonable field",
            "Suggest: run without VMEC first to verify optimization",
        ])
    elif fc == "validation":
        suggestions.extend([
            "Suggest: fix case config to pass validation",
            "Suggest: check resource caps and policy limits",
        ])
    else:
        suggestions.append("Suggest: inspect logs and adjust config or optimizer")

    if suggestions:
        lines.append("")
        lines.extend(suggestions)

    margins = summary.get("margins", {})
    if margins:
        negative = [(k, v) for k, v in margins.items() if isinstance(v, (int, float)) and v < 0]
        if negative:
            lines.append("")
            lines.append("Negative margins (constraint violations):")
            for k, v in negative:
                lines.append(f"  {k}: {v:.4e}")

    return "\n".join(lines)


def main() -> int:
    """Read summary from stdin or file, print postmortem to stdout."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        summary = json.loads(path.read_text())
    else:
        summary = json.load(sys.stdin)
    out = make_postmortem(summary)
    if out:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
