#!/usr/bin/env python3
"""
Generate a short text "run card" from a CI run summary for embedding and semantic search.

Turns summary.json into a 10–20 line human-readable card suitable for vector search.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def make_run_card(summary: dict) -> str:
    """Generate a run card string from a summary dict."""
    lines: list[str] = []
    cid = summary.get("case_id", "?")
    success = summary.get("success", False)
    score = summary.get("total_score", float("inf"))
    iters = summary.get("iterations_used", 0)
    wall = summary.get("walltime_sec", 0)
    tags = summary.get("tags", [])
    parents = summary.get("parent_ids", [])

    status = "SUCCESS" if success else "FAILED"
    lines.append(f"Run {cid}: {status}")
    lines.append(f"Score: {score:.4e} | Iterations: {iters} | Walltime: {wall:.0f}s")
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")
    if parents:
        lines.append(f"Parents: {', '.join(parents[:3])}{'...' if len(parents) > 3 else ''}")

    cfg = summary.get("case_config", {})
    sp = cfg.get("surface_params", {})
    surface = sp.get("surface", "?") if isinstance(sp, dict) else "?"
    cp = cfg.get("coils_params", {})
    ncoils = cp.get("ncoils", "?") if isinstance(cp, dict) else "?"
    order = cp.get("order", "?") if isinstance(cp, dict) else "?"
    lines.append(f"Surface: {surface} | ncoils={ncoils} order={order}")

    metrics = summary.get("metrics", {})
    if metrics:
        if "final_min_cc_separation" in metrics:
            lines.append(f"CC separation: {metrics['final_min_cc_separation']:.4f}")
        if "final_min_cs_separation" in metrics:
            lines.append(f"CS separation: {metrics['final_min_cs_separation']:.4f}")
        if "final_max_curvature" in metrics:
            lines.append(f"Max curvature: {metrics['final_max_curvature']:.4f}")
        if "BdotN_over_B" in metrics:
            lines.append(f"B·n/B: {metrics['BdotN_over_B']:.4e}")

    margins = summary.get("margins", {})
    if margins:
        tight = [k for k, v in margins.items() if isinstance(v, (int, float)) and v < 0.1]
        if tight:
            lines.append(f"Tight margins: {', '.join(tight)}")

    if not success:
        fc = summary.get("failure_class", "")
        fr = summary.get("failure_reason", "")[:80]
        lines.append(f"Failure: {fc} — {fr}")

    return "\n".join(lines)


def main() -> int:
    """Read summary from stdin or file, print run card to stdout."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        summary = json.loads(path.read_text())
    else:
        summary = json.load(sys.stdin)
    print(make_run_card(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
