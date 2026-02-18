#!/usr/bin/env python3
"""
Desktop KB updater: pull latest from git and ingest run summaries into local KB.

Usage:
    cd /path/to/stellcoilbench
    python knowledge/scripts/kb_updater.py

Or with explicit paths:
    python knowledge/scripts/kb_updater.py --repo-dir . --done-dir cases/done

Requires: KB server running (docker compose up) and ingest deps.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add repo root for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd)",
    )
    parser.add_argument(
        "--done-dir",
        type=Path,
        default=None,
        help="Path to cases/done (default: repo-dir/cases/done)",
    )
    parser.add_argument(
        "--kb-url",
        type=str,
        default="http://localhost:8000",
        help="KB server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip git pull",
    )
    args = parser.parse_args()

    repo = args.repo_dir.resolve()
    done_dir = args.done_dir or (repo / "cases" / "done")

    if not repo.is_dir():
        print(f"ERROR: repo dir not found: {repo}", file=sys.stderr)
        return 1

    # Optional: git pull
    if not args.no_pull:
        try:
            subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as e:
            print(f"WARNING: git pull failed: {e}", file=sys.stderr)

    if not done_dir.is_dir():
        print(f"No done dir: {done_dir}", file=sys.stderr)
        return 0

    # Ingest each summary
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        pass

    count = 0
    for summary_path in sorted(done_dir.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        case_id = summary.get("case_id", "")
        if not case_id:
            continue

        url = f"{args.kb_url.rstrip('/')}/ingest/run"
        body = json.dumps(summary).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status in (200, 201):
                    count += 1
        except Exception as e:
            print(f"WARNING: ingest {case_id} failed: {e}", file=sys.stderr)

    print(f"Ingested {count} run(s) into KB at {args.kb_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
