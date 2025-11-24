#!/bin/bash
# Safe git pull that handles untracked files in docs/ and submissions/
# Usage: ./scripts/git-pull-safe.sh

set -e

echo "Cleaning untracked files in docs/ and submissions/..."
git clean -fd docs/ submissions/ 2>/dev/null || true

echo "Pulling from remote..."
git pull --no-ff -X theirs

echo "Done!"

