#!/bin/bash
# Clone stellcoilbench excluding cases/done and submissions/ via sparse checkout.
# Run from parent of desired clone location (e.g. ~ or /Users/aak572).

set -e

REPO="https://github.com/akaptano/stellcoilbench.git"
DEST="${1:-stellcoilbench_clone}"

echo "Cloning $REPO into $DEST (excluding cases/done, submissions/)..."
git clone --filter=blob:none --no-checkout --depth 1 --sparse "$REPO" "$DEST"

cd "$DEST"
git sparse-checkout set --no-cone "/*" "!/submissions/" "!/cases/done/"
git checkout main

# Tooling may expect submissions dir to exist
mkdir -p submissions

echo "Done. Verifying exclusions:"
ls -la cases/ | head -10
echo "---"
if [ -d cases/done ]; then
  echo "WARNING: cases/done still exists (sparse-checkout may need adjustment)"
else
  echo "cases/done excluded OK"
fi
if [ -d submissions ]; then
  echo "submissions/ exists (empty, contents excluded)"
fi
echo "Clone ready at $(pwd)"
