#!/bin/bash
# Safe git pull that handles untracked files in docs/leaderboards/

# Remove untracked files in docs/leaderboards/ that would conflict
if [ -d "docs/leaderboards" ]; then
    echo "Cleaning untracked files in docs/leaderboards/..."
    find docs/leaderboards -name "*.md" -type f ! -path "*/\.git/*" | while read file; do
        if ! git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
            echo "Removing untracked file: $file"
            rm -f "$file"
        fi
    done
    git clean -fd docs/leaderboards/ 2>/dev/null || true
fi

# Remove untracked leaderboard.json if it exists
if [ -f "docs/leaderboard.json" ] && ! git ls-files --error-unmatch docs/leaderboard.json >/dev/null 2>&1; then
    echo "Removing untracked file: docs/leaderboard.json"
    rm -f docs/leaderboard.json
fi

# Now do the pull
git pull "$@"
