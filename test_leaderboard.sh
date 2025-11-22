#!/bin/bash
# Test script to verify leaderboard generation works correctly locally

set -e  # Exit on error

echo "=========================================="
echo "Testing StellCoilBench Leaderboard Generation"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from repository root${NC}"
    exit 1
fi

# Check if conda environment is activated, if not try to activate simsopt_39
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}No conda environment detected. Attempting to activate simsopt_39...${NC}"
    # Try to initialize conda if not already initialized
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/opt/miniconda3/etc/profile.d/conda.sh"
    fi
    # Activate the environment
    if command -v conda &> /dev/null; then
        conda activate simsopt_39 2>/dev/null || {
            echo -e "${YELLOW}Could not activate conda. Will try using 'conda run' instead${NC}"
            USE_CONDA_RUN=true
        }
    else
        USE_CONDA_RUN=true
    fi
else
    USE_CONDA_RUN=false
fi

# Function to run commands with conda if needed
run_cmd() {
    if [ "$USE_CONDA_RUN" = true ]; then
        conda run -n simsopt_39 --no-capture-output "$@"
    else
        "$@"
    fi
}

echo "Step 1: Checking existing submissions..."
SUBMISSION_COUNT=$(find submissions -name "results.json" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "Found $SUBMISSION_COUNT existing submission(s)"
echo ""

# Check if leaderboard.json exists and show its contents
if [ -f "db/leaderboard.json" ]; then
    echo "Current leaderboard.json structure:"
    python3 -c "import json; data = json.load(open('db/leaderboard.json')); print(f\"  Entries: {len(data.get('entries', []))}\"); print(f\"  Cases: {len(data.get('cases', {}))}\")"
    echo ""
fi

echo "Step 2: Running update-db to regenerate leaderboard..."
if [ "$USE_CONDA_RUN" = true ]; then
    conda run -n stellcoilbench --no-capture-output stellcoilbench update-db --db-dir db --docs-dir docs || {
        echo -e "${RED}ERROR: update-db failed!${NC}"
        exit 1
    }
else
    stellcoilbench update-db --db-dir db --docs-dir docs || {
        echo -e "${RED}ERROR: update-db failed!${NC}"
        exit 1
    }
fi
echo ""

echo "Step 3: Verifying leaderboard.json was created..."
if [ ! -f "db/leaderboard.json" ]; then
    echo -e "${RED}ERROR: leaderboard.json was not created!${NC}"
    exit 1
fi

# Check if file is empty or invalid JSON
FILE_SIZE=$(stat -f%z "db/leaderboard.json" 2>/dev/null || stat -c%s "db/leaderboard.json" 2>/dev/null)
if [ "$FILE_SIZE" -eq 0 ]; then
    echo -e "${RED}ERROR: leaderboard.json is empty!${NC}"
    exit 1
fi

# Validate JSON structure
python3 << 'PYTHON_SCRIPT'
import json
import sys

try:
    with open('db/leaderboard.json', 'r') as f:
        data = json.load(f)
    
    # Check structure
    if not isinstance(data, dict):
        print("ERROR: leaderboard.json is not a dictionary!")
        sys.exit(1)
    
    if "entries" not in data:
        print("ERROR: leaderboard.json missing 'entries' key!")
        sys.exit(1)
    
    entries = data.get("entries", [])
    
    print(f"✓ leaderboard.json structure is valid")
    print(f"  - Entries: {len(entries)}")
    
    if len(entries) == 0:
        print("\n⚠ WARNING: No entries in leaderboard!")
        print("  This might be because:")
        print("    1. All cases are dev_ or test_ cases (filtered out)")
        print("    2. No valid submissions found")
        print("    3. Submissions don't have score_primary or final_flux")
    else:
        print("\n✓ Leaderboard has entries!")
        # Show first entry details
        first_entry = entries[0]
        print(f"\n  First entry:")
        print(f"    Method: {first_entry.get('method_name', 'N/A')}")
        print(f"    Contact: {first_entry.get('contact', 'N/A')}")
        print(f"    Score: {first_entry.get('score_primary', 'N/A')}")
        
        # Show metrics from first entry
        metrics = first_entry.get('metrics', {})
        if metrics:
            print(f"\n  First entry metrics:")
            metric_keys = sorted([k for k in metrics.keys() if k != 'score_primary'])[:10]
            for key in metric_keys:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value}")
    
    
except json.JSONDecodeError as e:
    print(f"ERROR: leaderboard.json is not valid JSON: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: leaderboard.json validation failed!${NC}"
    exit 1
fi

echo ""
echo "Step 4: Checking markdown files..."
if [ -f "docs/leaderboard.md" ]; then
    echo -e "${GREEN}✓ docs/leaderboard.md exists${NC}"
    MD_SIZE=$(stat -f%z "docs/leaderboard.md" 2>/dev/null || stat -c%s "docs/leaderboard.md" 2>/dev/null)
    echo "  Size: $MD_SIZE bytes"
else
    echo -e "${YELLOW}⚠ docs/leaderboard.md not found${NC}"
fi

if [ -d "docs/leaderboards" ]; then
    CASE_FILES=$(find docs/leaderboards -name "*.md" -type f | wc -l | tr -d ' ')
    echo -e "${GREEN}✓ docs/leaderboards/ directory exists with $CASE_FILES file(s)${NC}"
fi

echo ""
echo "Step 5: Analyzing submission metrics..."
python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

submissions_dir = Path("submissions")
results_files = list(submissions_dir.rglob("results.json"))

print(f"Found {len(results_files)} results.json file(s):")
print()

for results_file in results_files[:5]:  # Show first 5
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        meta = data.get("metadata", {})
        metrics = data.get("metrics", {})
        cases = data.get("cases", [])  # Old format support
        
        print(f"  {results_file.relative_to(submissions_dir.parent)}")
        print(f"    Contact: {meta.get('contact', 'N/A')}")
        
        # Check for new format (metrics directly) or old format (cases array)
        if metrics:
            print(f"    Format: New (metrics directly)")
            print(f"    Metrics: {len(metrics)} fields")
            
            # Check for score_primary
            has_score_primary = "score_primary" in metrics
            has_final_flux = "final_flux" in metrics
            has_final_normalized_squared_flux = "final_normalized_squared_flux" in metrics
            
            print(f"      Has score_primary: {has_score_primary}")
            print(f"      Has final_flux: {has_final_flux}")
            print(f"      Has final_normalized_squared_flux: {has_final_normalized_squared_flux}")
            
            if has_final_normalized_squared_flux and not has_final_flux:
                print(f"      ⚠ NOTE: Has final_normalized_squared_flux but not final_flux")
                print(f"         The code looks for 'final_flux' or 'final_normalized_squared_flux' as fallback")
            
            # Count total metrics
            numeric_metrics = sum(1 for v in metrics.values() if isinstance(v, (int, float)))
            print(f"      Total numeric metrics: {numeric_metrics}")
        elif cases:
            print(f"    Format: Old (cases array) - {len(cases)} case(s)")
            print(f"    ⚠ NOTE: Old format detected. New submissions use 'metrics' directly.")
            for case in cases:
                case_id = case.get("case_id", "N/A")
                case_metrics = case.get("metrics", {})
                print(f"      Case: {case_id}, Metrics: {len(case_metrics)}")
        else:
            print(f"    ⚠ WARNING: No metrics or cases found!")
        
        # Check for VTK files in submission directory
        submission_dir = results_file.parent
        vtu_files = list(submission_dir.glob("*.vtu"))
        vts_files = list(submission_dir.glob("*.vts"))
        coils_json = submission_dir / "coils.json"
        
        print(f"    Files in submission directory:")
        print(f"      coils.json: {'✓' if coils_json.exists() else '✗'}")
        print(f"      .vtu files: {len(vtu_files)}")
        print(f"      .vts files: {len(vts_files)}")
        if vtu_files:
            print(f"        VTU files: {', '.join([f.name for f in vtu_files[:3]])}")
        if vts_files:
            print(f"        VTS files: {', '.join([f.name for f in vts_files[:3]])}")
        
        print()
    except Exception as e:
        print(f"    ERROR reading {results_file}: {e}")
        import traceback
        traceback.print_exc()
        print()
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo -e "${GREEN}Test completed!${NC}"
echo "=========================================="
echo ""
echo "To view the leaderboard:"
echo "  - JSON: cat db/leaderboard.json | python3 -m json.tool"
echo "  - Markdown: cat docs/leaderboard.md"
echo ""
echo "To test with a new submission:"
echo "  1. Create a test case (NOT starting with dev_ or test_):"
echo "     cp cases/case.yaml cases/test_production.yaml"
echo "     # Edit test_production.yaml and change case_id to something like 'prod_test_001'"
echo "  2. stellcoilbench submit-case cases/test_production.yaml"
echo "  3. stellcoilbench update-db"
echo "  4. Check db/leaderboard.json and docs/leaderboard.md"
echo ""
echo "NOTE: Cases starting with 'dev_' or 'test_' are filtered out of the leaderboard."
echo "      Use a case_id that doesn't start with these prefixes for testing."
echo ""

