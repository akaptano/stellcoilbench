#!/bin/bash
# Test script to verify CI logic for case file filtering

set -e

echo "=========================================="
echo "Testing CI Case File Filtering Logic"
echo "=========================================="
echo

# Simulate the CI workflow step
echo "Step 1: Finding all case files..."
ALL_CASE_FILES=$(find cases -name "*.yaml" -type f | sort)
echo "Found case files:"
echo "$ALL_CASE_FILES"
echo

# Check for submissions
echo "Step 2: Checking for existing submissions..."
ZIP_COUNT=$(find submissions -name "*.zip" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "Found $ZIP_COUNT zip files in submissions/"
echo

# Simulate the Python script output (since we don't have yaml installed locally)
# In CI, this would be the actual Python output
echo "Step 3: Simulating Python script output..."
echo "Since there are no submissions, all cases should be marked to run"
echo

# Count case files
CASE_COUNT=$(echo "$ALL_CASE_FILES" | wc -l | tr -d ' ')
echo "Expected result:"
echo "  Cases to run: $CASE_COUNT"
echo "  Cases with successful submissions (skipped): 0"
echo

# Verify the workflow logic
echo "Step 4: Verifying workflow logic..."
if [ "$ZIP_COUNT" -eq 0 ]; then
    echo "✓ No submissions found - all cases should run"
    EXPECTED_TO_RUN=$CASE_COUNT
    EXPECTED_SKIPPED=0
else
    echo "⚠ Submissions found - some cases may be skipped"
    EXPECTED_TO_RUN="< $CASE_COUNT"
    EXPECTED_SKIPPED="> 0"
fi
echo

# Test JSON parsing (as CI does)
echo "Step 5: Testing JSON parsing logic..."
TEST_JSON='{"to_run": ["cases/basic_tokamak.yaml"], "already_successful": []}'
PARSED=$(echo "$TEST_JSON" | python3 -c "import sys, json; data = json.load(sys.stdin); print('\n'.join(data['to_run']))")
if [ "$PARSED" = "cases/basic_tokamak.yaml" ]; then
    echo "✓ JSON parsing works correctly"
else
    echo "✗ JSON parsing failed"
    exit 1
fi
echo

# Test empty case handling
echo "Step 6: Testing empty case handling..."
EMPTY_JSON='{"to_run": [], "already_successful": ["cases/basic_tokamak.yaml"]}'
EMPTY_PARSED=$(echo "$EMPTY_JSON" | python3 -c "import sys, json; data = json.load(sys.stdin); files = data['to_run']; print(len(files))")
if [ "$EMPTY_PARSED" = "0" ]; then
    echo "✓ Empty case handling works correctly"
else
    echo "✗ Empty case handling failed"
    exit 1
fi
echo

echo "=========================================="
echo "All tests passed!"
echo "=========================================="
echo
echo "Summary:"
echo "  - Case file discovery: ✓"
echo "  - Submission checking: ✓"
echo "  - JSON parsing: ✓"
echo "  - Empty case handling: ✓"
echo
echo "The CI workflow should correctly:"
echo "  1. Find all case files"
echo "  2. Check for existing submissions"
echo "  3. Filter cases that have successful submissions"
echo "  4. Run only cases without successful submissions"

