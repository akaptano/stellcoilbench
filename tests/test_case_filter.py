#!/usr/bin/env python3
"""Test script to verify case file filtering logic works correctly."""

import sys
import yaml
import zipfile
import tempfile
from pathlib import Path
import json

def normalize_yaml_content(content):
    """Normalize YAML content for comparison (remove comments, normalize whitespace)."""
    try:
        data = yaml.safe_load(content)
        return yaml.dump(data, sort_keys=True, default_flow_style=False)
    except Exception as e:
        print(f"Warning: Failed to normalize YAML content: {e}", file=sys.stderr)
        return content

def case_has_successful_submission(case_file_path, submissions_dir=None):
    """Check if a case file has at least one successful submission with matching content."""
    if submissions_dir is None:
        submissions_dir = Path("submissions")
    
    case_path = Path(case_file_path)
    if not case_path.exists():
        return False
    
    # Read current case file content
    try:
        current_case_content = case_path.read_text()
        current_case_normalized = normalize_yaml_content(current_case_content)
    except Exception as e:
        print(f"Warning: Failed to read case file {case_file_path}: {e}", file=sys.stderr)
        return False
    
    # Get relative path from repo root
    try:
        repo_root = Path.cwd()
        case_file_rel = str(case_path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        case_file_rel = str(case_path.resolve())
    
    case_file_rel_normalized = case_file_rel.replace("\\", "/")
    
    if not submissions_dir.exists():
        return False
    
    zip_files = list(submissions_dir.rglob("*.zip"))
    if len(zip_files) == 0:
        return False  # No submissions exist
    
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                if 'case.yaml' not in zf.namelist() or 'results.json' not in zf.namelist():
                    continue
                
                zip_case_content = zf.read('case.yaml').decode('utf-8')
                zip_case_data = yaml.safe_load(zip_case_content)
                
                # Check if this submission is for this case file
                submission_source = zip_case_data.get('source_case_file', '')
                if submission_source:
                    submission_source_normalized = submission_source.replace("\\", "/")
                    if (submission_source == case_file_rel or 
                        submission_source_normalized == case_file_rel_normalized):
                        # Found a submission for this case file - now check if content matches
                        zip_case_normalized = normalize_yaml_content(zip_case_content)
                        if zip_case_normalized == current_case_normalized:
                            return True  # Content matches, submission is still valid
                        else:
                            # Content differs - case file has changed, should rerun
                            print(f"Case file changed: {case_file_path} (submission from {zip_path.name})", file=sys.stderr)
                            continue  # Keep looking, but this submission is outdated
        except Exception as e:
            print(f"Warning: Failed to process {zip_path}: {e}", file=sys.stderr)
            continue
    
    return False  # No matching submission found

def main():
    """Test the case filtering logic."""
    print("=" * 60)
    print("Testing Case File Filtering Logic")
    print("=" * 60)
    print()
    
    # Test 1: No submissions exist - all cases should run
    print("Test 1: No submissions exist")
    print("-" * 60)
    case_files = list(Path("cases").glob("*.yaml"))
    print(f"Found {len(case_files)} case files")
    
    submissions_dir = Path("submissions")
    zip_count = len(list(submissions_dir.rglob("*.zip"))) if submissions_dir.exists() else 0
    print(f"Found {zip_count} submission zip files")
    print()
    
    cases_to_run = []
    cases_already_successful = []
    
    for case_file in sorted(case_files):
        if case_has_successful_submission(str(case_file)):
            cases_already_successful.append(str(case_file))
            print(f"SKIP: {case_file}")
        else:
            cases_to_run.append(str(case_file))
            print(f"RUN:  {case_file}")
    
    print()
    print("Summary:")
    print(f"  Cases to run: {len(cases_to_run)}")
    print(f"  Cases with successful submissions (skipped): {len(cases_already_successful)}")
    print()
    
    # Test 2: Create a mock submission and verify it's detected
    print("=" * 60)
    print("Test 2: Create mock submission and verify detection")
    print("-" * 60)
    
    if len(case_files) > 0:
        test_case = case_files[0]
        print(f"Testing with case file: {test_case}")
        
        # Create a temporary mock submission
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_submissions = Path(tmpdir) / "submissions" / "test_surface" / "test_user"
            mock_submissions.mkdir(parents=True)
            
            # Read the case file
            case_content = test_case.read_text()
            case_data = yaml.safe_load(case_content)
            
            # Add source_case_file to the case data
            case_data['source_case_file'] = str(test_case.relative_to(Path.cwd()))
            
            # Create a mock zip file
            zip_path = mock_submissions / "test_submission.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                # Write case.yaml with source_case_file
                zf.writestr('case.yaml', yaml.dump(case_data))
                # Write a mock results.json
                zf.writestr('results.json', json.dumps({
                    "metadata": {"method_name": "test"},
                    "metrics": {"final_normalized_squared_flux": 0.001}
                }))
            
            print(f"Created mock submission: {zip_path}")
            
            # Test if the case is detected as having a successful submission
            has_submission = case_has_successful_submission(str(test_case), submissions_dir=Path(tmpdir) / "submissions")
            print(f"Case has successful submission: {has_submission}")
            
            if has_submission:
                print("✓ Test 2 PASSED: Mock submission correctly detected")
            else:
                print("✗ Test 2 FAILED: Mock submission not detected")
                return 1
    
    print()
    print("=" * 60)
    print("All tests completed")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

