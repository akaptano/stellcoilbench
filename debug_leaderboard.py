#!/usr/bin/env python3
"""
Debug script to check why rotating_ellipse.md isn't being generated.

Checks:
1. Zip files exist and are valid
2. Zip files contain results.json and case.yaml
3. results.json has valid metrics
4. Entries are loaded into leaderboard.json
5. Surface names are extracted correctly
6. Surface leaderboards are generated
"""

import json
import zipfile
from pathlib import Path

def check_zip_files():
    """Check if zip files exist and are valid."""
    print("=" * 60)
    print("1. CHECKING ZIP FILES")
    print("=" * 60)
    
    submissions_dir = Path("submissions")
    if not submissions_dir.exists():
        print("ERROR: submissions/ directory does not exist!")
        return []
    
    zip_files = list(submissions_dir.rglob("*.zip"))
    print(f"Found {len(zip_files)} zip files total")
    
    rotating_zips = [z for z in zip_files if 'rotating' in str(z).lower()]
    print(f"Found {len(rotating_zips)} rotating_ellipse zip files")
    
    valid_zips = []
    invalid_zips = []
    
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                has_results = 'results.json' in zf.namelist()
                has_case = 'case.yaml' in zf.namelist()
                
                if has_results and has_case:
                    # Check if results.json has metrics
                    try:
                        results_content = zf.read('results.json').decode('utf-8')
                        results_data = json.loads(results_content)
                        has_metrics = bool(results_data.get('metrics'))
                        
                        if has_metrics:
                            valid_zips.append(zip_path)
                            if 'rotating' in str(zip_path).lower():
                                print(f"  ✓ Valid: {zip_path.name}")
                                metrics = results_data.get('metrics', {})
                                print(f"    Metrics: {list(metrics.keys())[:5]}")
                        else:
                            invalid_zips.append((zip_path, "no metrics"))
                            if 'rotating' in str(zip_path).lower():
                                print(f"  ✗ Invalid (no metrics): {zip_path.name}")
                    except Exception as e:
                        invalid_zips.append((zip_path, f"json error: {e}"))
                        if 'rotating' in str(zip_path).lower():
                            print(f"  ✗ Invalid (json error): {zip_path.name} - {e}")
                else:
                    missing = []
                    if not has_results:
                        missing.append("results.json")
                    if not has_case:
                        missing.append("case.yaml")
                    invalid_zips.append((zip_path, f"missing: {', '.join(missing)}"))
                    if 'rotating' in str(zip_path).lower():
                        print(f"  ✗ Invalid (missing {', '.join(missing)}): {zip_path.name}")
        except Exception as e:
            invalid_zips.append((zip_path, f"zip error: {e}"))
            if 'rotating' in str(zip_path).lower():
                print(f"  ✗ Invalid (zip error): {zip_path.name} - {e}")
    
    print(f"\nSummary: {len(valid_zips)} valid, {len(invalid_zips)} invalid")
    return valid_zips


def check_leaderboard_json():
    """Check what's in leaderboard.json."""
    print("\n" + "=" * 60)
    print("2. CHECKING LEADERBOARD.JSON")
    print("=" * 60)
    
    leaderboard_file = Path("docs/leaderboard.json")
    if not leaderboard_file.exists():
        print("ERROR: docs/leaderboard.json does not exist!")
        return None
    
    try:
        data = json.loads(leaderboard_file.read_text())
        entries = data.get('entries', [])
        print(f"Total entries: {len(entries)}")
        
        # Group by surface
        surfaces = {}
        rotating_entries = []
        
        for entry in entries:
            path = entry.get('path', '')
            if 'submissions/' in path:
                try:
                    parts = Path(path).parts
                    if 'submissions' in parts:
                        idx = parts.index('submissions')
                        if idx + 1 < len(parts):
                            surface = parts[idx + 1]
                            if surface not in surfaces:
                                surfaces[surface] = []
                            surfaces[surface].append(entry)
                            
                            if 'rotating' in surface.lower():
                                rotating_entries.append(entry)
                except Exception as e:
                    print(f"  Warning: Failed to parse path {path}: {e}")
        
        print(f"\nSurfaces found: {sorted(surfaces.keys())}")
        for surface, entries_list in sorted(surfaces.items()):
            print(f"  {surface}: {len(entries_list)} entries")
        
        print(f"\nRotating ellipse entries: {len(rotating_entries)}")
        if rotating_entries:
            print("Sample paths:")
            for e in rotating_entries[:3]:
                print(f"  {e.get('path')}")
                print(f"    Score: {e.get('score_primary', 'N/A')}")
        else:
            print("  ⚠️  No rotating_ellipse entries found!")
        
        return data
    except Exception as e:
        print(f"ERROR: Failed to read leaderboard.json: {e}")
        return None


def check_surface_leaderboards():
    """Check what surface leaderboard files exist."""
    print("\n" + "=" * 60)
    print("3. CHECKING SURFACE LEADERBOARD FILES")
    print("=" * 60)
    
    leaderboards_dir = Path("docs/leaderboards")
    if not leaderboards_dir.exists():
        print("ERROR: docs/leaderboards/ directory does not exist!")
        return []
    
    md_files = list(leaderboards_dir.glob("*.md"))
    print(f"Found {len(md_files)} .md files:")
    for f in sorted(md_files):
        size = f.stat().st_size
        mtime = f.stat().st_mtime
        print(f"  {f.name} ({size} bytes, modified: {mtime})")
    
    rotating_md = leaderboards_dir / "rotating_ellipse.md"
    if rotating_md.exists():
        print("\n✓ rotating_ellipse.md exists")
        content = rotating_md.read_text()
        print(f"  Size: {len(content)} bytes")
        print(f"  First 200 chars: {content[:200]}...")
    else:
        print("\n✗ rotating_ellipse.md does NOT exist")
    
    return md_files


def check_surface_extraction():
    """Test surface name extraction logic."""
    print("\n" + "=" * 60)
    print("4. TESTING SURFACE NAME EXTRACTION")
    print("=" * 60)
    
    test_paths = [
        "submissions/rotating_ellipse/akaptano/11-28-2025_23-13.zip",
        "submissions/LandremanPaul2021_QA/akaptano/11-28-2025_23-13.zip",
        "submissions/circular_tokamak/akaptano/11-28-2025_23-13.zip",
    ]
    
    for path_str in test_paths:
        print(f"\nTesting: {path_str}")
        try:
            parts = Path(path_str).parts
            print(f"  Parts: {parts}")
            
            if 'submissions' in parts:
                idx = parts.index('submissions')
                print(f"  submissions index: {idx}")
                
                if len(parts) > idx + 1:
                    surface = parts[idx + 1]
                    print(f"  ✓ Extracted surface: {surface}")
                    
                    # Test filename generation
                    safe_filename = surface.replace(".", "_")
                    print(f"  Safe filename: {safe_filename}.md")
                else:
                    print("  ✗ Not enough parts after 'submissions'")
            else:
                print("  ✗ 'submissions' not found in path")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def simulate_update_db():
    """Simulate what update_db does."""
    print("\n" + "=" * 60)
    print("5. SIMULATING UPDATE_DB PROCESS")
    print("=" * 60)
    
    try:
        from stellcoilbench.update_db import _load_submissions, build_methods_json, build_leaderboard_json, build_surface_leaderboards
        
        submissions_dir = Path("submissions")
        repo_root = Path.cwd()
        
        print("Loading submissions...")
        submissions = list(_load_submissions(submissions_dir))
        print(f"Loaded {len(submissions)} submissions")
        
        rotating_submissions = [s for s in submissions if 'rotating' in str(s[1]).lower()]
        print(f"Rotating ellipse submissions: {len(rotating_submissions)}")
        for method_key, path, data in rotating_submissions[:3]:
            print(f"  {method_key}: {path}")
            print(f"    Has metrics: {bool(data.get('metrics'))}")
        
        print("\nBuilding methods.json...")
        methods = build_methods_json(submissions_dir, repo_root)
        print(f"Methods: {len(methods)}")
        
        rotating_methods = {k: v for k, v in methods.items() if 'rotating' in str(v.get('path', '')).lower()}
        print(f"Rotating ellipse methods: {len(rotating_methods)}")
        
        print("\nBuilding leaderboard.json...")
        leaderboard = build_leaderboard_json(methods)
        entries = leaderboard.get('entries', [])
        print(f"Leaderboard entries: {len(entries)}")
        
        rotating_entries = [e for e in entries if 'rotating' in e.get('path', '').lower()]
        print(f"Rotating ellipse entries: {len(rotating_entries)}")
        
        print("\nBuilding surface leaderboards...")
        surface_leaderboards = build_surface_leaderboards(leaderboard, submissions_dir, Path("plasma_surfaces"))
        print(f"Surface leaderboards: {sorted(surface_leaderboards.keys())}")
        
        if 'rotating_ellipse' in surface_leaderboards:
            entries = surface_leaderboards['rotating_ellipse'].get('entries', [])
            print(f"  rotating_ellipse: {len(entries)} entries")
        else:
            print("  ✗ rotating_ellipse NOT in surface leaderboards!")
    except ImportError:
        print("⚠️  stellcoilbench module not available (this is OK if running outside CI)")
    except Exception as e:
        print(f"⚠️  Error simulating update_db: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("DEBUGGING LEADERBOARD GENERATION")
    print("=" * 60)
    
    # Check zip files
    valid_zips = check_zip_files()
    
    # Check leaderboard.json
    _ = check_leaderboard_json()
    
    # Check surface leaderboard files
    _ = check_surface_leaderboards()
    
    # Test surface extraction
    check_surface_extraction()
    
    # Simulate update_db
    try:
        simulate_update_db()
    except Exception as e:
        print(f"\n⚠️  Could not simulate update_db: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    rotating_zips = [z for z in valid_zips if 'rotating' in str(z).lower()]
    rotating_md = Path("docs/leaderboards/rotating_ellipse.md")
    
    print(f"Valid rotating_ellipse zip files: {len(rotating_zips)}")
    print(f"rotating_ellipse.md exists: {rotating_md.exists()}")
    
    if rotating_zips and not rotating_md.exists():
        print("\n⚠️  ISSUE DETECTED: Zip files exist but .md file not generated!")
        print("   This suggests an issue with surface leaderboard generation.")
    elif not rotating_zips:
        print("\n⚠️  ISSUE DETECTED: No valid rotating_ellipse zip files found!")
        print("   Check if submissions were generated correctly.")


if __name__ == "__main__":
    main()

