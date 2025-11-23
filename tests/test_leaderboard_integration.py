"""
Integration tests for leaderboard generation.

This test converts the functionality from test_leaderboard.sh into pytest format.
It tests the full leaderboard generation pipeline.
"""
import json
from stellcoilbench.update_db import update_database


class TestLeaderboardIntegration:
    """Integration tests for leaderboard generation."""
    
    def test_leaderboard_generation_with_submissions(self, tmp_path):
        """Test full leaderboard generation pipeline with mock submissions."""
        # Create temporary directories
        submissions_dir = tmp_path / "submissions"
        db_dir = tmp_path / "db"
        docs_dir = tmp_path / "docs"
        
        submissions_dir.mkdir(parents=True)
        db_dir.mkdir(parents=True)
        docs_dir.mkdir(parents=True)
        
        # Create a mock submission
        submission_path = submissions_dir / "LandremanPaul2021_QA" / "testuser" / "2024-01-01_12-00"
        submission_path.mkdir(parents=True)
        
        # Create results.json
        results_json = submission_path / "results.json"
        results_json.write_text(json.dumps({
            "metadata": {
                "method_name": "test_method",
                "contact": "testuser",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00"
            },
            "metrics": {
                "final_normalized_squared_flux": 0.001,
                "final_total_length": 100.0,
                "final_linking_number": 0,
                "optimization_time": 60.0
            }
        }))
        
        # Create case.yaml
        case_yaml = submission_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        
        # Run update-db
        update_database(
            repo_root=tmp_path,
            submissions_root=submissions_dir,
            db_dir=db_dir,
            docs_dir=docs_dir
        )
        
        # Verify leaderboard.json was created
        leaderboard_json = db_dir / "leaderboard.json"
        assert leaderboard_json.exists(), "leaderboard.json was not created"
        
        # Verify it's not empty
        assert leaderboard_json.stat().st_size > 0, "leaderboard.json is empty"
        
        # Validate JSON structure
        with open(leaderboard_json, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict), "leaderboard.json is not a dictionary"
        assert "entries" in data, "leaderboard.json missing 'entries' key"
        
        entries = data.get("entries", [])
        assert len(entries) > 0, "No entries in leaderboard"
        
        # Verify first entry structure
        first_entry = entries[0]
        assert "method_name" in first_entry
        assert "contact" in first_entry
        assert "score_primary" in first_entry
        assert "metrics" in first_entry
        
        # Verify metrics
        metrics = first_entry.get("metrics", {})
        assert "final_normalized_squared_flux" in metrics
        assert metrics["final_normalized_squared_flux"] == 0.001
        
        # Verify sorting (ascending order - lower is better)
        if len(entries) > 1:
            scores = [e.get("score_primary", float('inf')) for e in entries]
            assert scores == sorted(scores), "Entries are not sorted ascending by score"
    
    def test_leaderboard_generation_empty_submissions(self, tmp_path):
        """Test leaderboard generation with no submissions."""
        submissions_dir = tmp_path / "submissions"
        db_dir = tmp_path / "db"
        docs_dir = tmp_path / "docs"
        
        submissions_dir.mkdir(parents=True)
        db_dir.mkdir(parents=True)
        docs_dir.mkdir(parents=True)
        
        # Run update-db with no submissions
        update_database(
            repo_root=tmp_path,
            submissions_root=submissions_dir,
            db_dir=db_dir,
            docs_dir=docs_dir
        )
        
        # Verify leaderboard.json was created (even if empty)
        leaderboard_json = db_dir / "leaderboard.json"
        assert leaderboard_json.exists(), "leaderboard.json was not created"
        
        # Should have empty entries list
        with open(leaderboard_json, 'r') as f:
            data = json.load(f)
        
        assert "entries" in data
        assert data["entries"] == []
    
    def test_leaderboard_generation_with_coil_params(self, tmp_path):
        """Test that coil parameters are extracted from case.yaml."""
        submissions_dir = tmp_path / "submissions"
        db_dir = tmp_path / "db"
        docs_dir = tmp_path / "docs"
        
        submissions_dir.mkdir(parents=True)
        db_dir.mkdir(parents=True)
        docs_dir.mkdir(parents=True)
        
        # Create submission with case.yaml containing coil params
        submission_path = submissions_dir / "LandremanPaul2021_QA" / "testuser" / "2024-01-01_12-00"
        submission_path.mkdir(parents=True)
        
        results_json = submission_path / "results.json"
        results_json.write_text(json.dumps({
            "metadata": {
                "method_name": "test_method",
                "contact": "testuser",
                "run_date": "2024-01-01T12:00:00"
            },
            "metrics": {
                "final_normalized_squared_flux": 0.001
            }
        }))
        
        # Create case.yaml with coil parameters
        case_yaml = submission_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
  order: 8
optimizer_params:
  algorithm: l-bfgs
""")
        
        # Run update-db
        update_database(
            repo_root=tmp_path,
            submissions_root=submissions_dir,
            db_dir=db_dir,
            docs_dir=docs_dir
        )
        
        # Verify coil parameters are in metrics
        leaderboard_json = db_dir / "leaderboard.json"
        with open(leaderboard_json, 'r') as f:
            data = json.load(f)
        
        entries = data.get("entries", [])
        assert len(entries) > 0
        
        metrics = entries[0].get("metrics", {})
        assert "num_coils" in metrics
        assert "coil_order" in metrics
        assert metrics["num_coils"] == 4.0
        assert metrics["coil_order"] == 8.0
    
    def test_surface_leaderboards_generated(self, tmp_path):
        """Test that surface leaderboards are generated."""
        submissions_dir = tmp_path / "submissions"
        db_dir = tmp_path / "db"
        docs_dir = tmp_path / "docs"
        
        submissions_dir.mkdir(parents=True)
        db_dir.mkdir(parents=True)
        docs_dir.mkdir(parents=True)
        
        # Create submission
        submission_path = submissions_dir / "LandremanPaul2021_QA" / "testuser" / "2024-01-01_12-00"
        submission_path.mkdir(parents=True)
        
        results_json = submission_path / "results.json"
        results_json.write_text(json.dumps({
            "metadata": {
                "method_name": "test_method",
                "contact": "testuser",
                "run_date": "2024-01-01T12:00:00"
            },
            "metrics": {
                "final_normalized_squared_flux": 0.001
            }
        }))
        
        case_yaml = submission_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
optimizer_params:
  algorithm: l-bfgs
""")
        
        # Run update-db
        update_database(
            repo_root=tmp_path,
            submissions_root=submissions_dir,
            db_dir=db_dir,
            docs_dir=docs_dir
        )
        
        # Verify surfaces.md was created
        surfaces_md = docs_dir / "surfaces.md"
        assert surfaces_md.exists(), "surfaces.md was not created"
        
        # Verify surface leaderboard directory exists
        surfaces_dir = docs_dir / "surfaces"
        assert surfaces_dir.exists(), "docs/surfaces/ directory was not created"
        
        # Verify surface leaderboard file exists
        surface_leaderboard = surfaces_dir / "LandremanPaul2021_QA.md"
        assert surface_leaderboard.exists(), "Surface leaderboard file was not created"
        
        # Verify content includes expected elements
        content = surface_leaderboard.read_text()
        assert "Landremanpaul2021 Qa Leaderboard" in content or "LandremanPaul2021_QA" in content
        assert "testuser" in content
        assert "Legend" in content
        assert "f_B" in content  # Should have metric shorthand
    
    def test_leaderboard_sorts_ascending(self, tmp_path):
        """Test that leaderboard entries are sorted ascending by score."""
        submissions_dir = tmp_path / "submissions"
        db_dir = tmp_path / "db"
        docs_dir = tmp_path / "docs"
        
        submissions_dir.mkdir(parents=True)
        db_dir.mkdir(parents=True)
        docs_dir.mkdir(parents=True)
        
        # Create multiple submissions with different scores
        for i, score in enumerate([0.003, 0.001, 0.002]):
            submission_path = submissions_dir / "LandremanPaul2021_QA" / f"user{i}" / "2024-01-01_12-00"
            submission_path.mkdir(parents=True)
            
            results_json = submission_path / "results.json"
            results_json.write_text(json.dumps({
                "metadata": {
                    "method_name": f"method{i}",
                    "contact": f"user{i}",
                    "run_date": f"2024-01-01T12:00:0{i}"
                },
                "metrics": {
                    "final_normalized_squared_flux": score
                }
            }))
            
            case_yaml = submission_path / "case.yaml"
            case_yaml.write_text("""description: Test case
surface_params:
  surface: input.LandremanPaul2021_QA
coils_params:
  ncoils: 4
optimizer_params:
  algorithm: l-bfgs
""")
        
        # Run update-db
        update_database(
            repo_root=tmp_path,
            submissions_root=submissions_dir,
            db_dir=db_dir,
            docs_dir=docs_dir
        )
        
        # Verify sorting
        leaderboard_json = db_dir / "leaderboard.json"
        with open(leaderboard_json, 'r') as f:
            data = json.load(f)
        
        entries = data.get("entries", [])
        assert len(entries) == 3
        
        scores = [e.get("score_primary") for e in entries]
        assert scores == sorted(scores), f"Entries not sorted ascending: {scores}"
        
        # Verify ranks are assigned correctly
        for i, entry in enumerate(entries, start=1):
            assert entry["rank"] == i

