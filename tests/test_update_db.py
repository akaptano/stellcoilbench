"""
Unit tests for update_db.py
"""
import tempfile
import json
from pathlib import Path
from stellcoilbench.update_db import (
    _metric_shorthand,
    _metric_definition,
    _load_submissions,
    build_methods_json,
    build_leaderboard_json,
)


class TestMetricShorthand:
    """Tests for _metric_shorthand function."""
    
    def test_known_metrics(self):
        """Test shorthand for known metrics."""
        assert _metric_shorthand("final_normalized_squared_flux") == "f_B"
        assert _metric_shorthand("max_BdotN_over_B") == "max⟨Bn⟩/⟨B⟩"
        assert _metric_shorthand("final_average_curvature") == "κ̄"
        assert _metric_shorthand("final_linking_number") == "LN"
        assert _metric_shorthand("coil_order") == "n"
        assert _metric_shorthand("num_coils") == "N"
    
    def test_unknown_metric(self):
        """Test shorthand for unknown metric."""
        assert _metric_shorthand("unknown_metric_name") == "unknown metric name"


class TestMetricDefinition:
    """Tests for _metric_definition function."""
    
    def test_known_metric_definitions(self):
        """Test definitions for known metrics."""
        flux_def = _metric_definition("final_normalized_squared_flux")
        assert "Normalized squared flux" in flux_def
        assert r"\int" in flux_def  # Check for the integral formula
        
        ln_def = _metric_definition("final_linking_number")
        assert "Linking number" in ln_def
        assert r"4\pi" in ln_def or "4π" in ln_def  # Check for the formula
    
    def test_unknown_metric_definition(self):
        """Test definition for unknown metric."""
        defn = _metric_definition("unknown_metric")
        assert defn == "Unknown Metric"


class TestLoadSubmissions:
    """Tests for _load_submissions function."""
    
    def test_load_submissions_empty_dir(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            submissions = list(_load_submissions(submissions_root))
            assert submissions == []
    
    def test_load_submissions_nonexistent_dir(self):
        """Test loading from nonexistent directory."""
        submissions_root = Path("/nonexistent/directory")
        submissions = list(_load_submissions(submissions_root))
        assert submissions == []
    
    def test_load_submissions_single_file(self):
        """Test loading single submission file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text(json.dumps({
                "metadata": {
                    "method_name": "test_method",
                    "contact": "test@example.com"
                },
                "metrics": {
                    "final_normalized_squared_flux": 0.001
                }
            }))
            
            submissions = list(_load_submissions(submissions_root))
            assert len(submissions) == 1
            method_key, path, data = submissions[0]
            assert method_key == "test_method:surface1:user1:2024-01-01_12-00"
            assert data["metadata"]["method_name"] == "test_method"
    
    def test_load_submissions_skips_non_results_json(self):
        """Test that non-results.json files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            # Create a non-results.json file
            other_file = submission_dir / "other.json"
            other_file.write_text(json.dumps({"test": "data"}))
            
            submissions = list(_load_submissions(submissions_root))
            assert len(submissions) == 0
    
    def test_load_submissions_invalid_json(self):
        """Test that invalid JSON files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text("invalid json content {")
            
            submissions = list(_load_submissions(submissions_root))
            assert len(submissions) == 0


class TestBuildMethodsJson:
    """Tests for build_methods_json function."""
    
    def test_build_methods_json_empty(self):
        """Test building methods JSON from empty submissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            repo_root = Path(tmpdir)
            methods = build_methods_json(submissions_root, repo_root)
            assert methods == {}
    
    def test_build_methods_json_single_submission(self):
        """Test building methods JSON from single submission."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir).resolve()
            repo_root = Path(tmpdir).resolve()
            
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text(json.dumps({
                "metadata": {
                    "method_name": "test_method",
                    "contact": "test@example.com",
                    "hardware": "CPU: Test",
                    "run_date": "2024-01-01T12:00:00"
                },
                "metrics": {
                    "final_normalized_squared_flux": 0.001,
                    "final_total_length": 100.0
                }
            }))
            
            methods = build_methods_json(submissions_root, repo_root)
            assert len(methods) == 1
            
            method_key = "test_method:surface1:user1:2024-01-01_12-00"
            assert method_key in methods
            method_data = methods[method_key]
            assert method_data["method_name"] == "test_method"
            assert method_data["contact"] == "test@example.com"
            assert method_data["metrics"]["final_normalized_squared_flux"] == 0.001
            assert method_data["score_primary"] == 0.001
    
    def test_build_methods_json_extracts_coil_params(self):
        """Test that coil parameters are extracted from case.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir).resolve()
            repo_root = Path(tmpdir).resolve()
            
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text(json.dumps({
                "metadata": {"method_name": "test_method"},
                "metrics": {"final_normalized_squared_flux": 0.001}
            }))
            
            # Create case.yaml with coil parameters
            case_yaml = submission_dir / "case.yaml"
            case_yaml.write_text("""coils_params:
  ncoils: 4
  order: 16
""")
            
            methods = build_methods_json(submissions_root, repo_root)
            method_key = "test_method:surface1:user1:2024-01-01_12-00"
            method_data = methods[method_key]
            
            assert method_data["metrics"]["num_coils"] == 4.0
            assert method_data["metrics"]["coil_order"] == 16.0
    
    def test_build_methods_json_skips_no_metrics(self):
        """Test that submissions without metrics are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            repo_root = Path(tmpdir)
            
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text(json.dumps({
                "metadata": {"method_name": "test_method"},
                "metrics": {}
            }))
            
            methods = build_methods_json(submissions_root, repo_root)
            assert methods == {}


class TestBuildLeaderboardJson:
    """Tests for build_leaderboard_json function."""
    
    def test_build_leaderboard_json_empty(self):
        """Test building leaderboard from empty methods."""
        leaderboard = build_leaderboard_json({})
        assert leaderboard == {"entries": []}
    
    def test_build_leaderboard_json_single_entry(self):
        """Test building leaderboard with single entry."""
        methods = {
            "method1:1.0": {
                "method_name": "method1",
                "method_version": "1.0",
                "contact": "user1",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "submissions/surface1/user1/2024-01-01_12-00/results.json",
                "score_primary": 0.001,
                "metrics": {"final_normalized_squared_flux": 0.001}
            }
        }
        leaderboard = build_leaderboard_json(methods)
        
        assert len(leaderboard["entries"]) == 1
        entry = leaderboard["entries"][0]
        assert entry["rank"] == 1
        assert entry["score_primary"] == 0.001
        assert entry["method_name"] == "method1"
    
    def test_build_leaderboard_json_sorts_ascending(self):
        """Test that leaderboard is sorted by score_primary ascending."""
        methods = {
            "method1:1.0": {
                "method_name": "method1",
                "method_version": "1.0",
                "contact": "user1",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "path1",
                "score_primary": 0.003,
                "metrics": {}
            },
            "method2:1.0": {
                "method_name": "method2",
                "method_version": "1.0",
                "contact": "user2",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "path2",
                "score_primary": 0.001,
                "metrics": {}
            },
        }
        leaderboard = build_leaderboard_json(methods)
        
        assert len(leaderboard["entries"]) == 2
        # Lower score should be first (ascending order)
        assert leaderboard["entries"][0]["score_primary"] == 0.001
        assert leaderboard["entries"][1]["score_primary"] == 0.003
        assert leaderboard["entries"][0]["rank"] == 1
        assert leaderboard["entries"][1]["rank"] == 2
    
    def test_build_leaderboard_json_skips_no_score(self):
        """Test that entries without score_primary are skipped."""
        methods = {
            "method1:1.0": {
                "method_name": "method1",
                "method_version": "1.0",
                "contact": "user1",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "path1",
                "score_primary": None,
                "metrics": {}
            }
        }
        leaderboard = build_leaderboard_json(methods)
        assert leaderboard == {"entries": []}

