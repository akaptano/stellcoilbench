"""
Unit tests for evaluate.py
"""
import pytest
import tempfile
from pathlib import Path
from stellcoilbench.evaluate import (
    load_case_config,
    evaluate_case,
    build_leaderboard,
    CaseConfig,
)


class TestLoadCaseConfig:
    """Tests for load_case_config function."""
    
    def test_load_from_file(self):
        """Test loading config from a file path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""description: Test case
surface_params:
  surface: input.test
  range: half period
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
  max_iterations: 200
  max_iter_lag: 10
""")
            f.flush()
            file_path = Path(f.name)
        
        try:
            config = load_case_config(file_path)
            assert config.description == "Test case"
            assert config.surface_params["surface"] == "input.test"
            assert config.coils_params["ncoils"] == 4
        finally:
            file_path.unlink()
    
    def test_load_from_directory(self):
        """Test loading config from a directory containing case.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            case_yaml = case_dir / "case.yaml"
            case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
optimizer_params:
  algorithm: l-bfgs
""")
            
            config = load_case_config(case_dir)
            assert config.description == "Test case"
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_case_config(Path("/nonexistent/path.yaml"))
    
    def test_invalid_config_raises_error(self):
        """Test that invalid config raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("description: Test case\n")  # Missing required fields
            f.flush()
            file_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Configuration validation failed"):
                load_case_config(file_path)
        finally:
            file_path.unlink()


class TestEvaluateCase:
    """Tests for evaluate_case function."""
    
    def test_evaluate_case_passes_through(self):
        """Test that evaluate_case returns the results_dict as-is."""
        case_cfg = CaseConfig(
            description="Test",
            surface_params={"surface": "input.test"},
            coils_params={},
            optimizer_params={}
        )
        results_dict = {"metric1": 1.0, "metric2": 2.0}
        
        output = evaluate_case(case_cfg, results_dict)
        assert output == results_dict


class TestBuildLeaderboard:
    """Tests for build_leaderboard function."""
    
    def test_build_leaderboard_empty(self):
        """Test building leaderboard from empty submissions."""
        submissions = []
        leaderboard = build_leaderboard(submissions)
        assert leaderboard == {"entries": []}
    
    def test_build_leaderboard_single_entry(self):
        """Test building leaderboard with single entry."""
        submissions = [
            (
                Path("submission1.json"),
                {
                    "metadata": {"method_name": "method1", "method_version": "1.0"},
                    "cases": [
                        {"scores": {"score_primary": 0.5}},
                        {"scores": {"score_primary": 0.3}},
                    ]
                }
            )
        ]
        leaderboard = build_leaderboard(submissions)
        
        assert len(leaderboard["entries"]) == 1
        entry = leaderboard["entries"][0]
        assert entry["method_name"] == "method1"
        assert entry["method_version"] == "1.0"
        assert entry["mean_score_primary"] == 0.4
        assert entry["num_cases"] == 2
        assert entry["rank"] == 1
    
    def test_build_leaderboard_multiple_entries(self):
        """Test building leaderboard with multiple entries."""
        submissions = [
            (
                Path("submission1.json"),
                {
                    "metadata": {"method_name": "method1", "method_version": "1.0"},
                    "cases": [{"scores": {"score_primary": 0.5}}]
                }
            ),
            (
                Path("submission2.json"),
                {
                    "metadata": {"method_name": "method2", "method_version": "2.0"},
                    "cases": [{"scores": {"score_primary": 0.3}}]
                }
            ),
        ]
        leaderboard = build_leaderboard(submissions)
        
        assert len(leaderboard["entries"]) == 2
        # Should be sorted descending by score
        assert leaderboard["entries"][0]["mean_score_primary"] == 0.5
        assert leaderboard["entries"][1]["mean_score_primary"] == 0.3
        assert leaderboard["entries"][0]["rank"] == 1
        assert leaderboard["entries"][1]["rank"] == 2
    
    def test_build_leaderboard_skips_empty_cases(self):
        """Test that entries with no cases are skipped."""
        submissions = [
            (
                Path("submission1.json"),
                {
                    "metadata": {"method_name": "method1"},
                    "cases": []
                }
            ),
            (
                Path("submission2.json"),
                {
                    "metadata": {"method_name": "method2"},
                    "cases": [{"scores": {"score_primary": 0.3}}]
                }
            ),
        ]
        leaderboard = build_leaderboard(submissions)
        
        assert len(leaderboard["entries"]) == 1
        assert leaderboard["entries"][0]["method_name"] == "method2"
    
    def test_build_leaderboard_skips_no_scores(self):
        """Test that entries with no scores are skipped."""
        submissions = [
            (
                Path("submission1.json"),
                {
                    "metadata": {"method_name": "method1"},
                    "cases": [{"scores": {}}]
                }
            ),
        ]
        leaderboard = build_leaderboard(submissions)
        
        assert len(leaderboard["entries"]) == 0
    
    def test_build_leaderboard_custom_score_key(self):
        """Test building leaderboard with custom score key."""
        submissions = [
            (
                Path("submission1.json"),
                {
                    "metadata": {"method_name": "method1"},
                    "cases": [{"scores": {"custom_score": 0.7}}]
                }
            ),
        ]
        leaderboard = build_leaderboard(submissions, primary_score_key="custom_score")
        
        assert len(leaderboard["entries"]) == 1
        assert leaderboard["entries"][0]["mean_score_primary"] == 0.7

