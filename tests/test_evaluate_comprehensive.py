"""
Comprehensive unit tests for evaluate.py functions.
"""
import pytest
from pathlib import Path
from stellcoilbench.evaluate import (
    load_case_config,
    build_leaderboard,
    SubmissionResults,
)
from stellcoilbench.config_scheme import CaseConfig, SubmissionMetadata


class TestLoadCaseConfig:
    """Tests for load_case_config function."""
    
    def test_load_from_file_path(self, tmp_path):
        """Test loading config from direct file path."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        
        config = load_case_config(case_yaml)
        assert isinstance(config, CaseConfig)
        assert config.description == "Test case"
        assert config.coils_params["ncoils"] == 4
    
    def test_load_from_directory(self, tmp_path):
        """Test loading config from directory containing case.yaml."""
        case_dir = tmp_path / "case_dir"
        case_dir.mkdir()
        case_yaml = case_dir / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        
        config = load_case_config(case_dir)
        assert isinstance(config, CaseConfig)
        assert config.description == "Test case"
    
    def test_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_case_config(nonexistent)
        
        nonexistent_dir = tmp_path / "nonexistent_dir"
        with pytest.raises(FileNotFoundError):
            load_case_config(nonexistent_dir)
    
    def test_invalid_config(self, tmp_path):
        """Test that invalid config raises ValueError."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4.0  # Invalid: should be int
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="ncoils should be an integer"):
            load_case_config(case_yaml)
    
    def test_missing_required_fields(self, tmp_path):
        """Test that missing required fields raise ValueError."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
# Missing surface_params
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError):
            load_case_config(case_yaml)


class TestBuildLeaderboard:
    """Tests for build_leaderboard function."""
    
    def test_build_empty_leaderboard(self):
        """Test building leaderboard with no submissions."""
        leaderboard = build_leaderboard([])
        assert "entries" in leaderboard
        assert leaderboard["entries"] == []
    
    def test_build_simple_leaderboard(self):
        """Test building leaderboard with simple submissions."""
        submissions = [
            (Path("sub1"), {
                "metadata": {"method_name": "method1"},
                "cases": [{
                    "case_name": "case1",
                    "scores": {"score_primary": 0.5}
                }]
            }),
            (Path("sub2"), {
                "metadata": {"method_name": "method2"},
                "cases": [{
                    "case_name": "case1",
                    "scores": {"score_primary": 0.3}
                }]
            }),
        ]
        
        leaderboard = build_leaderboard(submissions)
        assert len(leaderboard["entries"]) == 2
        # Should be sorted descending by mean_score_primary (higher is better in this implementation)
        assert leaderboard["entries"][0]["mean_score_primary"] == 0.5
        assert leaderboard["entries"][1]["mean_score_primary"] == 0.3
    
    def test_build_leaderboard_custom_score_key(self):
        """Test building leaderboard with custom score key."""
        submissions = [
            (Path("sub1"), {
                "metadata": {"method_name": "method1"},
                "cases": [{
                    "case_name": "case1",
                    "scores": {"custom_score": 0.5}
                }]
            }),
        ]
        
        leaderboard = build_leaderboard(submissions, primary_score_key="custom_score")
        assert len(leaderboard["entries"]) == 1
        assert leaderboard["entries"][0]["mean_score_primary"] == 0.5
    
    def test_build_leaderboard_multiple_cases(self):
        """Test building leaderboard with multiple cases per submission."""
        submissions = [
            (Path("sub1"), {
                "metadata": {"method_name": "method1"},
                "cases": [
                    {"case_name": "case1", "scores": {"score_primary": 0.5}},
                    {"case_name": "case2", "scores": {"score_primary": 0.3}},
                ]
            }),
        ]
        
        leaderboard = build_leaderboard(submissions)
        assert len(leaderboard["entries"]) == 1
        # Should use mean of all case scores
        assert leaderboard["entries"][0]["mean_score_primary"] == 0.4  # (0.5 + 0.3) / 2
    
    def test_build_leaderboard_missing_metadata(self):
        """Test building leaderboard with missing metadata."""
        submissions = [
            (Path("sub1"), {
                "cases": [{
                    "case_name": "case1",
                    "scores": {"score_primary": 0.5}
                }]
            }),
        ]
        
        leaderboard = build_leaderboard(submissions)
        assert len(leaderboard["entries"]) == 1
        assert leaderboard["entries"][0].get("username") is None
    
    def test_build_leaderboard_missing_cases(self):
        """Test building leaderboard with missing cases."""
        submissions = [
            (Path("sub1"), {
                "metadata": {"username": "user1"},
            }),
        ]
        
        leaderboard = build_leaderboard(submissions)
        assert len(leaderboard["entries"]) == 0
    
    def test_build_leaderboard_empty_cases(self):
        """Test building leaderboard with empty cases list."""
        submissions = [
            (Path("sub1"), {
                "metadata": {"username": "user1"},
                "cases": []
            }),
        ]
        
        leaderboard = build_leaderboard(submissions)
        assert len(leaderboard["entries"]) == 0
    
    def test_build_leaderboard_missing_score(self):
        """Test building leaderboard with missing score."""
        submissions = [
            (Path("sub1"), {
                "metadata": {"username": "user1"},
                "cases": [{
                    "case_name": "case1",
                    "scores": {}
                }]
            }),
        ]
        
        leaderboard = build_leaderboard(submissions)
        assert len(leaderboard["entries"]) == 0
    
    def test_build_leaderboard_sorting(self):
        """Test that leaderboard is sorted correctly."""
        submissions = [
            (Path("sub3"), {
                "metadata": {"method_name": "method3"},
                "cases": [{"case_name": "case1", "scores": {"score_primary": 0.9}}]
            }),
            (Path("sub1"), {
                "metadata": {"method_name": "method1"},
                "cases": [{"case_name": "case1", "scores": {"score_primary": 0.1}}]
            }),
            (Path("sub2"), {
                "metadata": {"method_name": "method2"},
                "cases": [{"case_name": "case1", "scores": {"score_primary": 0.5}}]
            }),
        ]
        
        leaderboard = build_leaderboard(submissions)
        assert len(leaderboard["entries"]) == 3
        # Should be sorted descending by mean_score_primary
        assert leaderboard["entries"][0]["mean_score_primary"] == 0.9
        assert leaderboard["entries"][1]["mean_score_primary"] == 0.5
        assert leaderboard["entries"][2]["mean_score_primary"] == 0.1


class TestSubmissionResults:
    """Tests for SubmissionResults dataclass."""
    
    def test_submission_results_init(self):
        """Test SubmissionResults initialization."""
        metadata = SubmissionMetadata(
            method_name="testmethod",
            method_version="1.0",
            contact="test@example.com",
            hardware="CPU"
        )
        metrics = {"score_primary": 0.5}
        
        results = SubmissionResults(metadata=metadata, metrics=metrics)
        assert results.metadata == metadata
        assert results.metrics == metrics
    
    def test_submission_results_empty_metrics(self):
        """Test SubmissionResults with empty metrics."""
        metadata = SubmissionMetadata(
            method_name="testmethod",
            method_version="1.0",
            contact="test@example.com",
            hardware="CPU"
        )
        metrics = {}
        
        results = SubmissionResults(metadata=metadata, metrics=metrics)
        assert results.metrics == {}
