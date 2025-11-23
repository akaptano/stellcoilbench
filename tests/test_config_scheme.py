"""
Unit tests for config_scheme.py
"""
from stellcoilbench.config_scheme import CaseConfig, SubmissionMetadata


class TestCaseConfig:
    """Tests for CaseConfig dataclass."""
    
    def test_from_dict_minimal(self):
        """Test creating CaseConfig from minimal dictionary."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
        }
        config = CaseConfig.from_dict(data)
        
        assert config.description == "Test case"
        assert config.surface_params == {"surface": "input.test"}
        assert config.coils_params == {"ncoils": 4}
        assert config.optimizer_params == {"algorithm": "l-bfgs"}
        assert config.scoring is None
        assert config.coil_objective_terms is None
    
    def test_from_dict_with_optional_fields(self):
        """Test creating CaseConfig with optional fields."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "scoring": {"primary": "flux"},
            "coil_objective_terms": {"total_length": "l2_threshold"},
        }
        config = CaseConfig.from_dict(data)
        
        assert config.scoring == {"primary": "flux"}
        assert config.coil_objective_terms == {"total_length": "l2_threshold"}
    
    def test_from_dict_missing_fields(self):
        """Test that missing fields get default values."""
        data = {
            "description": "",
            "surface_params": {},
            "coils_params": {},
            "optimizer_params": {},
        }
        config = CaseConfig.from_dict(data)
        
        assert config.description == ""
        assert config.surface_params == {}
        assert config.coils_params == {}
        assert config.optimizer_params == {}
        assert config.scoring is None
        assert config.coil_objective_terms is None


class TestSubmissionMetadata:
    """Tests for SubmissionMetadata dataclass."""
    
    def test_submission_metadata_creation(self):
        """Test creating SubmissionMetadata."""
        metadata = SubmissionMetadata(
            method_name="test_method",
            method_version="1.0.0",
            contact="test@example.com",
            hardware="CPU: Test",
            notes="Test notes"
        )
        
        assert metadata.method_name == "test_method"
        assert metadata.method_version == "1.0.0"
        assert metadata.contact == "test@example.com"
        assert metadata.hardware == "CPU: Test"
        assert metadata.notes == "Test notes"
    
    def test_submission_metadata_default_notes(self):
        """Test that notes defaults to empty string."""
        metadata = SubmissionMetadata(
            method_name="test_method",
            method_version="1.0.0",
            contact="test@example.com",
            hardware="CPU: Test"
        )
        
        assert metadata.notes == ""

