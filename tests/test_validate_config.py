"""
Unit tests for validate_config.py
"""
import tempfile
from pathlib import Path
from stellcoilbench.validate_config import validate_case_config, validate_case_yaml_file


class TestValidateCaseConfig:
    """Tests for validate_case_config function."""
    
    def test_valid_config(self):
        """Test validation of a valid configuration."""
        data = {
            "description": "Test case",
            "surface_params": {
                "surface": "input.test",
                "range": "half period"
            },
            "coils_params": {
                "ncoils": 4,
                "order": 16
            },
            "optimizer_params": {
                "algorithm": "l-bfgs",
                "max_iterations": 200,
                "max_iter_lag": 10
            }
        }
        errors = validate_case_config(data)
        assert errors == []
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        data = {
            "description": "Test case"
        }
        errors = validate_case_config(data)
        assert len(errors) == 3
        assert any("Missing required field: surface_params" in e for e in errors)
        assert any("Missing required field: coils_params" in e for e in errors)
        assert any("Missing required field: optimizer_params" in e for e in errors)
    
    def test_invalid_surface_params(self):
        """Test validation with invalid surface_params."""
        data = {
            "description": "Test case",
            "surface_params": "not a dict",
            "coils_params": {},
            "optimizer_params": {}
        }
        errors = validate_case_config(data)
        assert any("surface_params must be a dictionary" in e for e in errors)
    
    def test_missing_surface_field(self):
        """Test validation with missing surface field."""
        data = {
            "description": "Test case",
            "surface_params": {},
            "coils_params": {},
            "optimizer_params": {}
        }
        errors = validate_case_config(data)
        assert any("surface_params must contain 'surface' field" in e for e in errors)
    
    def test_invalid_range(self):
        """Test validation with invalid range."""
        data = {
            "description": "Test case",
            "surface_params": {
                "surface": "input.test",
                "range": "invalid_range"
            },
            "coils_params": {},
            "optimizer_params": {}
        }
        errors = validate_case_config(data)
        assert any("surface_params.range must be one of" in e for e in errors)
    
    def test_invalid_ncoils(self):
        """Test validation with invalid ncoils."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": -1},
            "optimizer_params": {}
        }
        errors = validate_case_config(data)
        assert any("coils_params.ncoils must be a positive integer" in e for e in errors)
    
    def test_invalid_order(self):
        """Test validation with invalid order."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"order": 0},
            "optimizer_params": {}
        }
        errors = validate_case_config(data)
        assert any("coils_params.order must be a positive integer" in e for e in errors)
    
    def test_invalid_max_iterations(self):
        """Test validation with invalid max_iterations."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {},
            "optimizer_params": {"max_iterations": 0}
        }
        errors = validate_case_config(data)
        assert any("optimizer_params.max_iterations must be a positive integer" in e for e in errors)
    
    def test_valid_coil_objective_terms(self):
        """Test validation with valid coil_objective_terms."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "total_length": "l2_threshold",
                "coil_coil_distance": "l1_threshold",
                "coil_curvature": "lp_threshold",
                "coil_curvature_p": 2,
            }
        }
        errors = validate_case_config(data)
        assert errors == []
    
    def test_invalid_coil_objective_term_option(self):
        """Test validation with invalid coil_objective_term option."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "total_length": "invalid_option"
            }
        }
        errors = validate_case_config(data)
        assert any("total_length must be one of" in e for e in errors)
    
    def test_unknown_coil_objective_term(self):
        """Test validation with unknown coil_objective_term."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "unknown_term": "l2"
            }
        }
        errors = validate_case_config(data)
        assert any("Unknown coil_objective_terms key" in e for e in errors)
    
    def test_invalid_p_parameter(self):
        """Test validation with invalid _p parameter."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "coil_curvature": "lp_threshold",
                "coil_curvature_p": -1
            }
        }
        errors = validate_case_config(data)
        assert any("coil_curvature_p must be a positive number" in e for e in errors)


class TestValidateCaseYamlFile:
    """Tests for validate_case_yaml_file function."""
    
    def test_valid_yaml_file(self):
        """Test validation of a valid YAML file."""
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
""")
            f.flush()
            file_path = Path(f.name)
        
        try:
            errors = validate_case_yaml_file(file_path)
            assert errors == []
        finally:
            file_path.unlink()
    
    def test_invalid_yaml_syntax(self):
        """Test validation with invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: syntax: [")
            f.flush()
            file_path = Path(f.name)
        
        try:
            errors = validate_case_yaml_file(file_path)
            assert len(errors) > 0
            assert any("YAML parsing error" in e for e in errors)
        finally:
            file_path.unlink()
    
    def test_missing_fields_in_file(self):
        """Test validation with missing required fields in file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("description: Test case\n")
            f.flush()
            file_path = Path(f.name)
        
        try:
            errors = validate_case_yaml_file(file_path)
            assert len(errors) > 0
            assert any("Missing required field" in e for e in errors)
        finally:
            file_path.unlink()
    
    def test_empty_file(self):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            f.flush()
            file_path = Path(f.name)
        
        try:
            errors = validate_case_yaml_file(file_path)
            assert len(errors) > 0
            assert any("File is empty" in e or "no valid YAML" in e for e in errors)
        finally:
            file_path.unlink()

