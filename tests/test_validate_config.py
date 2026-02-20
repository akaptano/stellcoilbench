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

    def test_invalid_max_iter_lag(self):
        """Test validation with invalid max_iter_lag."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"max_iter_lag": 0},
        }
        errors = validate_case_config(data)
        assert any("optimizer_params.max_iter_lag must be a positive integer" in e for e in errors)

    def test_non_dict_coils_params(self):
        """Test validation when coils_params is not a dict."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": "not a dict",
            "optimizer_params": {"algorithm": "l-bfgs"},
        }
        errors = validate_case_config(data)
        assert any("coils_params must be a dictionary" in e for e in errors)

    def test_non_dict_optimizer_params(self):
        """Test validation when optimizer_params is not a dict."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": "not a dict",
        }
        errors = validate_case_config(data)
        assert any("optimizer_params must be a dictionary" in e for e in errors)

    def test_non_dict_coil_objective_terms(self):
        """Test validation when coil_objective_terms is not a dict."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": "not a dict",
        }
        errors = validate_case_config(data)
        assert any("coil_objective_terms must be a dictionary" in e for e in errors)
    
    def test_valid_coil_objective_terms(self):
        """Test validation with valid coil_objective_terms."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "total_length": "l2_threshold",
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
    
    def test_valid_named_weights(self):
        """Test validation with valid named weight parameters."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "L-BFGS-B"},
            "coil_objective_terms": {
                "total_length": "l2_threshold",
                "length_weight": 2.0,
                "linking_number": "",
                "linking_weight": 1000.0,
                "flux_weight": 1.5,
            }
        }
        errors = validate_case_config(data)
        assert errors == []
    
    def test_all_named_weights_valid(self):
        """Test validation with all named weight parameters."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "length_weight": 1.0,
                "cc_weight": 100.0,
                "cs_weight": 50.0,
                "curvature_weight": 0.5,
                "arclength_variation_weight": 0.1,
                "msc_weight": 2.0,
                "force_weight": 10.0,
                "torque_weight": 5.0,
                "flux_weight": 1.0,
                "linking_weight": 1000.0,
            }
        }
        errors = validate_case_config(data)
        assert errors == []
    
    def test_invalid_negative_weight(self):
        """Test validation with negative weight parameter."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "linking_weight": -1.0,
            }
        }
        errors = validate_case_config(data)
        assert any("linking_weight must be a non-negative number" in e for e in errors)
    
    def test_invalid_string_weight(self):
        """Test validation with non-numeric string weight parameter."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "length_weight": "high",
            }
        }
        errors = validate_case_config(data)
        assert any("length_weight must be a non-negative number" in e for e in errors)
    
    def test_valid_scientific_notation_weight(self):
        """Test validation with scientific notation weight (as string from YAML)."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "linking_weight": "1e3",  # YAML sometimes parses this as string
            }
        }
        errors = validate_case_config(data)
        assert errors == []
    
    def test_zero_weight_valid(self):
        """Test validation with zero weight (should be valid - disables the term)."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "length_weight": 0.0,
            }
        }
        errors = validate_case_config(data)
        assert errors == []
    
    def test_unknown_weight_rejected(self):
        """Test that unknown weight parameters are rejected."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "coil_objective_terms": {
                "unknown_weight": 1.0,
            }
        }
        errors = validate_case_config(data)
        assert any("Unknown coil_objective_terms key" in e for e in errors)


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

    def test_non_dict_root_in_file(self):
        """Test validation when YAML root is a list, not a dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("- item1\n- item2\n")
            f.flush()
            file_path = Path(f.name)
        try:
            errors = validate_case_yaml_file(file_path)
            assert len(errors) > 0
            assert any("Root element must be a dictionary" in e for e in errors)
        finally:
            file_path.unlink()

    def test_file_not_found(self):
        """Test validation when file does not exist."""
        errors = validate_case_yaml_file(Path("/nonexistent/file.yaml"))
        assert len(errors) > 0
        assert any("Error reading file" in e for e in errors)


class TestValidateCaseConfigEdgeCases:
    """Tests for edge case validation branches in validate_case_config."""

    def _base_config(self):
        return {
            "description": "Test",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 16},
            "optimizer_params": {"algorithm": "l-bfgs"},
        }

    def test_virtual_casing_non_boolean(self):
        """virtual_casing must be boolean."""
        data = self._base_config()
        data["surface_params"]["virtual_casing"] = "yes"
        errors = validate_case_config(data)
        assert any("virtual_casing must be a boolean" in e for e in errors)

    def test_threshold_boolean_rejected(self):
        """Boolean values should be rejected for threshold parameters."""
        data = self._base_config()
        data["coil_objective_terms"] = {"length_threshold": True}
        errors = validate_case_config(data)
        assert any("must be a non-negative number" in e for e in errors)

    def test_threshold_unsupported_type(self):
        """List/dict/None should be rejected for threshold parameters."""
        data = self._base_config()
        data["coil_objective_terms"] = {"length_threshold": [1, 2]}
        errors = validate_case_config(data)
        assert any("must be a non-negative number" in e for e in errors)

    def test_threshold_negative_value(self):
        """Negative threshold value should be rejected."""
        data = self._base_config()
        data["coil_objective_terms"] = {"length_threshold": -5}
        errors = validate_case_config(data)
        assert any("must be a non-negative number" in e for e in errors)

    def test_p_param_boolean_rejected(self):
        """Boolean values should be rejected for _p parameters."""
        data = self._base_config()
        data["coil_objective_terms"] = {"curvature_p": True}
        errors = validate_case_config(data)
        assert any("must be a positive number" in e for e in errors)

    def test_p_param_valid_string(self):
        """Valid string _p parameter should pass."""
        data = self._base_config()
        data["coil_objective_terms"] = {"curvature_p": "2.5"}
        errors = validate_case_config(data)
        assert not any("curvature_p" in e for e in errors)

    def test_p_param_invalid_string(self):
        """Invalid string _p parameter should fail."""
        data = self._base_config()
        data["coil_objective_terms"] = {"curvature_p": "abc"}
        errors = validate_case_config(data)
        assert any("must be a positive number" in e for e in errors)

    def test_p_param_zero_rejected(self):
        """Zero _p parameter should fail (must be positive)."""
        data = self._base_config()
        data["coil_objective_terms"] = {"curvature_p": "0"}
        errors = validate_case_config(data)
        assert any("must be a positive number" in e for e in errors)

    def test_invalid_coil_curvature(self):
        """Invalid coil_curvature value should be rejected."""
        data = self._base_config()
        data["coil_objective_terms"] = {"coil_curvature": "invalid"}
        errors = validate_case_config(data)
        assert any("coil_curvature must be one of" in e for e in errors)

    def test_invalid_coil_arclength_variation(self):
        """Invalid coil_arclength_variation should be rejected."""
        data = self._base_config()
        data["coil_objective_terms"] = {"coil_arclength_variation": "invalid"}
        errors = validate_case_config(data)
        assert any("coil_arclength_variation" in e and "must be one of" in e for e in errors)

    def test_invalid_coil_mean_squared_curvature(self):
        """Invalid coil_mean_squared_curvature should be rejected."""
        data = self._base_config()
        data["coil_objective_terms"] = {"coil_mean_squared_curvature": "invalid"}
        errors = validate_case_config(data)
        assert any("coil_mean_squared_curvature must be one of" in e for e in errors)

    def test_invalid_linking_number(self):
        """Invalid linking_number should be rejected."""
        data = self._base_config()
        data["coil_objective_terms"] = {"linking_number": "soft"}
        errors = validate_case_config(data)
        assert any("linking_number must be one of" in e for e in errors)

    def test_invalid_coil_coil_force(self):
        """Invalid coil_coil_force should be rejected."""
        data = self._base_config()
        data["coil_objective_terms"] = {"coil_coil_force": "invalid"}
        errors = validate_case_config(data)
        assert any("coil_coil_force must be one of" in e for e in errors)

    def test_invalid_coil_coil_torque(self):
        """Invalid coil_coil_torque should be rejected."""
        data = self._base_config()
        data["coil_objective_terms"] = {"coil_coil_torque": "invalid"}
        errors = validate_case_config(data)
        assert any("coil_coil_torque must be one of" in e for e in errors)

    def test_fourier_continuation_not_dict(self):
        """fourier_continuation must be a dict."""
        data = self._base_config()
        data["fourier_continuation"] = "not a dict"
        errors = validate_case_config(data)
        assert any("fourier_continuation must be a dictionary" in e for e in errors)

    def test_fourier_continuation_enabled_not_bool(self):
        """fourier_continuation.enabled must be boolean."""
        data = self._base_config()
        data["fourier_continuation"] = {"enabled": "yes"}
        errors = validate_case_config(data)
        assert any("enabled must be a boolean" in e for e in errors)

    def test_fourier_continuation_orders_not_list(self):
        """fourier_continuation.orders must be a list."""
        data = self._base_config()
        data["fourier_continuation"] = {"orders": "3,5"}
        errors = validate_case_config(data)
        assert any("orders must be a list" in e for e in errors)

    def test_fourier_continuation_orders_empty(self):
        """fourier_continuation.orders must be non-empty."""
        data = self._base_config()
        data["fourier_continuation"] = {"orders": []}
        errors = validate_case_config(data)
        assert any("must be non-empty" in e for e in errors)

    def test_fourier_continuation_orders_invalid_values(self):
        """fourier_continuation.orders must contain only positive integers."""
        data = self._base_config()
        data["fourier_continuation"] = {"orders": [0, 1, 2]}
        errors = validate_case_config(data)
        assert any("positive integers" in e for e in errors)

    def test_fourier_continuation_orders_unsorted(self):
        """fourier_continuation.orders must be in ascending order."""
        data = self._base_config()
        data["fourier_continuation"] = {"orders": [5, 3]}
        errors = validate_case_config(data)
        assert any("ascending order" in e for e in errors)

    def test_weight_boolean_rejected(self):
        """Boolean values should be rejected for weight parameters."""
        data = self._base_config()
        data["coil_objective_terms"] = {"cc_weight": False}
        errors = validate_case_config(data)
        assert any("must be a non-negative number" in e for e in errors)

