"""
Unit tests for coil_optimization.py

Note: Full optimization tests require simsopt and are integration tests.
These unit tests focus on functions that can be tested without running optimizations.
"""
import pytest
from pathlib import Path
from stellcoilbench.coil_optimization import load_coils_config
from stellcoilbench.evaluate import load_case_config


class TestLoadCoilsConfig:
    """Tests for load_coils_config function."""
    
    def test_load_coils_config_valid(self, tmp_path):
        """Test loading a valid coils config file."""
        config_file = tmp_path / "coils.yaml"
        config_file.write_text("""ncoils: 4
order: 16
verbose: "True"
""")
        
        config = load_coils_config(config_file)
        assert config["ncoils"] == 4
        assert config["order"] == 16
        assert config["verbose"] == "True"
    
    def test_load_coils_config_nonexistent(self):
        """Test loading nonexistent config file raises error."""
        config_file = Path("/nonexistent/config.yaml")
        with pytest.raises(Exception):  # FileNotFoundError or similar
            load_coils_config(config_file)
    
    def test_load_coils_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises error."""
        config_file = tmp_path / "coils.yaml"
        config_file.write_text("invalid: yaml: [")
        
        with pytest.raises(Exception):  # YAML parsing error
            load_coils_config(config_file)
    
    def test_load_coils_config_not_dict(self, tmp_path):
        """Test that non-dict YAML raises error."""
        config_file = tmp_path / "coils.yaml"
        config_file.write_text("- item1\n- item2\n")
        
        with pytest.raises(ValueError, match="Expected dict"):
            load_coils_config(config_file)


class TestCoilsParamsValidation:
    """Tests for coils_params validation in case.yaml files."""
    
    def test_unknown_parameter_target_B(self, tmp_path):
        """Test that target_B in coils_params raises validation error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
  target_B: 1.0
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="Unknown coils_params key.*target_B"):
            load_case_config(case_yaml)
    
    def test_unknown_parameter_other(self, tmp_path):
        """Test that other unknown parameters raise validation error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
  unknown_param: value
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="Unknown coils_params key.*unknown_param"):
            load_case_config(case_yaml)
    
    def test_ncoils_as_float(self, tmp_path):
        """Test that ncoils as float (even if integer value) raises error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4.0
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="ncoils should be an integer, not a float"):
            load_case_config(case_yaml)
    
    def test_order_as_float(self, tmp_path):
        """Test that order as float (even if integer value) raises error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16.0
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="order should be an integer, not a float"):
            load_case_config(case_yaml)
    
    def test_nturns_as_unknown_parameter(self, tmp_path):
        """Test that nturns as unknown parameter raises error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
  nturns: 200
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="Unknown coils_params key.*nturns"):
            load_case_config(case_yaml)
    
    def test_coil_radius_as_unknown_parameter(self, tmp_path):
        """Test that coil_radius as unknown parameter raises error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
  coil_radius: "0.05"
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="Unknown coils_params key.*coil_radius"):
            load_case_config(case_yaml)
    
    def test_ncoils_as_non_integer_type(self, tmp_path):
        """Test that ncoils as string raises error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: "4"
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="ncoils must be a positive integer"):
            load_case_config(case_yaml)
    
    def test_order_as_non_integer_type(self, tmp_path):
        """Test that order as string raises error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: "16"
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="order must be a positive integer"):
            load_case_config(case_yaml)
    
    def test_ncoils_as_negative(self, tmp_path):
        """Test that negative ncoils raises error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: -1
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="ncoils must be a positive integer"):
            load_case_config(case_yaml)
    
    def test_order_as_zero(self, tmp_path):
        """Test that zero order raises error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 0
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="order must be a positive integer"):
            load_case_config(case_yaml)
    
    def test_valid_coils_params(self, tmp_path):
        """Test that valid coils_params pass validation."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
  verbose: "True"
optimizer_params:
  algorithm: l-bfgs
""")
        
        # Should not raise an error
        config = load_case_config(case_yaml)
        assert config.coils_params["ncoils"] == 4
        assert config.coils_params["order"] == 16
        assert config.coils_params["verbose"] == "True"
    
    def test_unknown_surface_params(self, tmp_path):
        """Test that unknown surface_params raise validation error."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
  range: half period
  nphi: 32
  ntheta: 32
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        
        with pytest.raises(ValueError, match="Unknown surface_params key"):
            load_case_config(case_yaml)
