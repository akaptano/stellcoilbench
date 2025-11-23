"""
Unit tests for coil_optimization.py

Note: Full optimization tests require simsopt and are integration tests.
These unit tests focus on functions that can be tested without running optimizations.
"""
import pytest
from pathlib import Path
from stellcoilbench.coil_optimization import load_coils_config


class TestLoadCoilsConfig:
    """Tests for load_coils_config function."""
    
    def test_load_coils_config_valid(self, tmp_path):
        """Test loading a valid coils config file."""
        config_file = tmp_path / "coils.yaml"
        config_file.write_text("""ncoils: 4
order: 16
target_B: 1.0
""")
        
        config = load_coils_config(config_file)
        assert config["ncoils"] == 4
        assert config["order"] == 16
        assert config["target_B"] == 1.0
    
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

