"""
Unit tests for new plasma surface case files.

Tests verify that the newly added case files for different plasma surfaces
can be loaded, validated, and that their corresponding surface files exist.
"""
import pytest
from pathlib import Path
from stellcoilbench.evaluate import load_case_config


class TestNewPlasmaSurfaceCases:
    """Tests for newly added plasma surface case files."""
    
    @pytest.fixture
    def repo_root(self):
        """Get the repository root directory."""
        return Path(__file__).parent.parent
    
    @pytest.fixture
    def cases_dir(self, repo_root):
        """Get the cases directory."""
        return repo_root / "cases"
    
    @pytest.fixture
    def plasma_surfaces_dir(self, repo_root):
        """Get the plasma_surfaces directory."""
        return repo_root / "plasma_surfaces"
    
    def test_basic_CFQS_case_loads(self, cases_dir):
        """Test that basic_CFQS.yaml can be loaded and validated."""
        case_file = cases_dir / "basic_CFQS.yaml"
        assert case_file.exists(), f"Case file {case_file} does not exist"
        
        config = load_case_config(case_file)
        assert config.description == "Basic test case"
        assert config.surface_params["surface"] == "input.cfqs_2b40"
        assert config.surface_params["range"] == "half period"
        assert config.coils_params["ncoils"] == 4
        assert config.coils_params["order"] == 8
        assert config.optimizer_params["algorithm"] == "L-BFGS-B"
    
    def test_basic_CFQS_surface_file_exists(self, plasma_surfaces_dir):
        """Test that the CFQS surface file exists."""
        surface_file = plasma_surfaces_dir / "input.cfqs_2b40"
        assert surface_file.exists(), f"Surface file {surface_file} does not exist"
        assert surface_file.is_file(), f"Surface file {surface_file} is not a file"
    
    def test_basic_HSX_case_loads(self, cases_dir):
        """Test that basic_HSX.yaml can be loaded and validated."""
        case_file = cases_dir / "basic_HSX.yaml"
        assert case_file.exists(), f"Case file {case_file} does not exist"
        
        config = load_case_config(case_file)
        assert config.description == "Basic test case"
        assert config.surface_params["surface"] == "input.HSX_QHS_mn1824_ns101"
        assert config.surface_params["range"] == "half period"
        assert config.coils_params["ncoils"] == 5
        assert config.coils_params["order"] == 8
        assert config.optimizer_params["algorithm"] == "L-BFGS-B"
        # Check that fourier_continuation is enabled
        assert hasattr(config, "fourier_continuation")
        assert config.fourier_continuation["enabled"] is True
        assert config.fourier_continuation["orders"] == [4, 8, 16]
    
    def test_basic_HSX_surface_file_exists(self, plasma_surfaces_dir):
        """Test that the HSX surface file exists."""
        surface_file = plasma_surfaces_dir / "input.HSX_QHS_mn1824_ns101"
        assert surface_file.exists(), f"Surface file {surface_file} does not exist"
        assert surface_file.is_file(), f"Surface file {surface_file} is not a file"
    
    def test_basic_W7X_case_loads(self, cases_dir):
        """Test that basic_W7X.yaml can be loaded and validated."""
        case_file = cases_dir / "basic_W7X.yaml"
        assert case_file.exists(), f"Case file {case_file} does not exist"
        
        config = load_case_config(case_file)
        assert config.description == "Basic test case"
        assert config.surface_params["surface"] == "input.W7-X_without_coil_ripple_beta0p05_d23p4_tm"
        assert config.surface_params["range"] == "half period"
        assert config.coils_params["ncoils"] == 4
        assert config.coils_params["order"] == 4
        assert config.optimizer_params["algorithm"] == "L-BFGS-B"
        # Check that fourier_continuation is enabled
        assert hasattr(config, "fourier_continuation")
        assert config.fourier_continuation["enabled"] is True
        assert config.fourier_continuation["orders"] == [4, 8]
    
    def test_basic_W7X_surface_file_exists(self, plasma_surfaces_dir):
        """Test that the W7-X surface file exists."""
        surface_file = plasma_surfaces_dir / "input.W7-X_without_coil_ripple_beta0p05_d23p4_tm"
        assert surface_file.exists(), f"Surface file {surface_file} does not exist"
        assert surface_file.is_file(), f"Surface file {surface_file} is not a file"
    
    def test_all_new_cases_have_valid_surface_files(self, cases_dir, plasma_surfaces_dir):
        """Test that all new case files reference valid surface files."""
        new_cases = [
            ("basic_CFQS.yaml", "input.cfqs_2b40"),
            ("basic_HSX.yaml", "input.HSX_QHS_mn1824_ns101"),
            ("basic_W7X.yaml", "input.W7-X_without_coil_ripple_beta0p05_d23p4_tm"),
        ]
        
        for case_filename, expected_surface in new_cases:
            case_file = cases_dir / case_filename
            assert case_file.exists(), f"Case file {case_file} does not exist"
            
            config = load_case_config(case_file)
            surface_name = config.surface_params["surface"]
            assert surface_name == expected_surface, \
                f"Case {case_filename} references surface '{surface_name}', expected '{expected_surface}'"
            
            surface_file = plasma_surfaces_dir / surface_name
            assert surface_file.exists(), \
                f"Surface file {surface_file} referenced by {case_filename} does not exist"
            assert surface_file.is_file(), \
                f"Surface file {surface_file} referenced by {case_filename} is not a file"
    
    def test_new_cases_have_required_fields(self, cases_dir):
        """Test that all new case files have required fields."""
        new_cases = [
            "basic_CFQS.yaml",
            "basic_HSX.yaml",
            "basic_W7X.yaml",
        ]
        
        # required_fields = {
        #     "description",
        #     "surface_params",
        #     "coils_params",
        #     "optimizer_params",
        # }
        
        for case_filename in new_cases:
            case_file = cases_dir / case_filename
            config = load_case_config(case_file)
            
            # Check required top-level fields
            assert hasattr(config, "description"), \
                f"{case_filename} missing 'description' field"
            assert hasattr(config, "surface_params"), \
                f"{case_filename} missing 'surface_params' field"
            assert hasattr(config, "coils_params"), \
                f"{case_filename} missing 'coils_params' field"
            assert hasattr(config, "optimizer_params"), \
                f"{case_filename} missing 'optimizer_params' field"
            
            # Check surface_params has required fields
            assert "surface" in config.surface_params, \
                f"{case_filename} missing 'surface' in surface_params"
            assert "range" in config.surface_params, \
                f"{case_filename} missing 'range' in surface_params"
            
            # Check coils_params has required fields
            assert "ncoils" in config.coils_params, \
                f"{case_filename} missing 'ncoils' in coils_params"
            assert "order" in config.coils_params, \
                f"{case_filename} missing 'order' in coils_params"
            
            # Check optimizer_params has required fields
            assert "algorithm" in config.optimizer_params, \
                f"{case_filename} missing 'algorithm' in optimizer_params"
    
    def test_new_cases_have_valid_coil_objective_terms(self, cases_dir):
        """Test that new case files have valid coil_objective_terms if present."""
        new_cases = [
            "basic_CFQS.yaml",
            "basic_HSX.yaml",
            "basic_W7X.yaml",
        ]
        
        # Valid threshold parameter names (these are numeric values, not option strings)
        valid_threshold_names = {
            "length_threshold",
            "cc_threshold",
            "cs_threshold",
            "curvature_threshold",
            "arclength_variation_threshold",
            "msc_threshold",
            "force_threshold",
            "torque_threshold",
            "flux_threshold",
        }
        
        for case_filename in new_cases:
            case_file = cases_dir / case_filename
            config = load_case_config(case_file)
            
            # coil_objective_terms is optional, but if present should be valid
            if hasattr(config, "coil_objective_terms") and config.coil_objective_terms:
                # Valid options for each term
                valid_options = {
                    "l1", "l1_threshold", "l2", "l2_threshold",
                    "lp", "lp_threshold", ""
                }
                
                for term, value in config.coil_objective_terms.items():
                    # Skip threshold parameters (they are numeric values)
                    if term in valid_threshold_names:
                        assert isinstance(value, (int, float)) and value >= 0, \
                            f"{case_filename} coil_objective_terms.{term} should be a non-negative number"
                    elif term.endswith("_p"):
                        # p parameter should be numeric
                        assert isinstance(value, (int, float)) and value > 0, \
                            f"{case_filename} coil_objective_terms.{term} should be a positive number"
                    elif term == "linking_number":
                        # linking_number can be empty string
                        assert value == "", \
                            f"{case_filename} coil_objective_terms.linking_number should be empty string"
                    else:
                        # Regular terms should have valid options
                        assert value in valid_options, \
                            f"{case_filename} coil_objective_terms.{term} has invalid value '{value}'"

