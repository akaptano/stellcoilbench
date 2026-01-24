"""
Comprehensive unit tests for algorithm options validation functions.
"""
import pytest
from stellcoilbench.coil_optimization import (
    _get_scipy_algorithm_options,
    _validate_algorithm_options,
)


class TestGetScipyAlgorithmOptions:
    """Tests for _get_scipy_algorithm_options function."""
    
    def test_common_options(self):
        """Test that common options are present for all algorithms."""
        options = _get_scipy_algorithm_options("BFGS")
        assert "maxiter" in options
        assert "disp" in options
        assert isinstance(options["maxiter"], list)
        assert int in options["maxiter"]
        assert bool in options["disp"]
    
    def test_bfgs_options(self):
        """Test BFGS-specific options."""
        options = _get_scipy_algorithm_options("BFGS")
        assert "gtol" in options
        assert "norm" in options
        assert float in options["gtol"]
        assert float in options["norm"]
    
    def test_lbfgsb_options(self):
        """Test L-BFGS-B-specific options."""
        options = _get_scipy_algorithm_options("L-BFGS-B")
        assert "maxfun" in options
        assert "ftol" in options
        assert "gtol" in options
        assert "eps" in options
        assert "maxls" in options
        assert int in options["maxfun"]
        assert float in options["ftol"]
    
    def test_slsqp_options(self):
        """Test SLSQP-specific options."""
        options = _get_scipy_algorithm_options("SLSQP")
        assert "ftol" in options
        assert "eps" in options
    
    def test_nelder_mead_options(self):
        """Test Nelder-Mead-specific options."""
        options = _get_scipy_algorithm_options("Nelder-Mead")
        assert "xatol" in options
        assert "fatol" in options
        assert "adaptive" in options
        assert bool in options["adaptive"]
    
    def test_powell_options(self):
        """Test Powell-specific options."""
        options = _get_scipy_algorithm_options("Powell")
        assert "xtol" in options
        assert "ftol" in options
        assert "maxfev" in options
    
    def test_cg_options(self):
        """Test CG-specific options."""
        options = _get_scipy_algorithm_options("CG")
        assert "gtol" in options
        assert "norm" in options
    
    def test_newton_cg_options(self):
        """Test Newton-CG-specific options."""
        options = _get_scipy_algorithm_options("Newton-CG")
        assert "xtol" in options
        assert "eps" in options
    
    def test_tnc_options(self):
        """Test TNC-specific options."""
        options = _get_scipy_algorithm_options("TNC")
        assert "maxfun" in options
        assert "ftol" in options
        assert "gtol" in options
        assert "eps" in options
    
    def test_cobyla_options(self):
        """Test COBYLA-specific options."""
        options = _get_scipy_algorithm_options("COBYLA")
        assert "maxiter" in options
        assert "rhobeg" in options
        assert "tol" in options
    
    def test_trust_constr_options(self):
        """Test trust-constr-specific options."""
        options = _get_scipy_algorithm_options("trust-constr")
        assert "xtol" in options
        assert "gtol" in options
        assert "barrier_tol" in options
        assert "initial_barrier_parameter" in options
        assert "initial_barrier_tolerance" in options
        assert "initial_trust_radius" in options
        assert "max_trust_radius" in options
    
    def test_unknown_algorithm(self):
        """Test that unknown algorithm returns only common options."""
        options = _get_scipy_algorithm_options("UnknownAlgorithm")
        assert "maxiter" in options
        assert "disp" in options
        assert len(options) == 2  # Only common options


class TestValidateAlgorithmOptions:
    """Tests for _validate_algorithm_options function."""
    
    def test_valid_options(self):
        """Test that valid options pass validation."""
        _validate_algorithm_options("BFGS", {"maxiter": 100, "gtol": 1e-5, "disp": True})
        _validate_algorithm_options("L-BFGS-B", {"maxfun": 1000, "ftol": 1e-6})
        _validate_algorithm_options("SLSQP", {"ftol": 1e-6, "eps": 1e-8})
    
    def test_invalid_option_name(self):
        """Test that invalid option names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid algorithm options"):
            _validate_algorithm_options("BFGS", {"invalid_option": 100})
    
    def test_invalid_option_type(self):
        """Test that wrong option types raise ValueError."""
        # String instead of int should raise ValueError
        with pytest.raises(ValueError, match="Invalid algorithm options"):
            _validate_algorithm_options("BFGS", {"maxiter": "100"})  # Should be int, not str
        
        # Note: The validation checks if isinstance(option_value, t) for t in valid_types
        # bool is not in [int], so it should fail, but the check might pass bool through
        # Let's test with a type that definitely won't match
        with pytest.raises(ValueError, match="Invalid algorithm options"):
            _validate_algorithm_options("BFGS", {"maxiter": []})  # Should be int, not list
    
    def test_multiple_invalid_options(self):
        """Test that multiple invalid options are all reported."""
        with pytest.raises(ValueError, match="Invalid algorithm options"):
            try:
                _validate_algorithm_options("BFGS", {
                    "invalid1": 100,
                    "invalid2": 200,
                    "maxiter": "wrong_type"
                })
            except ValueError as e:
                error_msg = str(e)
                assert "invalid1" in error_msg or "invalid2" in error_msg
                assert "maxiter" in error_msg or "wrong type" in error_msg.lower()
                raise
    
    def test_empty_options(self):
        """Test that empty options dict passes validation."""
        _validate_algorithm_options("BFGS", {})
    
    def test_common_options_only(self):
        """Test validation with only common options."""
        _validate_algorithm_options("BFGS", {"maxiter": 100, "disp": False})
    
    def test_algorithm_specific_options(self):
        """Test validation with algorithm-specific options."""
        _validate_algorithm_options("Nelder-Mead", {
            "xatol": 1e-4,
            "fatol": 1e-4,
            "adaptive": True
        })
    
    def test_trust_constr_all_options(self):
        """Test validation with all trust-constr options."""
        _validate_algorithm_options("trust-constr", {
            "xtol": 1e-8,
            "gtol": 1e-8,
            "barrier_tol": 1e-8,
            "initial_barrier_parameter": 0.1,
            "initial_barrier_tolerance": 0.1,
            "initial_trust_radius": 1.0,
            "max_trust_radius": 100.0,
            "maxiter": 1000,
            "disp": False
        })
    
    def test_cobyla_options(self):
        """Test validation with COBYLA options."""
        _validate_algorithm_options("COBYLA", {
            "maxiter": 1000,
            "rhobeg": 1.0,
            "tol": 1e-4
        })
    
    def test_powell_options(self):
        """Test validation with Powell options."""
        _validate_algorithm_options("Powell", {
            "xtol": 1e-4,
            "ftol": 1e-4,
            "maxfev": 10000
        })
    
    def test_unknown_algorithm_validation(self):
        """Test validation with unknown algorithm (only common options valid)."""
        _validate_algorithm_options("UnknownAlgorithm", {"maxiter": 100, "disp": True})
        
        with pytest.raises(ValueError, match="Invalid algorithm options"):
            _validate_algorithm_options("UnknownAlgorithm", {"gtol": 1e-5})  # Not a common option
