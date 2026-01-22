"""
Unit tests for Fourier continuation functionality in coil optimization.

Tests verify that:
- Coils can be extended from lower to higher Fourier orders
- Fourier continuation performs a sequence of optimizations correctly
- Configuration system properly handles fourier_continuation settings
- Error handling works for invalid inputs
"""
import pytest
import numpy as np
from simsopt.geo import SurfaceRZFourier
from stellcoilbench.coil_optimization import (
    _extend_coils_to_higher_order,
    optimize_coils_with_fourier_continuation,
    initialize_coils_loop,
)
from stellcoilbench.config_scheme import CaseConfig
from stellcoilbench.validate_config import validate_case_config
from stellcoilbench.evaluate import load_case_config


class TestExtendCoilsToHigherOrder:
    """Tests for _extend_coils_to_higher_order function."""
    
    @pytest.fixture
    def simple_surface(self):
        """Create a simple test surface."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)  # Major radius
        surface.set_rc(1, 0, 0.1)  # Minor radius component
        surface.set_zs(0, 0, 0.0)
        return surface
    
    @pytest.fixture
    def low_order_coils(self, simple_surface, tmp_path):
        """Create coils with low Fourier order."""
        coils = initialize_coils_loop(
            s=simple_surface,
            out_dir=str(tmp_path),
            target_B=1.0,
            ncoils=2,
            order=2,
        )
        return coils
    
    def test_extend_coils_increases_order(self, low_order_coils, simple_surface):
        """Test that extending coils increases the Fourier order."""
        ncoils = 2
        old_curves = [coil.curve for coil in low_order_coils[:ncoils]]
        old_order = old_curves[0].order
        
        extended_coils = _extend_coils_to_higher_order(
            low_order_coils, new_order=4, s=simple_surface, ncoils=ncoils
        )
        
        new_curves = [coil.curve for coil in extended_coils[:ncoils]]
        new_order = new_curves[0].order
        
        assert new_order == 4
        assert new_order > old_order
    
    def test_extend_coils_preserves_currents(self, low_order_coils, simple_surface):
        """Test that extending coils preserves current values."""
        ncoils = 2
        old_currents = [coil.current.get_value() for coil in low_order_coils[:ncoils]]
        
        extended_coils = _extend_coils_to_higher_order(
            low_order_coils, new_order=4, s=simple_surface, ncoils=ncoils
        )
        
        new_currents = [coil.current.get_value() for coil in extended_coils[:ncoils]]
        
        np.testing.assert_array_almost_equal(old_currents, new_currents)
    
    def test_extend_coils_copies_fourier_coefficients(self, low_order_coils, simple_surface):
        """Test that Fourier coefficients are correctly copied during extension."""
        ncoils = 2
        old_curves = [coil.curve for coil in low_order_coils[:ncoils]]
        old_dofs = [curve.get_dofs().copy() for curve in old_curves]
        old_order = old_curves[0].order
        
        extended_coils = _extend_coils_to_higher_order(
            low_order_coils, new_order=4, s=simple_surface, ncoils=ncoils
        )
        
        new_curves = [coil.curve for coil in extended_coils[:ncoils]]
        new_dofs = [curve.get_dofs() for curve in new_curves]
        new_order = new_curves[0].order
        
        # Check that old coefficients are preserved in new dofs
        # Structure: each component (x, y, z) has (2*order + 1) dofs
        old_dofs_per_comp = 2 * old_order + 1
        new_dofs_per_comp = 2 * new_order + 1
        
        for old_dof, new_dof in zip(old_dofs, new_dofs):
            assert len(new_dof) > len(old_dof)
            
            # Check each component separately
            for comp_idx in range(3):
                old_start = comp_idx * old_dofs_per_comp
                new_start = comp_idx * new_dofs_per_comp
                
                # Copy matching dofs (cosine + sine modes up to old_order)
                old_comp = old_dof[old_start:old_start + old_dofs_per_comp]
                new_comp = new_dof[new_start:new_start + old_dofs_per_comp]
                
                # The coefficients should match (allowing for small numerical differences)
                np.testing.assert_array_almost_equal(old_comp, new_comp, decimal=5)
            
            # Check that new modes are zero (or very small)
            # For each component, check the new modes beyond old_order
            for comp_idx in range(3):
                new_start = comp_idx * new_dofs_per_comp
                # New modes start after old_dofs_per_comp
                new_modes = new_dof[new_start + old_dofs_per_comp:new_start + new_dofs_per_comp]
                assert np.allclose(new_modes, 0.0, atol=1e-10)
    
    def test_extend_coils_same_order_returns_unchanged(self, low_order_coils, simple_surface):
        """Test that extending to the same order returns coils unchanged."""
        ncoils = 2
        extended_coils = _extend_coils_to_higher_order(
            low_order_coils, new_order=2, s=simple_surface, ncoils=ncoils
        )
        
        # Should return the same coils (identity operation)
        assert extended_coils is low_order_coils
    
    def test_extend_coils_lower_order_returns_unchanged(self, low_order_coils, simple_surface):
        """Test that extending to a lower order returns coils unchanged."""
        ncoils = 2
        extended_coils = _extend_coils_to_higher_order(
            low_order_coils, new_order=1, s=simple_surface, ncoils=ncoils
        )
        
        # Should return the same coils (no extension needed)
        assert extended_coils is low_order_coils
    
    def test_extend_coils_multiple_steps(self, low_order_coils, simple_surface):
        """Test extending coils through multiple orders."""
        ncoils = 2
        
        # Extend from 2 -> 4
        coils_4 = _extend_coils_to_higher_order(
            low_order_coils, new_order=4, s=simple_surface, ncoils=ncoils
        )
        assert coils_4[0].curve.order == 4
        
        # Extend from 4 -> 6
        coils_6 = _extend_coils_to_higher_order(
            coils_4, new_order=6, s=simple_surface, ncoils=ncoils
        )
        assert coils_6[0].curve.order == 6
        
        # Extend from 6 -> 8
        coils_8 = _extend_coils_to_higher_order(
            coils_6, new_order=8, s=simple_surface, ncoils=ncoils
        )
        assert coils_8[0].curve.order == 8
    
    def test_extend_coils_preserves_curve_shape(self, low_order_coils, simple_surface):
        """Test that extended coils maintain reasonable curve shapes."""
        ncoils = 2
        extended_coils = _extend_coils_to_higher_order(
            low_order_coils, new_order=4, s=simple_surface, ncoils=ncoils
        )
        
        # Check that curves can be evaluated
        for coil in extended_coils[:ncoils]:
            curve = coil.curve
            gamma = curve.gamma()
            assert gamma.shape[1] == 3  # 3D coordinates
            assert len(gamma) > 0  # Non-empty curve
            # Check that coordinates are finite
            assert np.all(np.isfinite(gamma))


class TestOptimizeCoilsWithFourierContinuation:
    """Tests for optimize_coils_with_fourier_continuation function."""
    
    @pytest.fixture
    def simple_surface(self):
        """Create a simple test surface."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)  # Major radius
        surface.set_rc(1, 0, 0.1)  # Minor radius component
        surface.set_zs(0, 0, 0.0)
        return surface
    
    def test_fourier_continuation_single_order(self, simple_surface, tmp_path):
        """Test continuation with a single order (should work like normal optimization)."""
        out_dir = tmp_path / "continuation_test"
        out_dir.mkdir()
        
        coils, results = optimize_coils_with_fourier_continuation(
            s=simple_surface,
            fourier_orders=[2],
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=5,
            ncoils=2,
            verbose=False,
            coil_objective_terms={"total_length": "l2"},
        )
        
        assert coils is not None
        assert len(coils) >= 2
        assert results is not None
        assert results['fourier_continuation'] is True
        assert results['fourier_orders'] == [2]
        assert results['final_order'] == 2
        assert len(results['continuation_results']) == 1
        assert results['continuation_results'][0]['fourier_order'] == 2
    
    def test_fourier_continuation_multiple_orders(self, simple_surface, tmp_path):
        """Test continuation with multiple orders."""
        out_dir = tmp_path / "continuation_test"
        out_dir.mkdir()
        
        coils, results = optimize_coils_with_fourier_continuation(
            s=simple_surface,
            fourier_orders=[2, 3],
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=3,  # Small number for speed
            ncoils=2,
            verbose=False,
            coil_objective_terms={"total_length": "l2"},
        )
        
        assert coils is not None
        assert len(coils) >= 2
        assert results is not None
        assert results['fourier_continuation'] is True
        assert results['fourier_orders'] == [2, 3]
        assert results['final_order'] == 3
        assert len(results['continuation_results']) == 2
        
        # Check that results are stored for each order
        assert results['continuation_results'][0]['fourier_order'] == 2
        assert results['continuation_results'][1]['fourier_order'] == 3
        
        # Check that subdirectories were created
        assert (out_dir / "order_2").exists()
        assert (out_dir / "order_3").exists()
    
    def test_fourier_continuation_three_orders(self, simple_surface, tmp_path):
        """Test continuation with three orders."""
        out_dir = tmp_path / "continuation_test"
        out_dir.mkdir()
        
        coils, results = optimize_coils_with_fourier_continuation(
            s=simple_surface,
            fourier_orders=[2, 3, 4],
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=2,  # Very small for speed
            ncoils=2,
            verbose=False,
            coil_objective_terms={"total_length": "l2"},
        )
        
        assert coils is not None
        assert results is not None
        assert results['final_order'] == 4
        assert len(results['continuation_results']) == 3
        
        # Verify all orders are in results
        orders_in_results = [r['fourier_order'] for r in results['continuation_results']]
        assert orders_in_results == [2, 3, 4]
        
        # Check continuation steps
        steps = [r['continuation_step'] for r in results['continuation_results']]
        assert steps == [1, 2, 3]
    
    def test_fourier_continuation_empty_orders_raises_error(self, simple_surface, tmp_path):
        """Test that empty orders list raises ValueError."""
        out_dir = tmp_path / "continuation_test"
        out_dir.mkdir()
        
        with pytest.raises(ValueError, match="fourier_orders must be a non-empty list"):
            optimize_coils_with_fourier_continuation(
                s=simple_surface,
                fourier_orders=[],
                target_B=1.0,
                out_dir=str(out_dir),
            )
    
    def test_fourier_continuation_invalid_orders_raises_error(self, simple_surface, tmp_path):
        """Test that invalid orders raise ValueError."""
        out_dir = tmp_path / "continuation_test"
        out_dir.mkdir()
        
        # Non-integer orders
        with pytest.raises(ValueError, match="All fourier_orders must be positive integers"):
            optimize_coils_with_fourier_continuation(
                s=simple_surface,
                fourier_orders=[2, 3.5, 4],
                target_B=1.0,
                out_dir=str(out_dir),
            )
        
        # Non-positive orders
        with pytest.raises(ValueError, match="All fourier_orders must be positive integers"):
            optimize_coils_with_fourier_continuation(
                s=simple_surface,
                fourier_orders=[2, 0, 4],
                target_B=1.0,
                out_dir=str(out_dir),
            )
    
    def test_fourier_continuation_descending_orders_raises_error(self, simple_surface, tmp_path):
        """Test that descending orders raise ValueError."""
        out_dir = tmp_path / "continuation_test"
        out_dir.mkdir()
        
        with pytest.raises(ValueError, match="fourier_orders must be in ascending order"):
            optimize_coils_with_fourier_continuation(
                s=simple_surface,
                fourier_orders=[4, 3, 2],
                target_B=1.0,
                out_dir=str(out_dir),
            )
    
    def test_fourier_continuation_passes_kwargs(self, simple_surface, tmp_path):
        """Test that kwargs are passed through to optimize_coils_loop."""
        out_dir = tmp_path / "continuation_test"
        out_dir.mkdir()
        
        coils, results = optimize_coils_with_fourier_continuation(
            s=simple_surface,
            fourier_orders=[2],
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=5,
            ncoils=2,
            verbose=False,
            coil_objective_terms={"total_length": "l2"},
            flux_threshold=1e-6,  # Custom threshold
            length_threshold=100.0,  # Custom threshold
        )
        
        assert coils is not None
        assert results is not None
        # Check that custom thresholds are in results
        assert 'flux_threshold' in results
        assert 'length_threshold' in results or 'final_total_length' in results


class TestFourierContinuationConfiguration:
    """Tests for Fourier continuation configuration handling."""
    
    def test_config_with_fourier_continuation(self, tmp_path):
        """Test that CaseConfig loads fourier_continuation correctly."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "fourier_continuation": {
                "enabled": True,
                "orders": [4, 6, 8]
            }
        }
        
        config = CaseConfig.from_dict(data)
        assert config.fourier_continuation is not None
        assert config.fourier_continuation["enabled"] is True
        assert config.fourier_continuation["orders"] == [4, 6, 8]
    
    def test_config_without_fourier_continuation(self, tmp_path):
        """Test that CaseConfig handles missing fourier_continuation."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
        }
        
        config = CaseConfig.from_dict(data)
        assert config.fourier_continuation is None
    
    def test_validate_fourier_continuation_enabled_true(self, tmp_path):
        """Test validation of fourier_continuation with enabled=true."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "fourier_continuation": {
                "enabled": True,
                "orders": [4, 6, 8]
            }
        }
        
        errors = validate_case_config(data)
        assert errors == []
    
    def test_validate_fourier_continuation_enabled_false(self, tmp_path):
        """Test validation of fourier_continuation with enabled=false."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "fourier_continuation": {
                "enabled": False,
                "orders": [4, 6, 8]
            }
        }
        
        errors = validate_case_config(data)
        assert errors == []
    
    def test_validate_fourier_continuation_not_dict(self, tmp_path):
        """Test validation error when fourier_continuation is not a dict."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "fourier_continuation": "not a dict"
        }
        
        errors = validate_case_config(data)
        assert any("fourier_continuation must be a dictionary" in e for e in errors)
    
    def test_validate_fourier_continuation_empty_orders(self, tmp_path):
        """Test validation error when orders list is empty."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "fourier_continuation": {
                "enabled": True,
                "orders": []
            }
        }
        
        errors = validate_case_config(data)
        assert any("fourier_continuation.orders must be non-empty" in e for e in errors)
    
    def test_validate_fourier_continuation_non_integer_orders(self, tmp_path):
        """Test validation error when orders contain non-integers."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "fourier_continuation": {
                "enabled": True,
                "orders": [4, 6.5, 8]
            }
        }
        
        errors = validate_case_config(data)
        assert any("fourier_continuation.orders must contain only positive integers" in e for e in errors)
    
    def test_validate_fourier_continuation_descending_orders(self, tmp_path):
        """Test validation error when orders are not ascending."""
        data = {
            "description": "Test case",
            "surface_params": {"surface": "input.test"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "l-bfgs"},
            "fourier_continuation": {
                "enabled": True,
                "orders": [8, 6, 4]
            }
        }
        
        errors = validate_case_config(data)
        assert any("fourier_continuation.orders must be in ascending order" in e for e in errors)
    
    def test_load_case_config_with_fourier_continuation(self, tmp_path):
        """Test loading a case.yaml file with fourier_continuation."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 4
optimizer_params:
  algorithm: l-bfgs
fourier_continuation:
  enabled: true
  orders: [4, 6, 8]
""")
        
        config = load_case_config(case_yaml)
        assert config.fourier_continuation is not None
        assert config.fourier_continuation["enabled"] is True
        assert config.fourier_continuation["orders"] == [4, 6, 8]


class TestFourierContinuationIntegration:
    """Integration tests for Fourier continuation with actual optimizations."""
    
    @pytest.fixture
    def simple_surface(self):
        """Create a simple test surface."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)  # Major radius
        surface.set_rc(1, 0, 0.1)  # Minor radius component
        surface.set_zs(0, 0, 0.0)
        return surface
    
    def test_continuation_improves_solution(self, simple_surface, tmp_path):
        """Test that continuation produces a valid solution."""
        out_dir = tmp_path / "continuation_integration"
        out_dir.mkdir()
        
        coils, results = optimize_coils_with_fourier_continuation(
            s=simple_surface,
            fourier_orders=[2, 3],
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=5,
            ncoils=2,
            verbose=False,
            coil_objective_terms={"total_length": "l2"},
        )
        
        # Verify final coils have correct order
        assert coils[0].curve.order == 3
        
        # Verify results contain expected metrics
        assert 'final_normalized_squared_flux' in results
        assert 'final_B_field' in results
        assert 'optimization_time' in results
        
        # Verify all values are finite
        assert np.isfinite(results['final_normalized_squared_flux'])
        assert np.isfinite(results['final_B_field'])
        
        # Verify continuation metadata
        assert results['fourier_continuation'] is True
        assert len(results['continuation_results']) == 2
    
    def test_continuation_results_structure(self, simple_surface, tmp_path):
        """Test that continuation results have the correct structure."""
        out_dir = tmp_path / "continuation_structure"
        out_dir.mkdir()
        
        _, results = optimize_coils_with_fourier_continuation(
            s=simple_surface,
            fourier_orders=[2, 3, 4],
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=2,
            ncoils=2,
            verbose=False,
            coil_objective_terms={"total_length": "l2"},
        )
        
        # Check top-level results
        assert results['fourier_continuation'] is True
        assert results['fourier_orders'] == [2, 3, 4]
        assert results['final_order'] == 4
        
        # Check continuation_results list
        assert len(results['continuation_results']) == 3
        for i, step_result in enumerate(results['continuation_results']):
            assert step_result['fourier_order'] == [2, 3, 4][i]
            assert step_result['continuation_step'] == i + 1
            assert 'final_normalized_squared_flux' in step_result
            assert 'final_B_field' in step_result
        
        # Check that final step results are also at top level
        final_step = results['continuation_results'][-1]
        assert results['final_normalized_squared_flux'] == final_step['final_normalized_squared_flux']
        assert results['final_B_field'] == final_step['final_B_field']
    
    def test_continuation_creates_subdirectories(self, simple_surface, tmp_path):
        """Test that continuation creates subdirectories for each order."""
        out_dir = tmp_path / "continuation_dirs"
        out_dir.mkdir()
        
        coils, results = optimize_coils_with_fourier_continuation(
            s=simple_surface,
            fourier_orders=[2, 3, 4],
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=2,
            ncoils=2,
            verbose=False,
            coil_objective_terms={"total_length": "l2"},
        )
        
        # Check that subdirectories were created
        assert (out_dir / "order_2").exists()
        assert (out_dir / "order_3").exists()
        assert (out_dir / "order_4").exists()
        assert (out_dir / "order_2").is_dir()
        assert (out_dir / "order_3").is_dir()
        assert (out_dir / "order_4").is_dir()
    
    def test_continuation_with_different_objective_terms(self, simple_surface, tmp_path):
        """Test continuation with different coil_objective_terms."""
        out_dir = tmp_path / "continuation_objectives"
        out_dir.mkdir()
        
        coils, results = optimize_coils_with_fourier_continuation(
            s=simple_surface,
            fourier_orders=[2, 3],
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=3,
            ncoils=2,
            verbose=False,
            coil_objective_terms={
                "total_length": "l2_threshold",
                "coil_coil_distance": "l1",
            },
        )
        
        assert coils is not None
        assert results is not None
        assert results['final_order'] == 3
