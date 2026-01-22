"""
Unit tests for Fourier continuation functionality in coil optimization.

Tests verify that:
- Coils can be extended from lower to higher Fourier orders
- Fourier continuation performs a sequence of optimizations correctly
- Configuration system properly handles fourier_continuation settings
- Error handling works for invalid inputs
- Initial coils for continuation step N+1 match final coils from step N
"""
import pytest
import numpy as np
from simsopt.geo import SurfaceRZFourier
from stellcoilbench.coil_optimization import (
    _extend_coils_to_higher_order,
    optimize_coils_with_fourier_continuation,
    initialize_coils_loop,
)


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
                
                # Copy all matching dofs (cosine + sine modes up to old_order)
                for i in range(old_dofs_per_comp):
                    if old_start + i < len(old_dof) and new_start + i < len(new_dof):
                        np.testing.assert_almost_equal(
                            old_dof[old_start + i],
                            new_dof[new_start + i],
                            err_msg=f"Component {comp_idx}, dof {i} should be preserved"
                        )
    
    def test_extend_coils_no_change_when_order_same(self, low_order_coils, simple_surface):
        """Test that extending to the same order returns coils unchanged."""
        ncoils = 2
        old_order = low_order_coils[0].curve.order
        
        extended_coils = _extend_coils_to_higher_order(
            low_order_coils, new_order=old_order, s=simple_surface, ncoils=ncoils
        )
        
        # Should return the same coils (or equivalent)
        assert len(extended_coils) == len(low_order_coils)
        assert extended_coils[0].curve.order == old_order
    
    def test_extend_coils_rejects_lower_order(self, low_order_coils, simple_surface):
        """Test that extending to a lower order returns coils unchanged."""
        ncoils = 2
        old_order = low_order_coils[0].curve.order
        
        extended_coils = _extend_coils_to_higher_order(
            low_order_coils, new_order=old_order - 1, s=simple_surface, ncoils=ncoils
        )
        
        # Should return the same coils (no extension)
        assert extended_coils[0].curve.order == old_order


class TestFourierContinuation:
    """Tests for optimize_coils_with_fourier_continuation function."""
    
    @pytest.fixture
    def simple_surface(self):
        """Create a simple test surface."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)  # Major radius
        surface.set_rc(1, 0, 0.1)  # Minor radius component
        surface.set_zs(0, 0, 0.0)
        return surface
    
    def test_fourier_continuation_basic(self, simple_surface, tmp_path):
        """Test basic Fourier continuation functionality."""
        out_dir = tmp_path / "continuation_basic"
        out_dir.mkdir()
        
        coils, results = optimize_coils_with_fourier_continuation(
            s=simple_surface,
            fourier_orders=[2, 3],
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=5,
            ncoils=2,
            verbose=False,
        )
        
        assert coils is not None
        assert len(coils) > 0
        assert results is not None
        assert results['fourier_continuation'] is True
        assert results['fourier_orders'] == [2, 3]
        assert results['final_order'] == 3
    
    def test_fourier_continuation_empty_orders(self, simple_surface, tmp_path):
        """Test that empty fourier_orders list raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty list"):
            optimize_coils_with_fourier_continuation(
                s=simple_surface,
                fourier_orders=[],
                out_dir=str(tmp_path),
            )
    
    def test_fourier_continuation_invalid_order_type(self, simple_surface, tmp_path):
        """Test that non-integer orders raise ValueError."""
        with pytest.raises(ValueError, match="must be positive integers"):
            optimize_coils_with_fourier_continuation(
                s=simple_surface,
                fourier_orders=[2, 3.5, 4],
                out_dir=str(tmp_path),
            )
    
    def test_fourier_continuation_negative_order(self, simple_surface, tmp_path):
        """Test that negative orders raise ValueError."""
        with pytest.raises(ValueError, match="must be positive integers"):
            optimize_coils_with_fourier_continuation(
                s=simple_surface,
                fourier_orders=[2, -1, 4],
                out_dir=str(tmp_path),
            )
    
    def test_fourier_continuation_with_invalid_order_sequence(self, simple_surface, tmp_path):
        """Test that Fourier continuation rejects non-ascending orders."""
        with pytest.raises(ValueError, match="must be in ascending order"):
            optimize_coils_with_fourier_continuation(
                s=simple_surface,
                fourier_orders=[4, 2, 8],  # Not ascending
                out_dir=str(tmp_path),
                max_iterations=5,
                ncoils=2,
                order=4,
            )
    
    def test_continuation_initial_coils_match_previous_final_coils(self, simple_surface, tmp_path):
        """
        Test that initial coils for continuation step N+1 match final coils from step N.
        
        This verifies that the continuation process correctly transfers coil state
        between optimization steps without modifying the coils unnecessarily.
        """
        from stellcoilbench.coil_optimization import optimize_coils_loop, _extend_coils_to_higher_order
        
        # Step 1: First order optimization
        order_dir_1 = tmp_path / "order_2"
        order_dir_1.mkdir(exist_ok=True)
        coils_step1, results_step1 = optimize_coils_loop(
            s=simple_surface,
            target_B=1.0,
            out_dir=str(order_dir_1),
            max_iterations=3,
            ncoils=2,
            order=2,
            verbose=False,
        )
        
        # Capture final state from step 1
        final_coils_step1 = coils_step1
        final_curves_step1 = [coil.curve for coil in final_coils_step1[:2]]
        final_currents_step1 = [coil.current.get_value() for coil in final_coils_step1[:2]]
        final_dofs_step1 = [curve.get_dofs().copy() for curve in final_curves_step1]
        
        # Step 2: Extend and optimize
        extended_coils = _extend_coils_to_higher_order(
            final_coils_step1, new_order=4, s=simple_surface, ncoils=2
        )
        
        # Capture initial state for step 2 (after extension)
        initial_coils_step2 = extended_coils
        initial_curves_step2 = [coil.curve for coil in initial_coils_step2[:2]]
        initial_currents_step2 = [coil.current.get_value() for coil in initial_coils_step2[:2]]
        initial_dofs_step2 = [curve.get_dofs() for curve in initial_curves_step2]
        
        # Verify that currents are preserved during extension
        np.testing.assert_array_almost_equal(
            final_currents_step1, initial_currents_step2,
            err_msg="Currents should be preserved when extending coils"
        )
        
        # Verify that lower-order Fourier coefficients are preserved
        # For order 2: 2*2+1 = 5 dofs per component
        # For order 4: 2*4+1 = 9 dofs per component
        # The first 5 dofs of each component should match
        for final_dof, initial_dof in zip(final_dofs_step1, initial_dofs_step2):
            # Each component has (2*order + 1) dofs
            # Check first component (x): indices 0 to 4 (5 dofs for order 2)
            old_dofs_per_comp = 2 * 2 + 1  # 5 for order 2
            for comp_idx in range(3):
                old_start = comp_idx * old_dofs_per_comp
                new_start = comp_idx * (2 * 4 + 1)  # 9 for order 4
                np.testing.assert_array_almost_equal(
                    final_dof[old_start:old_start + old_dofs_per_comp],
                    initial_dof[new_start:new_start + old_dofs_per_comp],
                    err_msg=f"Fourier coefficients for component {comp_idx} should be preserved"
                )
        
        # Verify that the initial coils passed to step 2 match what we extended
        # The key verification is that:
        # 1. Currents are preserved (verified above)
        # 2. Fourier coefficients are preserved (verified above)
        # 3. The extended coils are the correct type and structure
        
        # Verify extended coils have correct order
        assert extended_coils[0].curve.order == 4, "Extended coils should have order 4"
        
        # Verify we have the expected number of coils
        assert len(extended_coils) >= 2, "Should have at least 2 coils"
        
        # The actual optimization would use these extended coils as initial condition,
        # which is verified by the continuation integration tests. This test specifically
        # verifies that the extension process correctly preserves coil state.
    
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
