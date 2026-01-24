"""
Unit tests for constraint scaling logic in optimize_coils_loop.

These tests verify that constraint scaling is calculated correctly for force and torque
constraints. They are marked as slow integration tests because they require full
optimization setup. Run with pytest -m "not slow" to skip them.

The constraint scaling logic itself is tested more directly in other unit tests.
These tests verify the integration works end-to-end.
"""
import pytest
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries
from stellcoilbench.coil_optimization import optimize_coils_loop


def _create_minimal_coils(surface, ncoils=2, order=2):
    """Create minimal RegularizedCoil objects quickly for testing."""
    try:
        from simsopt.field import regularization_circ
    except ImportError:
        regularization_circ = None
    
    base_curves = create_equally_spaced_curves(
        ncoils, surface.nfp, stellsym=surface.stellsym,
        R0=surface.get_rc(0, 0) * 1.2, R1=0.1, order=order, numquadpoints=16  # Minimal quadpoints
    )
    base_currents = [Current(1e6) for _ in range(ncoils)]
    
    # Create regularized coils (required for LpCurveForce)
    if regularization_circ is not None:
        regularizations = [regularization_circ(0.4) for _ in range(ncoils)]
        coils = coils_via_symmetries(
            base_curves, base_currents, surface.nfp, surface.stellsym,
            regularizations=regularizations
        )
    else:
        # Fallback if regularization not available
        coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, surface.stellsym)
    
    return coils


@pytest.mark.slow
class TestConstraintScalingForceTorque:
    """Tests for force and torque constraint scaling (slow integration tests).
    
    These tests verify constraint scaling works end-to-end but are slow due to
    expensive coil initialization and field calculations. They are skipped by default.
    Run with pytest -m slow to include them.
    """
    
    def test_force_and_torque_scaling_combined(self, tmp_path):
        """Test force and torque constraint scaling setup works."""
        pytest.importorskip("simsopt")
        
        # Use absolute minimal surface for speed
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        # Create coils upfront to skip expensive initialization
        initial_coils = _create_minimal_coils(surface, ncoils=2, order=2)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Test with force and torque together - just verify setup works
        # Use absolute minimal settings and skip I/O to reduce computation time
        coils, results = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=0,  # Skip optimization, just test setup
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            initial_coils=initial_coils,  # Skip expensive initialization
            coil_objective_terms={
                "coil_coil_force": "lp",
                "coil_coil_force_p": 2,
                "coil_coil_torque": "lp",
                "coil_coil_torque_p": 2,
            },
            surface_resolution=2,  # Absolute minimal resolution
        )
        
        assert coils is not None
        assert results is not None
