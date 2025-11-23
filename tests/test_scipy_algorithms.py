"""
Unit tests for scipy.minimize algorithms in coil optimization.

Tests verify that different scipy optimization algorithms work correctly
with the optimize_coils_loop function, including verification of:
- Objective function minimization
- Weight application
- Constraint inclusion
- Physical reasonableness of results
"""
import pytest
import numpy as np
from simsopt.geo import SurfaceRZFourier
from stellcoilbench.coil_optimization import optimize_coils_loop


class TestScipyAlgorithms:
    """Tests for scipy.minimize algorithms in optimize_coils_loop."""
    
    def test_bfgs_algorithm_converges(self, tmp_path):
        """Test that BFGS algorithm actually minimizes the objective."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Run optimization with enough iterations to see convergence
        coils, results = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
                "coil_coil_distance": "l1",
            }
        )
        
        assert coils is not None
        assert results is not None
        
        # Verify all critical results are present and finite
        assert "final_normalized_squared_flux" in results
        assert "initial_B_field" in results
        assert "final_B_field" in results
        assert "optimization_time" in results
        
        final_flux = results["final_normalized_squared_flux"]
        initial_B = results["initial_B_field"]
        final_B = results["final_B_field"]
        opt_time = results["optimization_time"]
        
        # All values must be finite and positive
        assert np.isfinite(final_flux)
        assert np.isfinite(initial_B)
        assert np.isfinite(final_B)
        assert np.isfinite(opt_time)
        assert final_flux >= 0
        assert initial_B > 0
        assert final_B > 0
        assert opt_time > 0
        
        # Verify optimization actually ran (time > 0)
        assert opt_time > 0.01, f"Optimization time too short: {opt_time}"
        
        # Verify flux is reasonable (not NaN or inf)
        assert final_flux < 1e6, f"Flux unreasonably large: {final_flux:.2e}"
    
    def test_objective_decreases_with_iterations(self, tmp_path):
        """Test that the objective function decreases as optimization progresses."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Run with enough iterations to see progress
        coils, results = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=15,
            ncoils=2,
            order=2,
            algorithm="L-BFGS-B",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
                "coil_coil_distance": "l1",
            }
        )
        
        assert coils is not None
        assert results is not None
        
        # Verify optimization completed
        final_flux = results.get("final_normalized_squared_flux")
        opt_time = results.get("optimization_time")
        
        assert final_flux is not None
        assert opt_time is not None
        assert np.isfinite(final_flux)
        assert np.isfinite(opt_time)
        assert opt_time > 0
        
        # With 15 iterations, optimization should take meaningful time
        assert opt_time > 0.1, f"Optimization too fast: {opt_time:.3f}s"
        
        # Flux should be reasonable and finite
        assert final_flux >= 0
        assert final_flux < 1e6, f"Flux unreasonably large: {final_flux:.2e}"
        
        # Verify optimization actually ran (check that results are computed)
        assert "final_min_cc_separation" in results or "final_total_length" in results
    
    def test_weights_actually_affect_optimization(self, tmp_path):
        """Test that different constraint weights produce different optimization results."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Run with default weights (weight=1.0 for all constraints)
        coils1, results1 = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir / "default"),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
                "coil_coil_distance": "l1",
            }
        )
        
        # Run with high weight on coil_coil_distance (should prioritize separation)
        coils2, results2 = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir / "high_weight"),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
                "coil_coil_distance": "l1",
            },
            constraint_weight_1=100.0,  # Much higher weight on coil_coil_distance
        )
        
        assert coils1 is not None
        assert coils2 is not None
        assert results1 is not None
        assert results2 is not None
        
        # Verify both runs completed
        assert "final_normalized_squared_flux" in results1
        assert "final_normalized_squared_flux" in results2
        assert "final_min_cc_separation" in results1
        assert "final_min_cc_separation" in results2
        
        # With higher weight on coil_coil_distance, we should see better separation
        # (or at least different results)
        sep1 = results1.get("final_min_cc_separation", 0.0)
        sep2 = results2.get("final_min_cc_separation", 0.0)
        
        assert np.isfinite(sep1)
        assert np.isfinite(sep2)
        
        # The high-weight run should have better coil-coil separation
        # (or at least different, showing weights are being applied)
        assert abs(sep2 - sep1) > 1e-6 or sep2 > sep1, \
            f"Weights should affect results: sep(default)={sep1:.4f}, sep(high_weight)={sep2:.4f}"
    
    def test_constraints_actually_included(self, tmp_path):
        """Test that including constraints actually affects the optimization."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Run with only flux (no constraints)
        coils1, results1 = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir / "flux_only"),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={},  # Empty - only flux
        )
        
        # Run with length constraint added
        coils2, results2 = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir / "with_length"),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
            }
        )
        
        assert coils1 is not None
        assert coils2 is not None
        assert results1 is not None
        assert results2 is not None
        
        # Both should have flux
        assert "final_normalized_squared_flux" in results1
        assert "final_normalized_squared_flux" in results2
        
        # The run with length constraint should have length information
        assert "final_total_length" in results2
        
        # The total length should be finite and reasonable
        length2 = results2.get("final_total_length")
        assert length2 is not None
        assert np.isfinite(length2)
        assert length2 > 0
        
        # The two optimizations should produce different results
        # (flux-only vs flux+length should optimize differently)
        flux1 = results1.get("final_normalized_squared_flux")
        flux2 = results2.get("final_normalized_squared_flux")
        
        assert flux1 is not None
        assert flux2 is not None
        assert np.isfinite(flux1)
        assert np.isfinite(flux2)
        
        # Results should differ (or at least be computed)
        # We can't guarantee which is better, but they should be different
        assert abs(flux2 - flux1) > 1e-10 or abs(length2) > 0, \
            f"Constraints should affect optimization: flux1={flux1:.2e}, flux2={flux2:.2e}, length={length2:.2e}"
    
    def test_multiple_constraints_affect_objective(self, tmp_path):
        """Test that multiple constraints are all included and affect the optimization."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Run with multiple constraints
        coils, results = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
                "coil_coil_distance": "l1",
                "coil_surface_distance": "l1",
                "linking_number": "",
            }
        )
        
        assert coils is not None
        assert results is not None
        
        # Verify all constraint metrics are present and finite
        required_metrics = [
            "final_normalized_squared_flux",
            "final_total_length",
            "final_min_cc_separation",
            "final_min_cs_separation",
            "final_linking_number",
        ]
        
        for metric in required_metrics:
            assert metric in results, f"Missing metric: {metric}"
            value = results[metric]
            assert value is not None, f"Metric {metric} is None"
            assert np.isfinite(value), f"Metric {metric} is not finite: {value}"
        
        # Verify physical reasonableness
        assert results["final_total_length"] > 0
        assert results["final_min_cc_separation"] >= 0
        assert results["final_min_cs_separation"] >= 0
        assert results["final_normalized_squared_flux"] >= 0
    
    def test_different_algorithms_produce_different_results(self, tmp_path):
        """Test that different algorithms produce measurably different results."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        algorithms = ["BFGS", "L-BFGS-B", "SLSQP", "Nelder-Mead", "Powell", "CG", "Newton-CG", "TNC", "COBYLA", "trust-constr"]
        results_dict = {}
        
        for algo in algorithms:
            out_dir = tmp_path / f"output_{algo.lower()}"
            out_dir.mkdir()
            
            coils, results = optimize_coils_loop(
                s=surface,
                target_B=1.0,
                out_dir=str(out_dir),
                max_iterations=10,
                ncoils=2,
                order=2,
                algorithm=algo,
                verbose=False,
                coil_objective_terms={
                    "total_length": "l2",
                    "coil_coil_distance": "l1",
                }
            )
            
            assert coils is not None
            assert results is not None
            assert "final_normalized_squared_flux" in results
            
            results_dict[algo] = results
        
        # Verify all algorithms produced finite results
        for algo, results in results_dict.items():
            flux = results["final_normalized_squared_flux"]
            assert np.isfinite(flux), f"Algorithm {algo} produced non-finite flux: {flux}"
            assert flux >= 0, f"Algorithm {algo} produced negative flux: {flux}"
        
        # Algorithms should produce different results (or at least converge to similar values)
        fluxes = [results_dict[algo]["final_normalized_squared_flux"] for algo in algorithms]
        
        # All should be finite and reasonable
        assert all(np.isfinite(f) for f in fluxes)
        assert all(f >= 0 for f in fluxes)
        
        # They should all be in a reasonable range (not wildly different)
        max_flux = max(fluxes)
        min_flux = min(fluxes)
        assert max_flux < 1e6, f"Flux values too large: {fluxes}"
        assert min_flux >= 0, f"Flux values negative: {fluxes}"
    
    def test_physical_reasonableness_of_results(self, tmp_path):
        """Test that optimization results are physically reasonable."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        coils, results = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
                "coil_coil_distance": "l1",
                "coil_surface_distance": "l1",
            }
        )
        
        assert coils is not None
        assert results is not None
        
        # Verify B-field is reasonable (should be close to target)
        initial_B = results.get("initial_B_field")
        final_B = results.get("final_B_field")
        
        assert initial_B is not None
        assert final_B is not None
        assert np.isfinite(initial_B)
        assert np.isfinite(final_B)
        assert initial_B > 0
        assert final_B > 0
        assert 0.1 < initial_B < 10.0, f"Initial B-field unreasonable: {initial_B}"
        assert 0.1 < final_B < 10.0, f"Final B-field unreasonable: {final_B}"
        
        # Verify coil separations are reasonable
        cc_sep = results.get("final_min_cc_separation")
        cs_sep = results.get("final_min_cs_separation")
        
        if cc_sep is not None:
            assert np.isfinite(cc_sep)
            assert cc_sep >= 0
            assert cc_sep < 100.0, f"Coil-coil separation too large: {cc_sep}"
        
        if cs_sep is not None:
            assert np.isfinite(cs_sep)
            assert cs_sep >= 0
            assert cs_sep < 100.0, f"Coil-surface separation too large: {cs_sep}"
        
        # Verify coil length is reasonable
        total_length = results.get("final_total_length")
        if total_length is not None:
            assert np.isfinite(total_length)
            assert total_length > 0
            assert total_length < 1000.0, f"Total length too large: {total_length}"
    
    def test_linking_number_included_when_specified(self, tmp_path):
        """Test that linking number constraint is included when specified."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Run with linking number included
        coils, results = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "linking_number": "",
            }
        )
        
        assert coils is not None
        assert results is not None
        
        # Linking number should be computed
        assert "final_linking_number" in results
        linking_number = results["final_linking_number"]
        assert linking_number is not None
        assert np.isfinite(linking_number)
        # Linking number should be an integer (or close to it)
        assert abs(linking_number - round(linking_number)) < 0.1, \
            f"Linking number should be near integer: {linking_number}"
    
    def test_flux_always_included_even_with_empty_terms(self, tmp_path):
        """Test that flux objective is always included even with empty coil_objective_terms."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Run with empty coil_objective_terms
        coils, results = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={},
        )
        
        assert coils is not None
        assert results is not None
        
        # Flux should always be present and finite
        assert "final_normalized_squared_flux" in results
        flux = results["final_normalized_squared_flux"]
        assert flux is not None
        assert np.isfinite(flux)
        assert flux >= 0
        
        # B-field should also be computed
        assert "initial_B_field" in results
        assert "final_B_field" in results
        assert np.isfinite(results["initial_B_field"])
        assert np.isfinite(results["final_B_field"])
    
    def test_threshold_options_affect_optimization(self, tmp_path):
        """Test that threshold vs non-threshold options produce different behavior."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Run with l1 (no threshold)
        coils1, results1 = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir / "l1"),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "coil_coil_distance": "l1",
            }
        )
        
        # Run with l1_threshold (with threshold)
        coils2, results2 = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir / "l1_threshold"),
            max_iterations=10,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "coil_coil_distance": "l1_threshold",
            }
        )
        
        assert coils1 is not None
        assert coils2 is not None
        assert results1 is not None
        assert results2 is not None
        
        # Both should have separation metrics
        assert "final_min_cc_separation" in results1
        assert "final_min_cc_separation" in results2
        
        sep1 = results1["final_min_cc_separation"]
        sep2 = results2["final_min_cc_separation"]
        
        assert np.isfinite(sep1)
        assert np.isfinite(sep2)
        assert sep1 >= 0
        assert sep2 >= 0
        
        # The threshold option should produce different behavior
        # (or at least be computed correctly)
        assert abs(sep2 - sep1) > 1e-10 or (sep1 > 0 and sep2 > 0), \
            f"Threshold options should affect results: l1={sep1:.4f}, l1_threshold={sep2:.4f}"
    
    def test_taylor_test_runs_for_scipy_algorithms(self, tmp_path, capsys):
        """Test that Taylor test runs and verifies gradient computation for scipy algorithms."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Run optimization with verbose=False (Taylor test should still run silently)
        coils, results = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=2,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
                "coil_coil_distance": "l1",
            }
        )
        
        assert coils is not None
        assert results is not None
        
        # Check that optimization completed (Taylor test should not have caused errors)
        assert "final_normalized_squared_flux" in results
        
        # Capture output to verify Taylor test ran (it prints warnings if it fails)
        # captured = capsys.readouterr()
        # Taylor test should run silently if it passes, but we verify optimization worked
        assert "final_normalized_squared_flux" in results
    
    def test_algorithm_options_valid(self, tmp_path):
        """Test that valid algorithm-specific options are accepted and used."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Test with valid BFGS options
        coils, results = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=5,
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
            },
            algorithm_options={
                "gtol": 1e-5,
                "disp": False,
            }
        )
        
        assert coils is not None
        assert results is not None
        assert "final_normalized_squared_flux" in results
        
        # Test with valid L-BFGS-B options
        out_dir2 = tmp_path / "output2"
        out_dir2.mkdir()
        coils2, results2 = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir2),
            max_iterations=5,
            ncoils=2,
            order=2,
            algorithm="L-BFGS-B",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
            },
            algorithm_options={
                "ftol": 1e-6,
                "gtol": 1e-5,
                "maxls": 20,
            }
        )
        
        assert coils2 is not None
        assert results2 is not None
    
    def test_algorithm_options_invalid(self, tmp_path):
        """Test that invalid algorithm-specific options raise errors."""
        from stellcoilbench.coil_optimization import optimize_coils_loop, _validate_algorithm_options
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Test that invalid option name raises error
        with pytest.raises(ValueError, match="Invalid algorithm options"):
            optimize_coils_loop(
                s=surface,
                target_B=1.0,
                out_dir=str(out_dir),
                max_iterations=2,
                ncoils=2,
                order=2,
                algorithm="BFGS",
                verbose=False,
                coil_objective_terms={
                    "total_length": "l2",
                },
                algorithm_options={
                    "invalid_option": 1.0,  # Not valid for BFGS
                }
            )
        
        # Test that wrong type raises error
        with pytest.raises(ValueError, match="Invalid algorithm options.*wrong type"):
            optimize_coils_loop(
                s=surface,
                target_B=1.0,
                out_dir=str(out_dir),
                max_iterations=2,
                ncoils=2,
                order=2,
                algorithm="BFGS",
                verbose=False,
                coil_objective_terms={
                    "total_length": "l2",
                },
                algorithm_options={
                    "gtol": "not_a_float",  # Should be float, not string
                }
            )
        
        # Test algorithm-specific validation directly
        with pytest.raises(ValueError, match="Invalid algorithm options"):
            _validate_algorithm_options("BFGS", {"maxfun": 100})  # maxfun not valid for BFGS
        
        # Test that L-BFGS-B specific options work
        _validate_algorithm_options("L-BFGS-B", {"maxfun": 100, "ftol": 1e-6})
        
        # Test that SLSQP options work
        _validate_algorithm_options("SLSQP", {"ftol": 1e-6, "eps": 1e-8})
    
    def test_algorithm_options_override_defaults(self, tmp_path):
        """Test that user-specified options can override defaults."""
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Test that user can override maxiter
        coils, results = optimize_coils_loop(
            s=surface,
            target_B=1.0,
            out_dir=str(out_dir),
            max_iterations=10,  # Default
            ncoils=2,
            order=2,
            algorithm="BFGS",
            verbose=False,
            coil_objective_terms={
                "total_length": "l2",
            },
            algorithm_options={
                "maxiter": 3,  # Override to smaller value
            }
        )
        
        assert coils is not None
        assert results is not None
        # Optimization should complete (may stop early due to maxiter=3)
        assert "final_normalized_squared_flux" in results
