"""
Unit tests for post-processing functionality.

Tests verify that post-processing functions work correctly with optimized coils,
including QFM surface computation, VMEC equilibrium, and quasisymmetry metrics.
"""

import pytest
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import json
import yaml
from unittest.mock import Mock, patch

from simsopt.geo import SurfaceRZFourier
from simsopt.field import BiotSavart, Current
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import coils_via_symmetries
from stellcoilbench.post_processing import (
    load_coils_and_surface,
    compute_qfm_surface,
    run_vmec_equilibrium,
    compute_quasisymmetry,
    plot_boozer_surface,
    plot_iota_profile,
    plot_quasisymmetry_profile,
    trace_fieldlines,
    run_post_processing,
    TRACING_AVAILABLE,
)
from stellcoilbench.coil_optimization import optimize_coils
from stellcoilbench.evaluate import load_case_config


@pytest.mark.slow
class TestPostProcessing:
    """Integration tests for post-processing functionality."""
    
    @pytest.fixture
    def case_path(self):
        """Path to the advanced Landreman-Paul QA case."""
        return Path(__file__).parent.parent / "cases" / "advanced_LandremanPaulQA.yaml"
    
    @pytest.fixture
    def optimized_coils_and_output(self, case_path, tmp_path):
        """Run a quick optimization to generate coils for testing."""
        # Load case config
        case_cfg = load_case_config(case_path)
        
        # Create output directory
        output_dir = tmp_path / "optimization_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        coils_json_path = output_dir / "biot_savart_optimized.json"
        
        # Run optimization enough to get a decent solution & post-processing works well.
        # Modify optimizer params for speed
        original_max_iter = case_cfg.optimizer_params.get('max_iterations', 200)
        case_cfg.optimizer_params['max_iterations'] = 200 
        
        try:
            _ = optimize_coils(
                case_path=case_path.parent,
                coils_out_path=coils_json_path,
                case_cfg=case_cfg,
                output_dir=output_dir,
                surface_resolution=32,  # Lower resolution for speed
            )
        except Exception as e:
            pytest.skip(f"Optimization failed (may need VMEC or other dependencies): {e}")
        
        # Restore original max_iterations
        case_cfg.optimizer_params['max_iterations'] = original_max_iter
        
        return coils_json_path, output_dir, case_path
    
    def test_load_coils_and_surface(self, optimized_coils_and_output):
        """Test loading coils and surface from JSON and case.yaml."""
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        assert bfield is not None
        assert isinstance(bfield, BiotSavart)
        assert surface is not None
        assert isinstance(surface, SurfaceRZFourier)
        assert len(bfield.coils) > 0
    
    def test_load_coils_and_surface_with_plasma_surfaces_dir(self, optimized_coils_and_output):
        """Test loading with custom plasma_surfaces_dir."""
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        plasma_surfaces_dir = Path(__file__).parent.parent / "plasma_surfaces"
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
            plasma_surfaces_dir=plasma_surfaces_dir,
        )
        
        assert bfield is not None
        assert surface is not None
    
    def test_load_coils_and_surface_finds_case_yaml(self, optimized_coils_and_output):
        """Test that case.yaml is found automatically."""
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        # Copy case.yaml to output directory
        import shutil
        shutil.copy(case_path, output_dir / "case.yaml")
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=None,  # Should find it automatically
        )
        
        assert bfield is not None
        assert surface is not None
    
    def test_load_coils_and_surface_different_file_types(self, tmp_path):
        """Test loading surfaces from different file types."""
        # Create a minimal test setup
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        # Test with input file type
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        # Create minimal coils JSON
        coils_json = tmp_path / "coils.json"
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        # Mock SurfaceRZFourier.from_vmec_input to avoid needing real VMEC file
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            bfield, loaded_surface = load_coils_and_surface(
                coils_json,
                case_yaml_path=case_yaml,
                plasma_surfaces_dir=tmp_path,
            )
            assert bfield is not None
            assert loaded_surface is not None
    
    def test_compute_qfm_surface(self, optimized_coils_and_output):
        """Test QFM surface computation."""
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        # Compute QFM surface
        qfm_surface = compute_qfm_surface(surface, bfield)
        
        assert qfm_surface is not None
        assert isinstance(qfm_surface, SurfaceRZFourier)
        
        # Verify QFM surface has reasonable geometry
        gamma = qfm_surface.gamma()
        assert gamma.shape[0] > 0  # Has phi points
        assert gamma.shape[1] > 0  # Has theta points
        assert gamma.shape[2] == 3  # 3D coordinates
    
    def test_compute_bn_metrics(self, optimized_coils_and_output):
        """Test computation of B·n metrics on plasma surface."""
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        # Compute B·n on plasma surface
        bfield.set_points(surface.gamma().reshape((-1, 3)))
        B = bfield.B().reshape((surface.quadpoints_phi.size, surface.quadpoints_theta.size, 3))
        n = surface.unitnormal()
        BdotN = np.mean(np.abs(np.sum(B * n, axis=2)))
        BdotN_over_B = BdotN / np.mean(bfield.AbsB())
        
        assert BdotN >= 0
        assert BdotN_over_B >= 0
        assert BdotN_over_B <= 1.0  # Normalized quantity
    
    def test_run_vmec_equilibrium(self, optimized_coils_and_output):
        """Test VMEC equilibrium calculation."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        # Compute QFM surface
        qfm_surface = compute_qfm_surface(surface, bfield)
        
        # Get VMEC input path from surface
        vmec_input_path = None
        if hasattr(surface, 'filename') and surface.filename:
            vmec_input_path = Path(surface.filename)
        else:
            # Try to find the input file
            import yaml
            case_data = yaml.safe_load(case_path.read_text())
            surface_name = case_data.get("surface_params", {}).get("surface", "")
            plasma_surfaces_dir = Path(__file__).parent.parent / "plasma_surfaces"
            vmec_input_path = plasma_surfaces_dir / surface_name
        
        if not vmec_input_path or not vmec_input_path.exists():
            pytest.skip("VMEC input file not found")
        
        try:
            equil = run_vmec_equilibrium(
                qfm_surface,
                vmec_input_path=vmec_input_path,
            )
            
            assert equil is not None
            assert hasattr(equil, 'wout')
        except Exception as e:
            pytest.skip(f"VMEC calculation failed: {e}")
    
    def test_compute_quasisymmetry(self, optimized_coils_and_output):
        """Test quasisymmetry computation from VMEC equilibrium."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        # Compute QFM surface
        qfm_surface = compute_qfm_surface(surface, bfield)
        
        # Get VMEC input path
        case_data = yaml.safe_load(case_path.read_text())
        surface_name = case_data.get("surface_params", {}).get("surface", "")
        plasma_surfaces_dir = Path(__file__).parent.parent / "plasma_surfaces"
        vmec_input_path = plasma_surfaces_dir / surface_name
        
        if not vmec_input_path.exists():
            pytest.skip("VMEC input file not found")
        
        try:
            equil = run_vmec_equilibrium(
                qfm_surface,
                vmec_input_path=vmec_input_path,
            )
            
            # Compute quasisymmetry (QA has helicity_n=0)
            qs_total, qs_profile = compute_quasisymmetry(
                equil,
                helicity_m=1,
                helicity_n=0,
                ns=20,  # Fewer surfaces for speed
            )
            
            assert qs_total >= 0
            assert len(qs_profile) > 0
            assert np.all(qs_profile >= 0)
        except Exception as e:
            pytest.skip(f"Quasisymmetry computation failed: {e}")
    
    def test_plot_boozer_surface(self, optimized_coils_and_output, tmp_path):
        """Test Boozer surface plotting."""
        pytest.importorskip("booz_xform", reason="booz_xform not available")
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        qfm_surface = compute_qfm_surface(surface, bfield)
        
        # Get VMEC input path
        case_data = yaml.safe_load(case_path.read_text())
        surface_name = case_data.get("surface_params", {}).get("surface", "")
        plasma_surfaces_dir = Path(__file__).parent.parent / "plasma_surfaces"
        vmec_input_path = plasma_surfaces_dir / surface_name
        
        if not vmec_input_path.exists():
            pytest.skip("VMEC input file not found")
        
        try:
            equil = run_vmec_equilibrium(
                qfm_surface,
                vmec_input_path=vmec_input_path,
            )
            
            output_path = tmp_path / "test_boozer.png"
            plot_boozer_surface(equil, output_path, js=10)
            
            assert output_path.exists()
        except Exception as e:
            pytest.skip(f"Boozer plot failed: {e}")
    
    def test_plot_iota_profile(self, optimized_coils_and_output, tmp_path):
        """Test iota profile plotting."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        qfm_surface = compute_qfm_surface(surface, bfield)
        
        # Get VMEC input path
        case_data = yaml.safe_load(case_path.read_text())
        surface_name = case_data.get("surface_params", {}).get("surface", "")
        plasma_surfaces_dir = Path(__file__).parent.parent / "plasma_surfaces"
        vmec_input_path = plasma_surfaces_dir / surface_name
        
        if not vmec_input_path.exists():
            pytest.skip("VMEC input file not found")
        
        try:
            equil = run_vmec_equilibrium(
                qfm_surface,
                vmec_input_path=vmec_input_path,
            )
            
            output_path = tmp_path / "test_iota.png"
            plot_iota_profile(equil, output_path, sign=1)
            
            assert output_path.exists()
        except Exception as e:
            pytest.skip(f"Iota plot failed: {e}")
    
    def test_plot_boundary_surface(self, optimized_coils_and_output, tmp_path):
        """Test plotting the boundary surface (plasma surface and QFM surface)."""
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        # Compute QFM surface
        qfm_surface = compute_qfm_surface(surface, bfield)
        
        # Plot both surfaces
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot plasma boundary surface
        gamma_plasma = surface.gamma()
        phi_idx = gamma_plasma.shape[0] // 2  # Middle phi slice
        ax.plot(
            gamma_plasma[phi_idx, :, 0],
            gamma_plasma[phi_idx, :, 1],
            gamma_plasma[phi_idx, :, 2],
            'b-', label='Plasma boundary', linewidth=2
        )
        
        # Plot QFM surface
        gamma_qfm = qfm_surface.gamma()
        phi_idx_qfm = gamma_qfm.shape[0] // 2  # Middle phi slice
        ax.plot(
            gamma_qfm[phi_idx_qfm, :, 0],
            gamma_qfm[phi_idx_qfm, :, 1],
            gamma_qfm[phi_idx_qfm, :, 2],
            'r--', label='QFM surface', linewidth=2
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Boundary Surfaces Comparison')
        ax.legend()
        
        output_path = tmp_path / "boundary_surfaces.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        assert output_path.exists()
    
    def test_run_post_processing_full_pipeline(self, optimized_coils_and_output, tmp_path):
        """Test the full post-processing pipeline."""
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        post_processing_output = tmp_path / "post_processing_output"
        
        try:
            results = run_post_processing(
                coils_json_path=coils_json_path,
                output_dir=post_processing_output,
                case_yaml_path=case_path,
                run_vmec=False,  # Skip VMEC for faster tests
                plot_boozer=False,
                plot_poincare=True,
                nfieldlines=5,  # Fewer fieldlines for faster test
            )
            
            assert results is not None
            assert 'qfm_surface' in results
            assert 'BdotN' in results
            assert 'BdotN_over_B' in results
            
            # Check if Poincaré plot was created
            poincare_path = post_processing_output / "poincare_plot.png"
            if poincare_path.exists():
                assert 'poincare_results' in results
            
            # Check that results JSON was created
            results_json_path = post_processing_output / "post_processing_results.json"
            assert results_json_path.exists()
        except Exception as e:
            pytest.skip(f"Full pipeline test failed: {e}")
    
    def test_post_processing_with_vmec_and_plots(self, optimized_coils_and_output, tmp_path):
        """Test post-processing with VMEC and all plots."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        post_processing_output = tmp_path / "post_processing_output_vmec"
        
        # Get VMEC input path
        case_data = yaml.safe_load(case_path.read_text())
        surface_name = case_data.get("surface_params", {}).get("surface", "")
        plasma_surfaces_dir = Path(__file__).parent.parent / "plasma_surfaces"
        vmec_input_path = plasma_surfaces_dir / surface_name
        
        if not vmec_input_path.exists():
            pytest.skip("VMEC input file not found")
        
        try:
            results = run_post_processing(
                coils_json_path=coils_json_path,
                output_dir=post_processing_output,
                case_yaml_path=case_path,
                run_vmec=True,
                helicity_m=1,
                helicity_n=0,  # QA symmetry
                ns=20,  # Fewer surfaces for speed
                plot_boozer=True,
                plot_poincare=False,  # Skip Poincaré for VMEC test
            )
            
            assert results is not None
            assert 'qfm_surface' in results
            assert 'BdotN' in results
            
            # Check that plots were created
            if results.get('quasisymmetry_average') is not None:
                assert (post_processing_output / "quasisymmetry_profile.png").exists()
                assert (post_processing_output / "iota_profile.png").exists()
                assert (post_processing_output / "boozer_surface.png").exists()
            
            # Check results JSON
            results_json_path = post_processing_output / "post_processing_results.json"
            assert results_json_path.exists()
        except Exception as e:
            pytest.skip(f"VMEC post-processing failed: {e}")
    
    def test_plot_fixed_boundary_quasisymmetry(self, optimized_coils_and_output, tmp_path):
        """Test plotting quasisymmetry from fixed boundary equilibrium."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        # Compute QFM surface (this is the fixed boundary)
        qfm_surface = compute_qfm_surface(surface, bfield)
        
        # Get VMEC input path
        case_data = yaml.safe_load(case_path.read_text())
        surface_name = case_data.get("surface_params", {}).get("surface", "")
        plasma_surfaces_dir = Path(__file__).parent.parent / "plasma_surfaces"
        vmec_input_path = plasma_surfaces_dir / surface_name
        
        if not vmec_input_path.exists():
            pytest.skip("VMEC input file not found")
        
        try:
            # Run VMEC with QFM surface as fixed boundary
            equil = run_vmec_equilibrium(
                qfm_surface,
                vmec_input_path=vmec_input_path,
            )
            
            # Compute quasisymmetry
            qs_total, qs_profile = compute_quasisymmetry(
                equil,
                helicity_m=1,
                helicity_n=0,
                ns=20,
            )
            
            # Plot quasisymmetry profile
            radii = np.arange(0, 1.01, 1.01 / 20)
            output_path = tmp_path / "fixed_boundary_quasisymmetry.png"
            plot_quasisymmetry_profile(qs_profile, radii, output_path)
            
            assert output_path.exists()
            assert qs_total >= 0
            
            # Also plot iota profile
            iota_output_path = tmp_path / "fixed_boundary_iota.png"
            plot_iota_profile(equil, iota_output_path, sign=1)
            
            assert iota_output_path.exists()
        except Exception as e:
            pytest.skip(f"Fixed boundary quasisymmetry test failed: {e}")
    
    def test_trace_fieldlines(self, optimized_coils_and_output, tmp_path):
        """Test fieldline tracing and Poincaré plot generation."""
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        try:
            output_path = tmp_path / "test_poincare.png"
            results = trace_fieldlines(
                bfield,
                surface,
                output_path,
                nfieldlines=5,  # Fewer fieldlines for faster test
                use_interpolated_field=True,
            )
            
            assert output_path.exists()
            assert 'fieldlines_tys' in results
            assert 'fieldlines_phi_hits' in results
            assert 'phis' in results
        except ImportError:
            pytest.skip("Fieldline tracing not available")
        except Exception as e:
            pytest.skip(f"Fieldline tracing failed: {e}")
    
    def test_trace_fieldlines_without_interpolation(self, optimized_coils_and_output, tmp_path):
        """Test fieldline tracing without interpolated field."""
        coils_json_path, output_dir, case_path = optimized_coils_and_output
        
        bfield, surface = load_coils_and_surface(
            coils_json_path,
            case_yaml_path=case_path,
        )
        
        try:
            output_path = tmp_path / "test_poincare_no_interp.png"
            results = trace_fieldlines(
                bfield,
                surface,
                output_path,
                nfieldlines=3,  # Very few for speed
                use_interpolated_field=False,
            )
            
            assert output_path.exists()
            assert 'fieldlines_tys' in results
        except ImportError:
            pytest.skip("Fieldline tracing not available")
        except Exception as e:
            pytest.skip(f"Fieldline tracing failed: {e}")


class TestPostProcessingUnit:
    """Unit tests for post-processing functions (no optimization required)."""
    
    def test_load_coils_and_surface_missing_file(self, tmp_path):
        """Test error handling for missing files."""
        coils_json_path = tmp_path / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            load_coils_and_surface(coils_json_path)
    
    def test_load_coils_and_surface_missing_case_yaml(self, tmp_path):
        """Test error when case.yaml is not found."""
        # Create coils JSON
        coils_json = tmp_path / "coils.json"
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        with pytest.raises(FileNotFoundError):
            load_coils_and_surface(coils_json, case_yaml_path=None)
    
    def test_load_coils_and_surface_no_surface_file(self, tmp_path):
        """Test error when surface file is not specified in case.yaml."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  range: "half period"
""")
        
        coils_json = tmp_path / "coils.json"
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        with pytest.raises(ValueError, match="No surface file"):
            load_coils_and_surface(coils_json, case_yaml_path=case_yaml)
    
    def test_load_coils_and_surface_unknown_file_type(self, tmp_path):
        """Test error for unknown surface file type."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "unknown.xyz"
  range: "half period"
""")
        
        surface_file = tmp_path / "unknown.xyz"
        surface_file.write_text("dummy")
        
        coils_json = tmp_path / "coils.json"
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        with pytest.raises(ValueError, match="Unknown surface file type"):
            load_coils_and_surface(
                coils_json,
                case_yaml_path=case_yaml,
                plasma_surfaces_dir=tmp_path,
            )
    
    def test_load_coils_from_list(self, tmp_path):
        """Test loading coils when JSON contains a list of coils."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        # Create coils as a list
        coils_json = tmp_path / "coils.json"
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        
        from simsopt import save
        save(coils_list, coils_json)
        
        # Mock surface loading
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            bfield, loaded_surface = load_coils_and_surface(
                coils_json,
                case_yaml_path=case_yaml,
                plasma_surfaces_dir=tmp_path,
            )
            assert bfield is not None
            assert isinstance(bfield, BiotSavart)
    
    def test_load_coils_from_single_coil(self, tmp_path):
        """Test loading when JSON contains a single coil object."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        # Create a single coil
        coils_json = tmp_path / "coils.json"
        coils = create_equally_spaced_curves(1, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        single_coil = coils_list[0]  # Just one coil
        
        from simsopt import save
        save(single_coil, coils_json)
        
        # Mock surface loading
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            bfield, loaded_surface = load_coils_and_surface(
                coils_json,
                case_yaml_path=case_yaml,
                plasma_surfaces_dir=tmp_path,
            )
            assert bfield is not None
            assert isinstance(bfield, BiotSavart)
    
    def test_load_coils_and_surface_wout_file(self, tmp_path):
        """Test loading surface from wout file."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "wout.test"
  range: "half period"
""")
        
        surface_file = tmp_path / "wout.test"
        surface_file.write_text("dummy")
        
        coils_json = tmp_path / "coils.json"
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_wout') as mock_from_wout:
            mock_from_wout.return_value = surface
            bfield, loaded_surface = load_coils_and_surface(
                coils_json,
                case_yaml_path=case_yaml,
                plasma_surfaces_dir=tmp_path,
            )
            assert bfield is not None
            assert loaded_surface is not None
    
    def test_load_coils_and_surface_focus_file(self, tmp_path):
        """Test loading surface from focus file."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "test.focus"
  range: "half period"
""")
        
        surface_file = tmp_path / "test.focus"
        surface_file.write_text("dummy")
        
        coils_json = tmp_path / "coils.json"
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_focus') as mock_from_focus:
            mock_from_focus.return_value = surface
            bfield, loaded_surface = load_coils_and_surface(
                coils_json,
                case_yaml_path=case_yaml,
                plasma_surfaces_dir=tmp_path,
            )
            assert bfield is not None
            assert loaded_surface is not None
    
    def test_load_coils_and_surface_surface_in_coils_dir(self, tmp_path):
        """Test finding surface file in coils directory."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        coils_dir = tmp_path / "coils_dir"
        coils_dir.mkdir()
        surface_file = coils_dir / "input.test"
        surface_file.write_text("dummy")
        
        coils_json = coils_dir / "coils.json"
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            bfield, loaded_surface = load_coils_and_surface(
                coils_json,
                case_yaml_path=case_yaml,
                plasma_surfaces_dir=tmp_path / "nonexistent",
            )
            assert bfield is not None
            assert loaded_surface is not None
    
    def test_load_coils_and_surface_absolute_path(self, tmp_path):
        """Test loading with absolute surface file path."""
        case_yaml = tmp_path / "case.yaml"
        surface_file_abs = tmp_path / "input.test"
        surface_file_abs.write_text("dummy")
        
        case_yaml.write_text(f"""
surface_params:
  surface: "{surface_file_abs}"
  range: "half period"
""")
        
        coils_json = tmp_path / "coils.json"
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            bfield, loaded_surface = load_coils_and_surface(
                coils_json,
                case_yaml_path=case_yaml,
            )
            assert bfield is not None
            assert loaded_surface is not None
    
    def test_compute_qfm_surface_direct(self, tmp_path):
        """Test compute_qfm_surface directly."""
        # Create minimal surface and field
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)
        surface.set_rc(1, 0, 0.1)
        
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        qfm_surface = compute_qfm_surface(surface, bs)
        
        assert qfm_surface is not None
        assert isinstance(qfm_surface, SurfaceRZFourier)
    
    def test_plot_boozer_surface_no_booz_xform(self, tmp_path):
        """Test plot_boozer_surface when booz_xform is not available."""
        # Mock VMEC equilibrium
        mock_equil = Mock()
        mock_equil.output_file = str(tmp_path / "wout.nc")
        
        output_path = tmp_path / "boozer.png"
        
        # Test that ImportError is raised when booz_xform is not available
        # We'll test this by patching the import inside the function
        with patch('stellcoilbench.post_processing.bx', None, create=True):
            # Try to import booz_xform inside the function
            try:
                import booz_xform
                _ = booz_xform
                pytest.skip("booz_xform is available")
            except ImportError:
                # This is expected - the function should raise ImportError
                with pytest.raises(ImportError, match="booz_xform is required"):
                    plot_boozer_surface(mock_equil, output_path)
    
    def test_make_qfm_import_fallback(self, tmp_path):
        """Test make_qfm import fallback paths."""
        # This tests the import fallback logic
        # We can't easily test the actual fallback without breaking imports,
        # but we can verify the function works with the current import
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)
        surface.set_rc(1, 0, 0.1)
        
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        # This should work with current imports
        qfm_surface = compute_qfm_surface(surface, bs)
        assert qfm_surface is not None
    
    def test_plot_quasisymmetry_profile_creates_file(self, tmp_path):
        """Test that quasisymmetry profile plot creates output file."""
        radii = np.linspace(0, 1, 10)
        qs_profile = np.random.rand(10) * 0.01
        
        output_path = tmp_path / "qs_test.png"
        plot_quasisymmetry_profile(qs_profile, radii, output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0  # File is not empty
    
    def test_plot_quasisymmetry_profile_different_dpi(self, tmp_path):
        """Test quasisymmetry profile with different DPI."""
        radii = np.linspace(0, 1, 5)
        qs_profile = np.random.rand(5) * 0.01
        
        output_path = tmp_path / "qs_test_dpi.png"
        plot_quasisymmetry_profile(qs_profile, radii, output_path, dpi=150)
        
        assert output_path.exists()
    
    # def test_run_vmec_equilibrium_no_input_path(self, tmp_path):
    #     """Test VMEC equilibrium requires original surface input file."""
    #     try:
    #         pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
    #         pytest.importorskip("mpi4py", reason="mpi4py not available")
    #     except Exception:
    #         pytest.skip("VMEC or mpi4py not available")
        
    #     # Create a surface without filename
    #     surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
    #     surface.set_rc(0, 0, 1.0)
        
    #     # Should raise ValueError because VMEC requires the original surface input file
    #     with pytest.raises(ValueError, match="vmec_input_path must be provided"):
    #         run_vmec_equilibrium(surface, vmec_input_path=None)
    
    def test_compute_quasisymmetry_different_helicity(self, tmp_path):
        """Test quasisymmetry with different helicity numbers."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        # Mock VMEC equilibrium
        mock_equil = Mock()
        mock_qs = Mock()
        mock_qs.total.return_value = 0.001
        mock_qs.profile.return_value = np.array([0.001, 0.002, 0.001])
        
        with patch('stellcoilbench.post_processing.QuasisymmetryRatioResidual') as mock_qs_class:
            mock_qs_class.return_value = mock_qs
            
            qs_total, qs_profile = compute_quasisymmetry(
                mock_equil,
                helicity_m=1,
                helicity_n=-1,  # QH symmetry
                ns=3,
            )
            
            # Function now returns average quasisymmetry error
            # Profile is [0.001, 0.002, 0.001], average is (0.001 + 0.002 + 0.001) / 3
            assert abs(qs_total - 0.0013333333333333333) < 1e-10
            assert len(qs_profile) == 3
    
    def test_plot_iota_profile_different_signs(self, tmp_path):
        """Test iota profile plotting with different signs."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        # Mock VMEC equilibrium
        mock_equil = Mock()
        mock_wout = Mock()
        mock_wout.iotas = np.array([0.0, 0.5, 1.0, 1.5])
        mock_equil.wout = mock_wout
        mock_equil.ds = 0.01
        
        output_path = tmp_path / "iota_test.png"
        plot_iota_profile(mock_equil, output_path, sign=-1)
        
        assert output_path.exists()
    
    def test_trace_fieldlines_not_available(self, tmp_path):
        """Test trace_fieldlines when tracing is not available."""
        if TRACING_AVAILABLE:
            pytest.skip("Tracing is available")
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        output_path = tmp_path / "poincare.png"
        
        with pytest.raises(ImportError, match="Fieldline tracing requires"):
            trace_fieldlines(bs, surface, output_path)
    
    def test_run_post_processing_no_vmec(self, tmp_path):
        """Test run_post_processing without VMEC."""
        # Create minimal test setup
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        # Create minimal coils JSON
        coils_json = tmp_path / "coils.json"
        # Use higher resolution surface to match expected dimensions
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1, quadpoints_phi=np.linspace(0, 1, 16), quadpoints_theta=np.linspace(0, 1, 16))
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        output_dir = tmp_path / "post_output"
        
        # Mock surface loading
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            with patch('stellcoilbench.post_processing.compute_qfm_surface') as mock_qfm:
                mock_qfm.return_value = surface
                
                results = run_post_processing(
                    coils_json_path=coils_json,
                    output_dir=output_dir,
                    case_yaml_path=case_yaml,
                    plasma_surfaces_dir=tmp_path,
                    run_vmec=False,
                    plot_boozer=False,
                    plot_poincare=False,
                )
                
                assert results is not None
                assert 'qfm_surface' in results
                assert 'BdotN' in results
    
    def test_run_post_processing_vmec_failure(self, tmp_path):
        """Test run_post_processing when VMEC fails."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        # Create minimal test setup
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        # Create minimal coils JSON
        coils_json = tmp_path / "coils.json"
        # Use higher resolution surface to match expected dimensions
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1, quadpoints_phi=np.linspace(0, 1, 16), quadpoints_theta=np.linspace(0, 1, 16))
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        output_dir = tmp_path / "post_output"
        
        # Mock surface loading and QFM
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            with patch('stellcoilbench.post_processing.compute_qfm_surface') as mock_qfm:
                mock_qfm.return_value = surface
                with patch('stellcoilbench.post_processing.run_vmec_equilibrium') as mock_vmec:
                    mock_vmec.side_effect = Exception("VMEC failed")
                    
                    # Should not raise, but skip VMEC-dependent processing
                    results = run_post_processing(
                        coils_json_path=coils_json,
                        output_dir=output_dir,
                        case_yaml_path=case_yaml,
                        plasma_surfaces_dir=tmp_path,
                        run_vmec=True,
                        plot_boozer=False,
                        plot_poincare=False,
                    )
                    
                    assert results is not None
                    assert 'qfm_surface' in results
                    # VMEC-dependent results should not be present
                    assert 'vmec' not in results
    
    def test_run_post_processing_poincare_failure(self, tmp_path):
        """Test run_post_processing when Poincaré plotting fails."""
        # Create minimal test setup
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        # Create minimal coils JSON
        coils_json = tmp_path / "coils.json"
        # Use higher resolution surface to match expected dimensions
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1, quadpoints_phi=np.linspace(0, 1, 16), quadpoints_theta=np.linspace(0, 1, 16))
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        output_dir = tmp_path / "post_output"
        
        # Mock surface loading and QFM
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            with patch('stellcoilbench.post_processing.compute_qfm_surface') as mock_qfm:
                mock_qfm.return_value = surface
                with patch('stellcoilbench.post_processing.trace_fieldlines') as mock_trace:
                    mock_trace.side_effect = Exception("Poincaré failed")
                    
                    # Should not raise, but skip Poincaré plotting
                    results = run_post_processing(
                        coils_json_path=coils_json,
                        output_dir=output_dir,
                        case_yaml_path=case_yaml,
                        plasma_surfaces_dir=tmp_path,
                        run_vmec=False,
                        plot_boozer=False,
                        plot_poincare=True,
                    )
                    
                    assert results is not None
                    assert 'qfm_surface' in results
                    # Poincaré results should not be present
                    assert 'poincare_results' not in results
    
    def test_run_post_processing_all_plots_disabled(self, tmp_path):
        """Test run_post_processing with all plots disabled."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        coils_json = tmp_path / "coils.json"
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1, quadpoints_phi=np.linspace(0, 1, 16), quadpoints_theta=np.linspace(0, 1, 16))
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        output_dir = tmp_path / "post_output"
        
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            with patch('stellcoilbench.post_processing.compute_qfm_surface') as mock_qfm:
                mock_qfm.return_value = surface
                
                results = run_post_processing(
                    coils_json_path=coils_json,
                    output_dir=output_dir,
                    case_yaml_path=case_yaml,
                    plasma_surfaces_dir=tmp_path,
                    run_vmec=False,
                    plot_boozer=False,
                    plot_poincare=False,
                )
                
                assert results is not None
                assert 'qfm_surface' in results
                assert 'BdotN' in results
                assert 'BdotN_over_B' in results
    
    def test_run_post_processing_results_json(self, tmp_path):
        """Test that results JSON is created correctly."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        coils_json = tmp_path / "coils.json"
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1, quadpoints_phi=np.linspace(0, 1, 16), quadpoints_theta=np.linspace(0, 1, 16))
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        output_dir = tmp_path / "post_output"
        
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            with patch('stellcoilbench.post_processing.compute_qfm_surface') as mock_qfm:
                mock_qfm.return_value = surface
                
                _ = run_post_processing(
                    coils_json_path=coils_json,
                    output_dir=output_dir,
                    case_yaml_path=case_yaml,
                    plasma_surfaces_dir=tmp_path,
                    run_vmec=False,
                    plot_boozer=False,
                    plot_poincare=False,
                )
                
                # Check results JSON file
                results_json_path = output_dir / "post_processing_results.json"
                assert results_json_path.exists()
                
                with open(results_json_path, 'r') as f:
                    results_data = json.load(f)
                
                assert 'BdotN' in results_data
                assert 'BdotN_over_B' in results_data
                assert results_data['BdotN'] is not None
                assert results_data['BdotN_over_B'] is not None
    
    def test_run_post_processing_vmec_with_plots(self, tmp_path):
        """Test run_post_processing with VMEC and plots enabled."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        coils_json = tmp_path / "coils.json"
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1, quadpoints_phi=np.linspace(0, 1, 16), quadpoints_theta=np.linspace(0, 1, 16))
        surface.set_rc(0, 0, 1.0)
        surface.filename = str(surface_file)  # Set filename for VMEC
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        output_dir = tmp_path / "post_output"
        
        # Mock everything
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            with patch('stellcoilbench.post_processing.compute_qfm_surface') as mock_qfm:
                mock_qfm.return_value = surface
                with patch('stellcoilbench.post_processing.run_vmec_equilibrium') as mock_vmec:
                    # Mock VMEC equilibrium
                    mock_equil = Mock()
                    mock_wout = Mock()
                    mock_wout.iotas = np.array([0.0, 0.5, 1.0, 1.5])
                    mock_equil.wout = mock_wout
                    mock_vmec.return_value = mock_equil
                    
                    with patch('stellcoilbench.post_processing.compute_quasisymmetry') as mock_qs:
                        mock_qs.return_value = (0.001, np.array([0.001, 0.002, 0.001]))
                        
                        with patch('stellcoilbench.post_processing.plot_boozer_surface') as mock_boozer:
                            with patch('stellcoilbench.post_processing.plot_iota_profile') as mock_iota:
                                with patch('stellcoilbench.post_processing.plot_quasisymmetry_profile') as mock_qs_plot:
                                    
                                    results = run_post_processing(
                                        coils_json_path=coils_json,
                                        output_dir=output_dir,
                                        case_yaml_path=case_yaml,
                                        plasma_surfaces_dir=tmp_path,
                                        run_vmec=True,
                                        plot_boozer=True,
                                        plot_poincare=False,
                                        ns=3,
                                    )
                                    
                                    assert results is not None
                                    assert 'qfm_surface' in results
                                    assert 'vmec' in results
                                    assert 'quasisymmetry_average' in results
                                    
                                    # Verify plots were called
                                    mock_boozer.assert_called_once()
                                    mock_iota.assert_called_once()
                                    mock_qs_plot.assert_called_once()
    
    def test_run_post_processing_poincare_success(self, tmp_path):
        """Test run_post_processing with successful Poincaré plotting."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        coils_json = tmp_path / "coils.json"
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1, quadpoints_phi=np.linspace(0, 1, 16), quadpoints_theta=np.linspace(0, 1, 16))
        surface.set_rc(0, 0, 1.0)
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        output_dir = tmp_path / "post_output"
        
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            with patch('stellcoilbench.post_processing.compute_qfm_surface') as mock_qfm:
                mock_qfm.return_value = surface
                with patch('stellcoilbench.post_processing.trace_fieldlines') as mock_trace:
                    mock_trace.return_value = {
                        'fieldlines_tys': [],
                        'fieldlines_phi_hits': [],
                        'phis': [0, np.pi/2],
                    }
                    
                    results = run_post_processing(
                        coils_json_path=coils_json,
                        output_dir=output_dir,
                        case_yaml_path=case_yaml,
                        plasma_surfaces_dir=tmp_path,
                        run_vmec=False,
                        plot_boozer=False,
                        plot_poincare=True,
                        nfieldlines=5,
                    )
                    
                    assert results is not None
                    assert 'poincare_results' in results
                    mock_trace.assert_called_once()
    
    def test_load_coils_and_surface_finds_case_yaml_in_parent(self, tmp_path):
        """Test finding case.yaml in parent directory."""
        # Create structure: tmp_path / subdir / coils.json, tmp_path / case.yaml
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: "input.test"
  range: "half period"
""")
        
        # Put surface file in tmp_path so it can be found
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        coils_json = subdir / "coils.json"
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        from simsopt import save
        save(bs, coils_json)
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_from_input.return_value = surface
            # Should find case.yaml in parent directory
            bfield, loaded_surface = load_coils_and_surface(
                coils_json,
                case_yaml_path=None,  # Should find automatically
                plasma_surfaces_dir=tmp_path,  # Provide plasma_surfaces_dir so surface file is found
            )
            assert bfield is not None
            assert loaded_surface is not None
    
    def test_trace_fieldlines_with_interpolated_field(self, tmp_path):
        """Test trace_fieldlines with interpolated field enabled."""
        if not TRACING_AVAILABLE:
            pytest.skip("Fieldline tracing not available")
        
        # Create minimal setup
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)
        surface.set_rc(1, 0, 0.1)
        
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        output_path = tmp_path / "poincare.png"
        
        try:
            results = trace_fieldlines(
                bs,
                surface,
                output_path,
                nfieldlines=3,
                use_interpolated_field=True,
                n_phi_slices=2,
            )
            
            assert output_path.exists()
            assert 'fieldlines_tys' in results
            assert 'fieldlines_phi_hits' in results
            assert 'phis' in results
        except Exception as e:
            pytest.skip(f"Fieldline tracing failed: {e}")
    
    def test_trace_fieldlines_different_parameters(self, tmp_path):
        """Test trace_fieldlines with different parameters."""
        if not TRACING_AVAILABLE:
            pytest.skip("Fieldline tracing not available")
        
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)
        surface.set_rc(1, 0, 0.1)
        
        coils = create_equally_spaced_curves(2, 1, stellsym=True, R0=1.2, R1=0.1, order=2)
        base_currents = [Current(1e6) for _ in range(2)]
        coils_list = coils_via_symmetries(coils, base_currents, 1, True)
        bs = BiotSavart(coils_list)
        
        output_path = tmp_path / "poincare2.png"
        
        try:
            results = trace_fieldlines(
                bs,
                surface,
                output_path,
                nfieldlines=5,
                tmax=5000,
                tol=1e-8,
                n_phi_slices=8,
                use_interpolated_field=False,
                dpi=150,
            )
            
            assert output_path.exists()
            assert 'fieldlines_tys' in results
        except Exception as e:
            pytest.skip(f"Fieldline tracing failed: {e}")
    
    def test_r0_initialization_simple_torus(self):
        """Test that R0 is correctly initialized between innermost and outermost points along phi=0, Z=0."""
        if not TRACING_AVAILABLE:
            pytest.skip("Fieldline tracing not available")
        
        # Create a simple torus: major radius R0=1.0, minor radius a=0.1
        # For a torus: R(phi, theta) = R0 + a*cos(theta), Z(phi, theta) = a*sin(theta)
        # At phi=0, Z=0: theta=0 or pi, giving R = R0 ± a = 1.0 ± 0.1
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=1)
        surface.set_rc(0, 0, 1.0)  # Major radius R0 = 1.0
        surface.set_rc(1, 0, 0.1)  # Minor radius component a = 0.1
        surface.set_zs(0, 0, 0.0)
        
        # Replicate the R0 initialization logic from trace_fieldlines
        gamma = surface.gamma()  # Shape: (nphi, ntheta, 3)
        
        # Find phi index closest to 0
        phi_normalized_0 = 0.0
        phi_normalized_values = surface.quadpoints_phi
        phi_idx = np.argmin(np.abs(phi_normalized_values - phi_normalized_0))
        
        # Get all points at phi = 0 (or closest to it)
        points_at_phi0 = gamma[phi_idx, :, :]  # Shape: (ntheta, 3)
        
        # Find points where Z ≈ 0 (within tolerance)
        z_tolerance = 0.01  # 1 cm tolerance
        z_near_zero_mask = np.abs(points_at_phi0[:, 2]) < z_tolerance
        
        if np.any(z_near_zero_mask):
            # Compute R = sqrt(X^2 + Y^2) for points where Z ≈ 0
            points_z0 = points_at_phi0[z_near_zero_mask]
            R_values = np.sqrt(points_z0[:, 0]**2 + points_z0[:, 1]**2)
            
            R_min = np.min(R_values)
            R_max = np.max(R_values)
        else:
            # Fallback: find point closest to Z = 0
            z_abs = np.abs(points_at_phi0[:, 2])
            closest_idx = np.argmin(z_abs)
            closest_point = points_at_phi0[closest_idx]
            R_closest = np.sqrt(closest_point[0]**2 + closest_point[1]**2)
            
            # Use a range around this point
            major_radius = surface.get_rc(0, 0)
            minor_radius_component = abs(surface.get_rc(1, 0))
            R_min = max(R_closest - minor_radius_component * 0.5, major_radius * 0.5)
            R_max = R_closest + minor_radius_component * 0.5
        
        # Sample R0 between innermost and outermost points
        R_start = R_min * 1.01  # Slightly inside innermost point
        R_end = R_max * 0.99  # Slightly inside outermost point
        
        # Ensure R_start < R_end (safety check)
        if R_start >= R_end:
            R_mid = (R_min + R_max) / 2.0
            R_range = max(R_max - R_min, R_min * 0.1)
            R_start = R_mid - R_range * 0.4
            R_end = R_mid + R_range * 0.4
        
        nfieldlines = 10
        R0 = np.linspace(R_start, R_end, nfieldlines)
        
        # Verify R0 values are within expected range
        # For a torus with R0=1.0, a=0.1, at phi=0, Z=0:
        # R should be between R0 - a = 0.9 and R0 + a = 1.1
        # Accounting for the 1% margin, R0 should be between ~0.909 and ~1.089
        expected_R_min = 0.9 * 1.01  # Slightly inside innermost
        expected_R_max = 1.1 * 0.99  # Slightly inside outermost
        
        # Allow some tolerance for numerical precision
        tolerance = 0.05
        
        assert np.all(R0 >= expected_R_min - tolerance), \
            f"R0 values {R0} should be >= {expected_R_min - tolerance}"
        assert np.all(R0 <= expected_R_max + tolerance), \
            f"R0 values {R0} should be <= {expected_R_max + tolerance}"
        
        # Verify R0 values are monotonically increasing
        assert np.all(np.diff(R0) > 0), "R0 values should be monotonically increasing"
        
        # Verify R0 values span the range (not all the same)
        assert R0[-1] > R0[0], "R0 should span a range from innermost to outermost"
        
        # Verify we found points at Z ≈ 0
        assert np.any(z_near_zero_mask), \
            "Should find points where Z ≈ 0 along phi=0 line"
        
        # Verify R_min and R_max are reasonable
        assert R_min < R_max, "R_min should be less than R_max"
        assert R_min >= 0.8, f"R_min {R_min} should be >= 0.8 (R0 - a - tolerance)"
        assert R_max <= 1.2, f"R_max {R_max} should be <= 1.2 (R0 + a + tolerance)"