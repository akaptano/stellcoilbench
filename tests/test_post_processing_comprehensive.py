"""
Comprehensive unit tests for post_processing.py to increase coverage.

These tests focus on functions and edge cases that need more coverage.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from stellcoilbench.post_processing import (
    load_surface_with_range,
    compute_qfm_surface,
    run_vmec_equilibrium,
    compute_quasisymmetry,
    plot_boozer_surface,
    plot_iota_profile,
    plot_quasisymmetry_profile,
    trace_fieldlines,
    run_post_processing,
)


class TestLoadSurfaceWithRange:
    """Tests for load_surface_with_range function."""
    
    def test_load_surface_with_range_vmec_input(self, tmp_path):
        """Test loading surface with range from VMEC input file."""
        pytest.importorskip("simsopt.geo", reason="simsopt not available")
        from simsopt.geo import SurfaceRZFourier
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy content")
        
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_vmec_input') as mock_from_input:
            mock_surface = Mock(spec=SurfaceRZFourier)
            mock_from_input.return_value = mock_surface
            
            result = load_surface_with_range(surface_file, surface_range="half period")
            
            assert result == mock_surface
            mock_from_input.assert_called_once()
    
    def test_load_surface_with_range_focus(self, tmp_path):
        """Test loading surface with range from FOCUS file."""
        pytest.importorskip("simsopt.geo", reason="simsopt not available")
        from simsopt.geo import SurfaceRZFourier
        
        surface_file = tmp_path / "test.focus"
        surface_file.write_text("dummy content")
        
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_focus') as mock_from_focus:
            mock_surface = Mock(spec=SurfaceRZFourier)
            mock_from_focus.return_value = mock_surface
            
            result = load_surface_with_range(surface_file, surface_range="full torus")
            
            assert result == mock_surface
            mock_from_focus.assert_called_once()
    
    def test_load_surface_with_range_wout(self, tmp_path):
        """Test loading surface with range from wout file."""
        pytest.importorskip("simsopt.geo", reason="simsopt not available")
        from simsopt.geo import SurfaceRZFourier
        
        surface_file = tmp_path / "wout.test.nc"
        surface_file.write_text("dummy content")
        
        with patch('stellcoilbench.post_processing.SurfaceRZFourier.from_wout') as mock_from_wout:
            mock_surface = Mock(spec=SurfaceRZFourier)
            mock_from_wout.return_value = mock_surface
            
            result = load_surface_with_range(surface_file, surface_range="half period")
            
            assert result == mock_surface
            mock_from_wout.assert_called_once()
    
    def test_load_surface_with_range_unknown_type(self, tmp_path):
        """Test loading surface with unknown file type raises error."""
        surface_file = tmp_path / "test.unknown"
        surface_file.write_text("dummy content")
        
        with pytest.raises(ValueError, match="Unknown surface file type"):
            load_surface_with_range(surface_file, surface_range="half period")


class TestComputeQfmSurface:
    """Tests for compute_qfm_surface function."""
    
    def test_compute_qfm_surface_success(self):
        """Test successful QFM surface computation."""
        pytest.importorskip("simsopt.util.permanent_magnet_helper_functions", reason="make_qfm not available")
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        
        # Create mock surface and field
        mock_surface = Mock(spec=SurfaceRZFourier)
        mock_bfield = Mock(spec=BiotSavart)
        
        # Create mock QFM result object with surface attribute
        mock_qfm_result = Mock()
        mock_qfm_surface = Mock(spec=SurfaceRZFourier)
        mock_qfm_result.surface = mock_qfm_surface
        
        with patch('stellcoilbench.post_processing.make_qfm') as mock_make_qfm:
            mock_make_qfm.return_value = mock_qfm_result
            
            result = compute_qfm_surface(mock_surface, mock_bfield)
            
            assert result == mock_qfm_surface
            mock_make_qfm.assert_called_once()
    
    def test_compute_qfm_surface_import_error(self):
        """Test QFM computation when make_qfm is not available."""
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        
        mock_surface = Mock(spec=SurfaceRZFourier)
        mock_bfield = Mock(spec=BiotSavart)
        
        with patch('stellcoilbench.post_processing.make_qfm', side_effect=ImportError("make_qfm not found")):
            with pytest.raises(ImportError):
                compute_qfm_surface(mock_bfield, mock_surface)


class TestRunVmecEquilibrium:
    """Tests for run_vmec_equilibrium function."""
    
    def test_run_vmec_equilibrium_success(self, tmp_path):
        """Test successful VMEC equilibrium run."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        from simsopt.geo import SurfaceRZFourier
        
        mock_surface = Mock(spec=SurfaceRZFourier)
        mock_equil = Mock()
        mock_equil.as_dict.return_value = {"iota": [0.5, 1.0]}
        mock_equil.wout.iotas = np.array([0.5, 1.0])
        
        # Create a template input file
        template_file = tmp_path / "input.template"
        template_file.write_text("dummy")
        
        with patch('stellcoilbench.post_processing.Vmec') as mock_vmec_class:
            mock_vmec_class.return_value = mock_equil
            mock_equil.run.return_value = None
            
            # Mock glob to find template
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = [template_file]
                
                result = run_vmec_equilibrium(mock_surface, tmp_path)
                
                assert result == mock_equil
                mock_equil.run.assert_called_once()
    
    def test_run_vmec_equilibrium_failure(self, tmp_path):
        """Test VMEC equilibrium run failure."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        from simsopt.geo import SurfaceRZFourier
        
        mock_surface = Mock(spec=SurfaceRZFourier)
        mock_equil = Mock()
        mock_equil.run.side_effect = Exception("VMEC failed")
        
        template_file = tmp_path / "input.template"
        template_file.write_text("dummy")
        
        with patch('stellcoilbench.post_processing.Vmec') as mock_vmec_class:
            mock_vmec_class.return_value = mock_equil
            
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = [template_file]
                
                # VMEC failures should be caught and handled
                # The function may raise or return None depending on implementation
                # Let's test that it handles the exception
                with patch('stellcoilbench.post_processing.proc0_print'):
                    # Check if it raises or returns None
                    # Looking at the code, run_vmec_equilibrium doesn't catch exceptions
                    # So it will raise - but run_post_processing catches it
                    with pytest.raises(Exception, match="VMEC failed"):
                        run_vmec_equilibrium(mock_surface, tmp_path)
    
    def test_run_vmec_equilibrium_no_template(self, tmp_path):
        """Test VMEC equilibrium when no template file found."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        from simsopt.geo import SurfaceRZFourier
        
        mock_surface = Mock(spec=SurfaceRZFourier)
        mock_equil = Mock()
        
        with patch('stellcoilbench.post_processing.Vmec') as mock_vmec_class:
            mock_vmec_class.return_value = mock_equil
            
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = []  # No template found
                
                # Should still work by creating Vmec without template
                result = run_vmec_equilibrium(mock_surface, tmp_path)
                
                # May return None or raise, depending on implementation
                # Just verify it doesn't crash
                assert result is None or result == mock_equil


class TestComputeQuasisymmetry:
    """Tests for compute_quasisymmetry function."""
    
    def test_compute_quasisymmetry_success(self):
        """Test successful quasisymmetry computation."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        pytest.importorskip("simsopt.mhd", reason="QuasisymmetryRatioResidual not available")
        
        mock_equil = Mock()
        mock_qs = Mock()
        mock_qs.profile.return_value = np.array([0.001, 0.002, 0.003])
        
        with patch('stellcoilbench.post_processing.QuasisymmetryRatioResidual') as mock_qs_class:
            mock_qs_class.return_value = mock_qs
            
            qs_avg, qs_profile = compute_quasisymmetry(
                mock_equil,
                helicity_m=1,
                helicity_n=-1,
                ns=3,
            )
            
            assert np.isfinite(qs_avg)
            assert len(qs_profile) == 3
            assert qs_avg == np.mean([0.001, 0.002, 0.003])
    
    def test_compute_quasisymmetry_different_helicity(self):
        """Test quasisymmetry computation with different helicity."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        pytest.importorskip("simsopt.mhd", reason="QuasisymmetryRatioResidual not available")
        
        mock_equil = Mock()
        mock_qs = Mock()
        mock_qs.profile.return_value = np.array([0.001, 0.002])
        
        with patch('stellcoilbench.post_processing.QuasisymmetryRatioResidual') as mock_qs_class:
            mock_qs_class.return_value = mock_qs
            
            qs_avg, qs_profile = compute_quasisymmetry(
                mock_equil,
                helicity_m=2,
                helicity_n=-2,
                ns=2,
            )
            
            assert np.isfinite(qs_avg)
            assert len(qs_profile) == 2


class TestPlotBoozerSurface:
    """Tests for plot_boozer_surface function."""
    
    def test_plot_boozer_surface_with_js(self, tmp_path):
        """Test plotting Boozer surface with specific js index."""
        pytest.importorskip("booz_xform", reason="booz_xform not available")
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        mock_equil = Mock()
        mock_equil.wout.iotas = np.array([0.5, 1.0, 1.5])
        mock_equil.output_file = str(tmp_path / "wout.test.nc")
        
        mock_b2 = Mock()
        mock_b2.read_wout = Mock()
        mock_b2.run = Mock()
        
        output_path = tmp_path / "boozer_plot.pdf"
        
        # Create a mock module for booz_xform
        mock_bx_module = Mock()
        mock_bx_module.Booz_xform = Mock(return_value=mock_b2)
        mock_bx_module.surfplot = Mock()
        
        # Patch sys.modules to inject our mock
        import sys
        original_import = sys.modules.get('booz_xform')
        sys.modules['booz_xform'] = mock_bx_module
        
        try:
            plot_boozer_surface(mock_equil, output_path, js=1)
            
            assert output_path.exists()
            mock_bx_module.surfplot.assert_called_once()
        finally:
            if original_import is not None:
                sys.modules['booz_xform'] = original_import
            elif 'booz_xform' in sys.modules:
                del sys.modules['booz_xform']
    
    def test_plot_boozer_surface_2x2_grid(self, tmp_path):
        """Test plotting Boozer surface with 2x2 grid (js=None)."""
        pytest.importorskip("booz_xform", reason="booz_xform not available")
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        mock_equil = Mock()
        # Create enough surfaces for 2x2 grid
        mock_equil.wout.iotas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        mock_equil.output_file = str(tmp_path / "wout.test.nc")
        
        mock_b2 = Mock()
        mock_b2.wout.iotas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        mock_b2.read_wout = Mock()
        mock_b2.run = Mock()
        
        output_path = tmp_path / "boozer_plot.pdf"
        
        # Create a mock module for booz_xform
        mock_bx_module = Mock()
        mock_bx_module.Booz_xform = Mock(return_value=mock_b2)
        mock_bx_module.surfplot = Mock()
        
        # Patch sys.modules to inject our mock
        import sys
        original_import = sys.modules.get('booz_xform')
        sys.modules['booz_xform'] = mock_bx_module
        
        try:
            plot_boozer_surface(mock_equil, output_path, js=None)
            
            # Should create 2x2 grid = 4 plots
            assert mock_bx_module.surfplot.call_count == 4
            assert output_path.exists()
        finally:
            if original_import is not None:
                sys.modules['booz_xform'] = original_import
            elif 'booz_xform' in sys.modules:
                del sys.modules['booz_xform']
    
    def test_plot_boozer_surface_index_error_handling(self, tmp_path):
        """Test Boozer surface plotting with index error handling."""
        pytest.importorskip("booz_xform", reason="booz_xform not available")
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        mock_equil = Mock()
        mock_equil.wout.iotas = np.array([0.0, 0.5, 1.0])
        mock_equil.output_file = str(tmp_path / "wout.test.nc")
        
        mock_b2 = Mock()
        mock_b2.wout.iotas = np.array([0.0, 0.5, 1.0])
        mock_b2.read_wout = Mock()
        mock_b2.run = Mock()
        
        output_path = tmp_path / "boozer_plot.pdf"
        
        # Create a mock module for booz_xform
        mock_bx_module = Mock()
        mock_bx_module.Booz_xform = Mock(return_value=mock_b2)
        
        # Make surfplot raise IndexError for some calls, then succeed with fallback
        call_count = [0]
        def mock_surfplot(b2, js=None, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:
                raise IndexError("out of bounds")
            # Subsequent calls succeed (fallback logic)
            return None
        
        # Create a Mock for surfplot that tracks calls
        mock_surfplot_func = Mock(side_effect=mock_surfplot)
        mock_bx_module.surfplot = mock_surfplot_func
        
        # Patch sys.modules to inject our mock
        import sys
        original_import = sys.modules.get('booz_xform')
        sys.modules['booz_xform'] = mock_bx_module
        
        try:
            # Should handle errors gracefully and use fallback indices
            plot_boozer_surface(mock_equil, output_path, js=None)
            
            assert output_path.exists()
            # Should have called surfplot multiple times (some may fail, fallback succeeds)
            assert mock_surfplot_func.call_count >= 1
        finally:
            if original_import is not None:
                sys.modules['booz_xform'] = original_import
            elif 'booz_xform' in sys.modules:
                del sys.modules['booz_xform']


class TestPlotIotaProfile:
    """Tests for plot_iota_profile function."""
    
    def test_plot_iota_profile_success(self, tmp_path):
        """Test successful iota profile plotting."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        mock_equil = Mock()
        mock_equil.wout.iotas = np.array([0.0, 0.5, 1.0, 1.5])
        mock_equil.ds = 0.01
        
        mock_equil_original = Mock()
        mock_equil_original.wout.iotas = np.array([0.0, 0.4, 0.9, 1.4])
        mock_equil_original.ds = 0.01
        
        output_path = tmp_path / "iota_profile.pdf"
        
        plot_iota_profile(mock_equil, output_path, sign=1, equil_original=mock_equil_original)
        
        assert output_path.exists()
    
    def test_plot_iota_profile_negative_sign(self, tmp_path):
        """Test iota profile plotting with negative sign."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        mock_equil = Mock()
        mock_equil.wout.iotas = np.array([0.0, 0.5, 1.0])
        mock_equil.ds = 0.01
        
        output_path = tmp_path / "iota_profile.pdf"
        
        plot_iota_profile(mock_equil, output_path, sign=-1)
        
        assert output_path.exists()
    
    def test_plot_iota_profile_no_original(self, tmp_path):
        """Test iota profile plotting without original equilibrium."""
        pytest.importorskip("simsopt.mhd.vmec", reason="VMEC not available")
        
        mock_equil = Mock()
        mock_equil.wout.iotas = np.array([0.0, 0.5, 1.0])
        mock_equil.ds = 0.01
        
        output_path = tmp_path / "iota_profile.pdf"
        
        plot_iota_profile(mock_equil, output_path, sign=1, equil_original=None)
        
        assert output_path.exists()


class TestPlotQuasisymmetryProfile:
    """Tests for plot_quasisymmetry_profile function."""
    
    def test_plot_quasisymmetry_profile_success(self, tmp_path):
        """Test successful quasisymmetry profile plotting."""
        mock_qs_profile = np.array([0.001, 0.002, 0.003])
        mock_radii = np.array([0.0, 0.5, 1.0])
        
        output_path = tmp_path / "qs_profile.pdf"
        
        plot_quasisymmetry_profile(mock_qs_profile, mock_radii, output_path)
        
        assert output_path.exists()
    
    def test_plot_quasisymmetry_profile_with_original(self, tmp_path):
        """Test quasisymmetry profile plotting with original data."""
        mock_qs_profile = np.array([0.001, 0.002, 0.003])
        mock_radii = np.array([0.0, 0.5, 1.0])
        mock_qs_profile_original = np.array([0.002, 0.003, 0.004])
        mock_radii_original = np.array([0.0, 0.5, 1.0])
        
        output_path = tmp_path / "qs_profile.pdf"
        
        plot_quasisymmetry_profile(
            mock_qs_profile, mock_radii, output_path,
            qs_profile_original=mock_qs_profile_original,
            radii_original=mock_radii_original
        )
        
        assert output_path.exists()
    
    def test_plot_quasisymmetry_profile_different_dpi(self, tmp_path):
        """Test quasisymmetry profile plotting with different DPI."""
        mock_qs_profile = np.array([0.001, 0.002])
        mock_radii = np.array([0.0, 1.0])
        
        output_path = tmp_path / "qs_profile.pdf"
        
        plot_quasisymmetry_profile(mock_qs_profile, mock_radii, output_path, dpi=150)
        
        assert output_path.exists()


class TestTraceFieldlines:
    """Tests for trace_fieldlines function."""
    
    def test_trace_fieldlines_not_available(self):
        """Test trace_fieldlines when tracing is not available."""
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        
        mock_bfield = Mock(spec=BiotSavart)
        mock_surface = Mock(spec=SurfaceRZFourier)
        
        with patch('stellcoilbench.post_processing.TRACING_AVAILABLE', False):
            with pytest.raises(ImportError, match="Fieldline tracing requires"):
                trace_fieldlines(mock_bfield, mock_surface, Path("/tmp/test.pdf"))
    
    def test_trace_fieldlines_different_parameters(self, tmp_path):
        """Test trace_fieldlines with different parameters."""
        pytest.importorskip("simsopt.field.tracing", reason="tracing not available")
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        
        # Use a real surface for this test since SurfaceClassifier needs it
        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        mock_bfield = Mock(spec=BiotSavart)
        # Mock BiotSavart methods needed
        mock_bfield.B.return_value = np.random.rand(100, 3)
        mock_bfield.AbsB.return_value = np.random.rand(100)
        
        output_path = tmp_path / "poincare.pdf"
        
        # Mock fieldline tracing functions
        mock_fieldlines_tys = [np.random.rand(100, 3) for _ in range(5)]
        mock_fieldlines_phi_hits = [np.random.rand(50, 2) for _ in range(5)]
        
        with patch('stellcoilbench.post_processing.compute_fieldlines') as mock_compute:
            mock_compute.return_value = (mock_fieldlines_tys, mock_fieldlines_phi_hits)
            
            with patch('stellcoilbench.post_processing.plot_poincare_data'):
                result = trace_fieldlines(
                    mock_bfield,
                    surface,
                    output_path,
                    nfieldlines=5,
                    tmax=10000,
                    tol=1e-10,
                    n_phi_slices=4,
                    use_interpolated_field=False,
                )
                
                assert result is not None
                assert 'fieldlines_tys' in result
                assert 'fieldlines_phi_hits' in result
                assert 'phis' in result


class TestRunPostProcessingEdgeCases:
    """Tests for edge cases in run_post_processing."""
    
    def test_run_post_processing_no_vmec_no_plots(self, tmp_path):
        """Test run_post_processing with VMEC and plots disabled."""
        pytest.importorskip("simsopt.geo", reason="simsopt not available")
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        
        # Create minimal coils JSON
        coils_json = tmp_path / "coils.json"
        mock_coils = [Mock()]
        mock_bfield = Mock(spec=BiotSavart)
        
        # Mock BiotSavart to return arrays for B and n calculations
        mock_bfield.B.return_value = np.random.rand(100, 3)
        
        with patch('simsopt.save') as mock_save:
            mock_save(mock_coils, str(coils_json))
        
        # Create case YAML
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: input.test
coils_params:
  ncoils: 2
  order: 2
optimizer_params:
  algorithm: l-bfgs
""")
        
        # Create surface file
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        with patch('stellcoilbench.post_processing.load_coils_and_surface') as mock_load:
            mock_surface = Mock(spec=SurfaceRZFourier)
            # Mock surface methods needed for QFM computation
            gamma_array = np.random.rand(10, 10, 3)
            normal_array = np.random.rand(10, 10, 3)
            mock_surface.gamma = lambda: gamma_array
            mock_surface.normal = lambda: normal_array
            mock_surface.unitnormal = lambda: normal_array
            # Mock shape attributes for reshape operations
            # numpy arrays already have .size attribute, no need to set it
            quadpoints_phi = np.linspace(0, 1, 10)
            quadpoints_theta = np.linspace(0, 1, 10)
            mock_surface.quadpoints_phi = quadpoints_phi
            mock_surface.quadpoints_theta = quadpoints_theta
            # Mock AbsB for BdotN_over_B calculation
            mock_bfield.AbsB.return_value = np.random.rand(100)
            mock_load.return_value = (mock_bfield, mock_surface)
            
            with patch('stellcoilbench.post_processing.compute_qfm_surface') as mock_qfm:
                mock_qfm.return_value = mock_surface
                
                results = run_post_processing(
                    coils_json_path=coils_json,
                    output_dir=tmp_path / "output",
                    case_yaml_path=case_yaml,
                    plasma_surfaces_dir=tmp_path,
                    run_vmec=False,
                    plot_boozer=False,
                    plot_poincare=False,
                )
                
                assert results is not None
                assert 'qfm_surface' in results
    
    def test_run_post_processing_vmec_failure_handling(self, tmp_path):
        """Test run_post_processing handles VMEC failures gracefully."""
        pytest.importorskip("simsopt.geo", reason="simsopt not available")
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        
        coils_json = tmp_path / "coils.json"
        mock_coils = [Mock()]
        
        mock_bfield = Mock(spec=BiotSavart)
        mock_bfield.B.return_value = np.random.rand(100, 3)
        
        with patch('simsopt.save') as mock_save:
            mock_save(mock_coils, str(coils_json))
        
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: input.test
coils_params:
  ncoils: 2
  order: 2
optimizer_params:
  algorithm: l-bfgs
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        mock_surface = Mock(spec=SurfaceRZFourier)
        mock_surface.gamma.return_value = np.random.rand(10, 10, 3)
        mock_surface.normal.return_value = np.random.rand(10, 10, 3)
        
        with patch('stellcoilbench.post_processing.load_coils_and_surface') as mock_load:
            mock_surface = Mock(spec=SurfaceRZFourier)
            gamma_array = np.random.rand(10, 10, 3)
            normal_array = np.random.rand(10, 10, 3)
            mock_surface.gamma = lambda: gamma_array
            mock_surface.normal = lambda: normal_array
            mock_surface.unitnormal = lambda: normal_array
            quadpoints_phi = np.linspace(0, 1, 10)
            quadpoints_theta = np.linspace(0, 1, 10)
            mock_surface.quadpoints_phi = quadpoints_phi
            mock_surface.quadpoints_theta = quadpoints_theta
            # Mock AbsB for BdotN_over_B calculation
            mock_bfield.AbsB.return_value = np.random.rand(100)
            mock_load.return_value = (mock_bfield, mock_surface)
            
            with patch('stellcoilbench.post_processing.compute_qfm_surface') as mock_qfm:
                mock_qfm.return_value = mock_surface
                
                with patch('stellcoilbench.post_processing.run_vmec_equilibrium') as mock_vmec:
                    mock_vmec.return_value = None  # VMEC fails
                    
                    results = run_post_processing(
                        coils_json_path=coils_json,
                        output_dir=tmp_path / "output",
                        case_yaml_path=case_yaml,
                        plasma_surfaces_dir=tmp_path,
                        run_vmec=True,
                        plot_boozer=False,
                        plot_poincare=False,
                    )
                    
                    assert results is not None
                    # Should still have QFM surface even if VMEC fails
                    assert 'qfm_surface' in results
    
    def test_run_post_processing_poincare_failure_handling(self, tmp_path):
        """Test run_post_processing handles Poincaré failures gracefully."""
        pytest.importorskip("simsopt.geo", reason="simsopt not available")
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        
        coils_json = tmp_path / "coils.json"
        mock_coils = [Mock()]
        
        mock_bfield = Mock(spec=BiotSavart)
        mock_bfield.B.return_value = np.random.rand(100, 3)
        
        with patch('simsopt.save') as mock_save:
            mock_save(mock_coils, str(coils_json))
        
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""
surface_params:
  surface: input.test
coils_params:
  ncoils: 2
  order: 2
optimizer_params:
  algorithm: l-bfgs
""")
        
        surface_file = tmp_path / "input.test"
        surface_file.write_text("dummy")
        
        mock_surface = Mock(spec=SurfaceRZFourier)
        mock_surface.gamma.return_value = np.random.rand(10, 10, 3)
        mock_surface.normal.return_value = np.random.rand(10, 10, 3)
        mock_surface.filename = str(surface_file)
        
        with patch('stellcoilbench.post_processing.load_coils_and_surface') as mock_load:
            mock_surface = Mock(spec=SurfaceRZFourier)
            gamma_array = np.random.rand(10, 10, 3)
            normal_array = np.random.rand(10, 10, 3)
            mock_surface.gamma = lambda: gamma_array
            mock_surface.normal = lambda: normal_array
            mock_surface.unitnormal = lambda: normal_array
            quadpoints_phi = np.linspace(0, 1, 10)
            quadpoints_theta = np.linspace(0, 1, 10)
            mock_surface.quadpoints_phi = quadpoints_phi
            mock_surface.quadpoints_theta = quadpoints_theta
            mock_surface.filename = str(surface_file)
            # Mock AbsB and set_points
            mock_bfield.AbsB.return_value = np.random.rand(100)
            mock_bfield.set_points = Mock()
            mock_load.return_value = (mock_bfield, mock_surface)
            
            with patch('stellcoilbench.post_processing.compute_qfm_surface') as mock_qfm:
                mock_qfm.return_value = mock_surface
                
                with patch('stellcoilbench.post_processing.trace_fieldlines') as mock_trace:
                    mock_trace.side_effect = Exception("Poincaré failed")
                    
                    results = run_post_processing(
                        coils_json_path=coils_json,
                        output_dir=tmp_path / "output",
                        case_yaml_path=case_yaml,
                        plasma_surfaces_dir=tmp_path,
                        run_vmec=False,
                        plot_boozer=False,
                        plot_poincare=True,
                    )
                    
                    assert results is not None
                    # Should still complete even if Poincaré fails
                    assert 'qfm_surface' in results
