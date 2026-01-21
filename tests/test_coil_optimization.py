"""
Unit tests for coil_optimization.py

Note: Full optimization tests require simsopt and are integration tests.
These unit tests focus on functions that can be tested without running optimizations.
"""
import pytest
import numpy as np
from pathlib import Path
from stellcoilbench.coil_optimization import load_coils_config, _plot_bn_error_3d
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
    
    def test_verbose_not_in_coils_params(self, tmp_path):
        """Test that verbose in coils_params raises validation error (verbose should only be in optimizer_params)."""
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
        
        with pytest.raises(ValueError, match="Unknown coils_params key.*verbose"):
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
optimizer_params:
  algorithm: l-bfgs
  verbose: False
""")
        
        # Should not raise an error
        config = load_case_config(case_yaml)
        assert config.coils_params["ncoils"] == 4
        assert config.coils_params["order"] == 16
        assert "verbose" not in config.coils_params
        assert not config.optimizer_params["verbose"]
    
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


class TestPlotBnError3D:
    """Tests for _plot_bn_error_3d function."""
    
    def test_plot_bn_error_3d_creates_pdf(self, tmp_path):
        """Test that _plot_bn_error_3d creates a PDF file."""
        pytest.importorskip("matplotlib")
        
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries
        from stellcoilbench.evaluate import load_case_config
        
        # Load a realistic surface from the LandremanPaul QA case
        repo_root = Path(__file__).resolve().parents[1]
        case_path = repo_root / "cases" / "expert_LandremanPaulQA.yaml"
        case_cfg = load_case_config(case_path)
        surface_file = Path(case_cfg.surface_params["surface"])
        if not surface_file.is_absolute():
            surface_file = repo_root / "plasma_surfaces" / surface_file
        surface = SurfaceRZFourier.from_vmec_input(
            filename=str(surface_file),
            range="full torus",
            quadpoints_phi=np.linspace(0, 1, 128),
            quadpoints_theta=np.linspace(0, 1, 128),
        )
        
        # Create a simple coil configuration
        ncoils = case_cfg.coils_params["ncoils"]
        coil_order = case_cfg.coils_params["order"]
        major_radius = surface.get_rc(0, 0)
        minor_component = abs(surface.get_rc(1, 0))
        coil_radius = max(0.25 * major_radius, 3.0 * minor_component)
        base_curves = create_equally_spaced_curves(
            ncoils, surface.nfp, stellsym=surface.stellsym,
            R0=major_radius, R1=coil_radius, order=coil_order, numquadpoints=512
        )
        base_currents = [Current(1.0e6 + 2.0e5 * i) for i in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, surface.stellsym)
        
        # Create BiotSavart object
        bs = BiotSavart(coils)
        
        # Create output directory
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Call plotting function (default output)
        _plot_bn_error_3d(surface, bs, coils, out_dir)
        
        # Verify PDF file was created
        pdf_path = out_dir / "bn_error_3d_plot.pdf"
        assert pdf_path.exists(), f"PDF file was not created at {pdf_path}"
        assert pdf_path.stat().st_size > 0, "PDF file is empty"
        
        # Call plotting function with custom filename/title (initial coils)
        _plot_bn_error_3d(
            surface,
            bs,
            coils,
            out_dir,
            filename="bn_error_3d_plot_initial.pdf",
            title="B_N/|B| Error on Plasma Surface with Initial Coils",
        )
        initial_pdf_path = out_dir / "bn_error_3d_plot_initial.pdf"
        assert initial_pdf_path.exists(), f"PDF file was not created at {initial_pdf_path}"
        assert initial_pdf_path.stat().st_size > 0, "PDF file is empty"

        # Also write copies to a stable artifacts directory for manual inspection
        artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "bn_error_3d_plot.pdf").write_bytes(pdf_path.read_bytes())
        (artifacts_dir / "bn_error_3d_plot_initial.pdf").write_bytes(initial_pdf_path.read_bytes())
    
    def test_plot_bn_error_3d_handles_missing_matplotlib(self, tmp_path, monkeypatch):
        """Test that _plot_bn_error_3d handles missing matplotlib gracefully."""
        # Mock matplotlib as unavailable
        import stellcoilbench.coil_optimization as co_module
        original_available = co_module.MATPLOTLIB_AVAILABLE
        co_module.MATPLOTLIB_AVAILABLE = False
        
        try:
            from simsopt.geo import SurfaceRZFourier
            from simsopt.field import BiotSavart
            from simsopt.geo import create_equally_spaced_curves
            from simsopt.field import Current, coils_via_symmetries
            
            # Create minimal objects with equal quadpoints
            import numpy as np
            surface = SurfaceRZFourier(
                nfp=1, stellsym=False, mpol=2, ntor=2,
                quadpoints_phi=np.linspace(0, 1, 16),
                quadpoints_theta=np.linspace(0, 1, 16)
            )
            surface.set_rc(0, 0, 1.0)
            surface.set_zs(0, 0, 0.0)
            
            ncoils = 2
            base_curves = create_equally_spaced_curves(
                ncoils, surface.nfp, stellsym=surface.stellsym,
                R0=1.0, R1=0.1, order=2, numquadpoints=64
            )
            base_currents = [Current(1e6) for _ in range(ncoils)]
            coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, surface.stellsym)
            bs = BiotSavart(coils)
            
            out_dir = tmp_path / "output"
            out_dir.mkdir()
            
            # Should not raise an error, just skip plotting
            _plot_bn_error_3d(surface, bs, coils, out_dir)
            
            # PDF should not be created
            pdf_path = out_dir / "bn_error_3d_plot.pdf"
            assert not pdf_path.exists(), "PDF should not be created when matplotlib is unavailable"
        finally:
            # Restore original value
            co_module.MATPLOTLIB_AVAILABLE = original_available
    
    def test_plot_bn_error_3d_with_full_torus_surface(self, tmp_path):
        """Test plotting with a full torus surface (more realistic case)."""
        pytest.importorskip("matplotlib")
        
        from simsopt.geo import SurfaceRZFourier
        from simsopt.field import BiotSavart
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries
        
        # Create surface with equal quadpoints_phi and quadpoints_theta
        # Use stellsym=False for full torus (not half due to symmetry)
        surface = SurfaceRZFourier(
            nfp=1, stellsym=False, mpol=2, ntor=2,
            quadpoints_phi=np.linspace(0, 1, 32),
            quadpoints_theta=np.linspace(0, 1, 32)
        )
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)
        
        # Create coils
        ncoils = 4
        base_curves = create_equally_spaced_curves(
            ncoils, surface.nfp, stellsym=surface.stellsym,
            R0=1.0, R1=0.1, order=4, numquadpoints=128
        )
        base_currents = [Current(1e6) for _ in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, surface.stellsym)
        
        # Create BiotSavart object
        bs = BiotSavart(coils)
        
        # Create output directory
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        
        # Call plotting function
        _plot_bn_error_3d(surface, bs, coils, out_dir)
        
        # Verify PDF file was created
        pdf_path = out_dir / "bn_error_3d_plot.pdf"
        assert pdf_path.exists(), f"PDF file was not created at {pdf_path}"
        assert pdf_path.stat().st_size > 0, "PDF file is empty"
        
        # Verify file is actually a PDF (check magic bytes)
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            assert header == b'%PDF', f"File does not appear to be a PDF (header: {header})"
