"""
Unit tests for coil_optimization.py

Note: Full optimization tests require simsopt and are integration tests.
These unit tests focus on functions that can be tested without running optimizations.
"""
import pytest
import numpy as np
from pathlib import Path
from stellcoilbench.coil_optimization import (
    load_coils_config,
    _plot_bn_error_3d,
    optimize_coils,
    _zip_output_files,
    initialize_coils_loop,
)
from stellcoilbench.config_scheme import CaseConfig
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


class TestOptimizeCoils:
    """Tests for optimize_coils path handling and argument plumbing."""

    def test_optimize_coils_resolves_surface_and_target_b(self, tmp_path, monkeypatch):
        plasma_dir = tmp_path / "plasma_surfaces"
        plasma_dir.mkdir()
        surface_file = plasma_dir / "INPUT.LandremanPaul2021_QA"
        surface_file.write_text("dummy")

        case_cfg = CaseConfig.from_dict({
            "description": "test",
            "surface_params": {"surface": "input.LandremanPaul2021_QA", "range": "full torus"},
            "coils_params": {"ncoils": 2, "order": 2},
            "optimizer_params": {"algorithm": "augmented_lagrangian", "algorithm_options": {"maxiter": 5}},
            "coil_objective_terms": {"total_length": "l2"},
        })

        monkeypatch.chdir(tmp_path)
        calls = {}

        def fake_from_vmec_input(filename, range, nphi, ntheta):
            calls["surface_filename"] = filename
            calls["surface_range"] = range
            return object()

        def fake_optimize(surface, **kwargs):
            calls["optimize_kwargs"] = kwargs
            return ["coil"], {"ok": True}

        def fake_save(coils, path):
            calls["save_path"] = path

        monkeypatch.setattr(
            "stellcoilbench.coil_optimization.SurfaceRZFourier.from_vmec_input",
            fake_from_vmec_input,
        )
        monkeypatch.setattr("stellcoilbench.coil_optimization.optimize_coils_loop", fake_optimize)
        monkeypatch.setattr("simsopt.save", fake_save)

        out_dir = tmp_path / "out"
        results = optimize_coils(
            case_path=tmp_path,
            coils_out_path=tmp_path / "coils.out",
            case_cfg=case_cfg,
            output_dir=out_dir,
        )

        assert results == {"ok": True}
        assert "LandremanPaul2021_QA" in calls["surface_filename"]
        assert calls["surface_range"] == "full torus"
        assert calls["optimize_kwargs"]["target_B"] == 1.0
        assert calls["optimize_kwargs"]["algorithm_options"] == {"maxiter": 5}
        assert calls["optimize_kwargs"]["output_dir"] == str(out_dir.resolve())
        assert calls["save_path"].name == "coils.json"

    def test_optimize_coils_fallback_without_output_dir(self, tmp_path, monkeypatch):
        surface_file = tmp_path / "input.W7-X"
        surface_file.write_text("dummy")

        case_cfg = CaseConfig.from_dict({
            "description": "test",
            "surface_params": {"surface": str(surface_file), "range": "full torus"},
            "coils_params": {"ncoils": 2, "order": 2},
            "optimizer_params": {"algorithm": "augmented_lagrangian"},
        })

        calls = {"count": 0}

        def fake_from_vmec_input(filename, range, nphi, ntheta):
            return object()

        def fake_optimize(surface, **kwargs):
            calls["count"] += 1
            if "output_dir" in kwargs:
                raise TypeError("no output_dir")
            return ["coil"], {"ok": True}

        def fake_save(coils, path):
            calls["save_path"] = path

        monkeypatch.setattr(
            "stellcoilbench.coil_optimization.SurfaceRZFourier.from_vmec_input",
            fake_from_vmec_input,
        )
        monkeypatch.setattr("stellcoilbench.coil_optimization.optimize_coils_loop", fake_optimize)
        monkeypatch.setattr("simsopt.save", fake_save)

        optimize_coils(
            case_path=tmp_path,
            coils_out_path=tmp_path / "coils.json",
            case_cfg=case_cfg,
        )

        assert calls["count"] == 2
        assert calls["save_path"].name == "coils.json"

    def test_optimize_coils_surface_type_branches(self, tmp_path, monkeypatch):
        calls = {"wout": 0, "focus": 0}

        def fake_from_wout(filename, range, nphi, ntheta):
            calls["wout"] += 1
            return object()

        def fake_from_focus(filename, range, nphi, ntheta):
            calls["focus"] += 1
            return object()

        def fake_optimize(surface, **kwargs):
            return ["coil"], {"ok": True}

        def fake_save(coils, path):
            pass

        monkeypatch.setattr("stellcoilbench.coil_optimization.SurfaceRZFourier.from_wout", fake_from_wout)
        monkeypatch.setattr("stellcoilbench.coil_optimization.SurfaceRZFourier.from_focus", fake_from_focus)
        monkeypatch.setattr("stellcoilbench.coil_optimization.optimize_coils_loop", fake_optimize)
        monkeypatch.setattr("simsopt.save", fake_save)

        wout_case = CaseConfig.from_dict({
            "description": "test",
            "surface_params": {"surface": str(tmp_path / "wout.W7-X"), "range": "full torus"},
            "coils_params": {"ncoils": 2, "order": 2},
            "optimizer_params": {"algorithm": "augmented_lagrangian"},
        })
        focus_case = CaseConfig.from_dict({
            "description": "test",
            "surface_params": {"surface": str(tmp_path / "focus.HSX"), "range": "full torus"},
            "coils_params": {"ncoils": 2, "order": 2},
            "optimizer_params": {"algorithm": "augmented_lagrangian"},
        })

        optimize_coils(tmp_path, tmp_path / "coils1.json", case_cfg=wout_case)
        optimize_coils(tmp_path, tmp_path / "coils2.json", case_cfg=focus_case)

        assert calls["wout"] == 1
        assert calls["focus"] == 1

    def test_optimize_coils_unknown_surface_type(self, tmp_path, monkeypatch):
        case_cfg = CaseConfig.from_dict({
            "description": "test",
            "surface_params": {"surface": str(tmp_path / "unknown.dat"), "range": "full torus"},
            "coils_params": {"ncoils": 2, "order": 2},
            "optimizer_params": {"algorithm": "augmented_lagrangian"},
        })

        with pytest.raises(ValueError, match="Unknown surface type"):
            optimize_coils(tmp_path, tmp_path / "coils.json", case_cfg=case_cfg)

    def test_optimize_coils_unknown_target_b_surface(self, tmp_path, monkeypatch):
        surface_file = tmp_path / "input.Unknown"
        surface_file.write_text("dummy")

        case_cfg = CaseConfig.from_dict({
            "description": "test",
            "surface_params": {"surface": str(surface_file), "range": "full torus"},
            "coils_params": {"ncoils": 2, "order": 2},
            "optimizer_params": {"algorithm": "augmented_lagrangian"},
        })

        def fake_from_vmec_input(filename, range, nphi, ntheta):
            return object()

        monkeypatch.setattr(
            "stellcoilbench.coil_optimization.SurfaceRZFourier.from_vmec_input",
            fake_from_vmec_input,
        )

        with pytest.raises(ValueError, match="Unknown surface file"):
            optimize_coils(tmp_path, tmp_path / "coils.json", case_cfg=case_cfg)


class TestZipOutputFiles:
    """Tests for _zip_output_files helper."""

    def test_zip_output_files_creates_archive(self, tmp_path):
        vtu = tmp_path / "file.vtu"
        vts = tmp_path / "file.vts"
        vtu.write_text("data")
        vts.write_text("data")

        zip_path = _zip_output_files(tmp_path)
        assert zip_path.exists()
        assert not vtu.exists()
        assert not vts.exists()

    def test_zip_output_files_no_vtk(self, tmp_path):
        zip_path = _zip_output_files(tmp_path)
        assert not zip_path.exists()


class TestInitializeCoilsLoop:
    """Tests for initialize_coils_loop helper."""

    def test_initialize_coils_loop_no_regularization(self, monkeypatch, tmp_path):
        from simsopt.geo import SurfaceRZFourier
        import simsopt.util.coil_optimization_helper_functions as cohf

        surface = SurfaceRZFourier(nfp=1, stellsym=True, mpol=2, ntor=2)
        surface.set_rc(0, 0, 1.0)
        surface.set_zs(0, 0, 0.0)

        monkeypatch.setattr(cohf, "calculate_modB_on_major_radius", lambda bs, s: 1.0)

        coils = initialize_coils_loop(
            s=surface,
            out_dir=tmp_path,
            target_B=1.0,
            ncoils=2,
            order=2,
            regularization=None,
        )
        assert len(coils) > 0
