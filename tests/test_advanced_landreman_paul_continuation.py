"""
Test for advanced_LandremanPaulQA.yaml with Fourier continuation.

This test verifies that the advanced case file works correctly with
Fourier continuation enabled, and that VTK and PDF files are saved
after each optimization step.
"""
import pytest
import numpy as np
from pathlib import Path
from stellcoilbench.evaluate import load_case_config
from stellcoilbench.coil_optimization import optimize_coils


class TestAdvancedLandremanPaulContinuation:
    """Tests for advanced_LandremanPaulQA.yaml with Fourier continuation."""
    
    @pytest.fixture
    def repo_root(self):
        """Get the repository root directory."""
        return Path(__file__).parent.parent
    
    @pytest.fixture
    def case_file(self, repo_root):
        """Get the advanced case file."""
        return repo_root / "cases" / "advanced_LandremanPaulQA.yaml"
    
    def test_case_loads_with_continuation(self, case_file):
        """Test that the case file loads correctly with Fourier continuation."""
        config = load_case_config(case_file)
        
        assert config.fourier_continuation is not None
        assert config.fourier_continuation["enabled"] is True
        assert config.fourier_continuation["orders"] == [4, 6, 8]
        assert config.coils_params["ncoils"] == 4
        assert config.surface_params["surface"] == "input.LandremanPaul2021_QA"
    
    def test_continuation_runs_and_saves_files(self, case_file, tmp_path, repo_root):
        """Test that continuation runs and saves VTK/PDF files after each step."""
        # Create a temporary case directory structure
        case_dir = tmp_path / "test_case"
        case_dir.mkdir()
        
        # Copy case.yaml to temp directory
        import shutil
        shutil.copy(case_file, case_dir / "case.yaml")
        
        # Create plasma_surfaces directory and dummy surface file
        plasma_dir = repo_root / "plasma_surfaces"
        if not plasma_dir.exists():
            pytest.skip("plasma_surfaces directory not found")
        
        surface_file = plasma_dir / "input.LandremanPaul2021_QA"
        if not surface_file.exists():
            pytest.skip(f"Surface file {surface_file} not found")
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Load config
        config = load_case_config(case_dir)
        
        # Modify config to use fewer iterations for faster testing
        config.optimizer_params["max_iterations"] = 5
        config.optimizer_params["max_iter_subopt"] = 2
        config.optimizer_params["verbose"] = False
        
        # Run optimization with continuation
        coils_out_path = output_dir / "coils.json"
        
        results = optimize_coils(
            case_path=case_dir,
            coils_out_path=coils_out_path,
            case_cfg=config,
            output_dir=output_dir,
        )
        
        # Verify results
        assert results is not None
        assert results.get('fourier_continuation') is True
        assert results.get('fourier_orders') == [4, 6, 8]
        assert results.get('final_order') == 8
        assert len(results.get('continuation_results', [])) == 3
        
        # Verify that subdirectories were created
        assert (output_dir / "order_4").exists()
        assert (output_dir / "order_6").exists()
        assert (output_dir / "order_8").exists()
        
        # Verify that VTK files exist in each order directory
        for order in [4, 6, 8]:
            order_dir = output_dir / f"order_{order}"
            # Check for VTK files (coils_optimized.vtu or similar)
            vtk_files = list(order_dir.glob("coils_optimized.*"))
            assert len(vtk_files) > 0, f"No VTK files found in {order_dir}"
            
            # Check for Bn error PDF
            pdf_file = order_dir / "bn_error_3d_plot.pdf"
            assert pdf_file.exists(), f"Bn error PDF not found in {order_dir}"
        
        # Verify that coils.json was created
        assert coils_out_path.exists()
    
    def test_continuation_results_structure(self, case_file, tmp_path, repo_root):
        """Test that continuation results have the correct structure."""
        case_dir = tmp_path / "test_case"
        case_dir.mkdir()
        
        import shutil
        shutil.copy(case_file, case_dir / "case.yaml")
        
        plasma_dir = repo_root / "plasma_surfaces"
        if not plasma_dir.exists():
            pytest.skip("plasma_surfaces directory not found")
        
        surface_file = plasma_dir / "input.LandremanPaul2021_QA"
        if not surface_file.exists():
            pytest.skip(f"Surface file {surface_file} not found")
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        config = load_case_config(case_dir)
        config.optimizer_params["max_iterations"] = 3
        config.optimizer_params["max_iter_subopt"] = 1
        config.optimizer_params["verbose"] = False
        
        coils_out_path = output_dir / "coils.json"
        
        results = optimize_coils(
            case_path=case_dir,
            coils_out_path=coils_out_path,
            case_cfg=config,
            output_dir=output_dir,
        )
        
        # Check structure of continuation results
        continuation_results = results.get('continuation_results', [])
        assert len(continuation_results) == 3
        
        for i, step_result in enumerate(continuation_results):
            assert step_result['fourier_order'] == [4, 6, 8][i]
            assert step_result['continuation_step'] == i + 1
            assert 'final_normalized_squared_flux' in step_result
            assert 'final_B_field' in step_result
            assert 'optimization_time' in step_result
            
            # Verify values are finite
            assert np.isfinite(step_result['final_normalized_squared_flux'])
            assert np.isfinite(step_result['final_B_field'])
