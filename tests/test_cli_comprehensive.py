"""
Comprehensive unit tests for CLI functions.
"""
import pytest
import json
import numpy as np
import subprocess
from unittest.mock import patch, MagicMock
from stellcoilbench.cli import (
    NumpyJSONEncoder,
    _detect_github_username,
    _zip_submission_directory,
    _detect_hardware,
    _write_autopilot_submission,
)


class TestNumpyJSONEncoder:
    """Tests for NumpyJSONEncoder class."""
    
    def test_encode_numpy_integer(self):
        """Test encoding numpy integer types."""
        encoder = NumpyJSONEncoder()
        assert encoder.default(np.int32(42)) == 42
        assert encoder.default(np.int64(100)) == 100
    
    def test_encode_numpy_float(self):
        """Test encoding numpy floating point types."""
        encoder = NumpyJSONEncoder()
        # Use approximate equality for float32 due to precision
        assert abs(encoder.default(np.float32(3.14)) - 3.14) < 1e-6
        assert encoder.default(np.float64(2.71)) == 2.71
    
    def test_encode_numpy_array(self):
        """Test encoding numpy arrays."""
        encoder = NumpyJSONEncoder()
        arr = np.array([1, 2, 3])
        result = encoder.default(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)
    
    def test_encode_numpy_bool(self):
        """Test encoding numpy boolean."""
        encoder = NumpyJSONEncoder()
        assert encoder.default(np.bool_(True)) is True
        assert encoder.default(np.bool_(False)) is False
    
    def test_encode_numpy_array_2d(self):
        """Test encoding 2D numpy arrays."""
        encoder = NumpyJSONEncoder()
        arr = np.array([[1, 2], [3, 4]])
        result = encoder.default(arr)
        assert result == [[1, 2], [3, 4]]
    
    def test_encode_array_like(self):
        """Test encoding array-like objects."""
        encoder = NumpyJSONEncoder()
        
        # Test with object that has __array__ method
        class ArrayLike:
            def __array__(self):
                return np.array([1, 2, 3])
        
        arr_like = ArrayLike()
        result = encoder.default(arr_like)
        assert result == [1, 2, 3]
    
    def test_encode_fallback(self):
        """Test that fallback to super().default() works."""
        encoder = NumpyJSONEncoder()
        # Should raise TypeError for non-numpy types
        with pytest.raises(TypeError):
            encoder.default("not numpy")
    
    def test_json_dumps_with_encoder(self):
        """Test using encoder with json.dumps."""
        data = {
            "int_val": np.int32(42),
            "float_val": np.float64(3.14),
            "array_val": np.array([1, 2, 3]),
            "bool_val": np.bool_(True),
        }
        json_str = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(json_str)
        assert parsed["int_val"] == 42
        assert parsed["float_val"] == 3.14
        assert parsed["array_val"] == [1, 2, 3]
        assert parsed["bool_val"] is True


class TestDetectGithubUsername:
    """Tests for _detect_github_username function."""
    
    def test_detect_from_https_url(self, monkeypatch):
        """Test detecting username from HTTPS GitHub URL."""
        def mock_run(cmd, **kwargs):
            mock_result = MagicMock()
            if cmd == ["git", "remote", "get-url", "origin"]:
                mock_result.returncode = 0
                mock_result.stdout = "https://github.com/testuser/testrepo.git\n"
            return mock_result
        
        with patch("subprocess.run", side_effect=mock_run):
            username = _detect_github_username()
            assert username == "testuser"
    
    def test_detect_from_ssh_url(self, monkeypatch):
        """Test detecting username from SSH GitHub URL."""
        def mock_run(cmd, **kwargs):
            mock_result = MagicMock()
            if cmd == ["git", "remote", "get-url", "origin"]:
                mock_result.returncode = 0
                mock_result.stdout = "git@github.com:testuser/testrepo.git\n"
            return mock_result
        
        with patch("subprocess.run", side_effect=mock_run):
            username = _detect_github_username()
            assert username == "testuser"
    
    def test_detect_from_env_var(self, monkeypatch):
        """Test detecting username from environment variable."""
        def mock_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 1  # Git command fails
            return mock_result
        
        with patch("subprocess.run", side_effect=mock_run):
            monkeypatch.setenv("GITHUB_ACTOR", "envuser")
            username = _detect_github_username()
            assert username == "envuser"
    
    def test_detect_from_github_user_env(self, monkeypatch):
        """Test detecting username from GITHUB_USER environment variable."""
        def mock_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 1
            return mock_result
        
        with patch("subprocess.run", side_effect=mock_run):
            monkeypatch.delenv("GITHUB_ACTOR", raising=False)
            monkeypatch.setenv("GITHUB_USER", "githubuser")
            username = _detect_github_username()
            assert username == "githubuser"
    
    def test_no_username_found(self, monkeypatch):
        """Test returning empty string when no username found."""
        def mock_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 1
            return mock_result
        
        with patch("subprocess.run", side_effect=mock_run):
            monkeypatch.delenv("GITHUB_ACTOR", raising=False)
            monkeypatch.delenv("GITHUB_USER", raising=False)
            username = _detect_github_username()
            assert username == ""
    
    def test_git_timeout(self, monkeypatch):
        """Test handling git command timeout."""
        def mock_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, timeout=2)
        
        with patch("subprocess.run", side_effect=mock_run):
            monkeypatch.delenv("GITHUB_ACTOR", raising=False)
            monkeypatch.delenv("GITHUB_USER", raising=False)
            username = _detect_github_username()
            assert username == ""
    
    def test_git_not_found(self, monkeypatch):
        """Test handling when git command is not found."""
        def mock_run(cmd, **kwargs):
            raise FileNotFoundError("git command not found")
        
        with patch("subprocess.run", side_effect=mock_run):
            monkeypatch.delenv("GITHUB_ACTOR", raising=False)
            monkeypatch.delenv("GITHUB_USER", raising=False)
            username = _detect_github_username()
            assert username == ""
    
    def test_non_github_url(self, monkeypatch):
        """Test handling non-GitHub remote URL."""
        def mock_run(cmd, **kwargs):
            mock_result = MagicMock()
            if cmd == ["git", "remote", "get-url", "origin"]:
                mock_result.returncode = 0
                mock_result.stdout = "https://gitlab.com/user/repo.git\n"
            return mock_result
        
        with patch("subprocess.run", side_effect=mock_run):
            monkeypatch.delenv("GITHUB_ACTOR", raising=False)
            monkeypatch.delenv("GITHUB_USER", raising=False)
            username = _detect_github_username()
            assert username == ""


class TestZipSubmissionDirectory:
    """Tests for _zip_submission_directory function."""
    
    def test_zip_creates_archive(self, tmp_path):
        """Test that zip file is created."""
        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        
        # Create some files
        (submission_dir / "coils.json").write_text('{"test": "data"}')
        (submission_dir / "results.json").write_text('{"results": "data"}')
        (submission_dir / "file.vtu").write_text("VTK data")
        
        zip_path = _zip_submission_directory(submission_dir)
        
        assert zip_path.exists()
        assert zip_path.name == "all_files.zip"
        assert zip_path.parent == submission_dir
    
    def test_zip_excludes_pdfs(self, tmp_path):
        """Test that PDF files are excluded from zip."""
        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        
        # Create files including PDFs
        (submission_dir / "coils.json").write_text('{"test": "data"}')
        (submission_dir / "plot.pdf").write_text("PDF content")
        (submission_dir / "another.pdf").write_text("More PDF content")
        
        zip_path = _zip_submission_directory(submission_dir)
        
        # PDFs should still exist in directory
        assert (submission_dir / "plot.pdf").exists()
        assert (submission_dir / "another.pdf").exists()
        
        # Check zip contents
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            files_in_zip = zf.namelist()
            assert "coils.json" in files_in_zip
            assert "plot.pdf" not in files_in_zip
            assert "another.pdf" not in files_in_zip
    
    def test_zip_empty_directory(self, tmp_path):
        """Test zipping empty directory."""
        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        
        zip_path = _zip_submission_directory(submission_dir)
        
        # Zip should not be created for empty directory
        assert not zip_path.exists()
    
    def test_zip_only_pdfs(self, tmp_path):
        """Test zipping directory with only PDFs."""
        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        
        (submission_dir / "plot.pdf").write_text("PDF content")
        (submission_dir / "another.pdf").write_text("More PDF content")
        
        zip_path = _zip_submission_directory(submission_dir)
        
        # Zip should not be created if only PDFs
        assert not zip_path.exists()
    
    def test_zip_includes_vtk_files(self, tmp_path):
        """Test that VTK files are included in zip."""
        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        
        (submission_dir / "coils.vtu").write_text("VTK data")
        (submission_dir / "surface.vts").write_text("VTK surface data")
        
        zip_path = _zip_submission_directory(submission_dir)
        
        assert zip_path.exists()
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            files_in_zip = zf.namelist()
            assert "coils.vtu" in files_in_zip
            assert "surface.vts" in files_in_zip


class TestDetectHardware:
    """Tests for _detect_hardware function."""
    
    def test_detect_cpu(self, monkeypatch):
        """Test detecting CPU hardware."""
        with patch("platform.processor", return_value="Intel Core i7"):
            hardware = _detect_hardware()
            assert "Intel Core i7" in hardware or "CPU" in hardware
    
    def test_detect_platform(self, monkeypatch):
        """Test that platform information is included."""
        hardware = _detect_hardware()
        # Should include some platform information
        assert len(hardware) > 0
    
    @patch("platform.processor")
    @patch("platform.machine")
    @patch("platform.system")
    def test_detect_various_platforms(self, mock_system, mock_machine, mock_processor):
        """Test hardware detection on various platforms."""
        mock_processor.return_value = "x86_64"
        mock_machine.return_value = "x86_64"
        mock_system.return_value = "Linux"
        
        hardware = _detect_hardware()
        assert len(hardware) > 0
        
        mock_system.return_value = "Darwin"
        hardware = _detect_hardware()
        assert len(hardware) > 0
        
        mock_system.return_value = "Windows"
        hardware = _detect_hardware()
        assert len(hardware) > 0


class TestWriteAutopilotSubmission:
    """Tests for _write_autopilot_submission function."""

    def _make_results_dict(self):
        """Minimal results_dict for testing."""
        return {
            "target_B_field": 1.0,
            "final_squared_flux": 1e-6,
            "final_min_cs_separation": 0.3,
            "final_min_cc_separation": 0.08,
            "final_total_length": 10.0,
            "final_max_curvature": 4.0,
            "final_average_curvature": 2.0,
            "final_mean_squared_curvature": 5.0,
            "final_arclength_variation": 0.001,
            "final_linking_number": 0,
            "coils_linked_to_surface": True,
            "avg_BdotN_over_B": 0.0004,
            "max_BdotN_over_B": 0.002,
            "_cached_thresholds": {"major_radius": 1.7, "minor_radius": 0.29, "a0": 5.88},
        }

    def _make_case_config(self, fourier_continuation=False):
        """Minimal case config for testing."""
        cfg = {
            "surface_params": {"surface": "input.LandremanPaul2021_QA", "range": "half period"},
            "coils_params": {"ncoils": 4, "order": 4},
            "optimizer_params": {"algorithm": "augmented_lagrangian", "max_iterations": 100},
            "coil_objective_terms": {"total_length": "l2_threshold"},
        }
        if fourier_continuation:
            cfg["fourier_continuation"] = {"enabled": True, "orders": [4, 8, 16]}
        return cfg

    def test_basic_submission_created(self, tmp_path):
        """Test that a basic submission is created correctly."""
        case_output = tmp_path / "case_output"
        case_output.mkdir()
        (case_output / "poincare_plot.png").write_bytes(b"png")

        _write_autopilot_submission(
            case_id="test_001",
            results_dict=self._make_results_dict(),
            case_cfg=None,
            case_config_dict=self._make_case_config(),
            walltime=10.0,
            repo_root=tmp_path,
            case_output_dir=case_output,
        )

        sub_dir = tmp_path / "submissions" / "LandremanPaul2021_QA" / "auto" / "test_001"
        assert (sub_dir / "results.json").exists()
        assert (sub_dir / "case.yaml").exists()
        assert (sub_dir / "poincare_plot.png").exists()

    def test_fc_plots_copied_from_order_dirs(self, tmp_path):
        """FC submissions copy plots from order_X/ subdirectories."""
        case_output = tmp_path / "case_output"
        case_output.mkdir()
        # Poincare is at top level
        (case_output / "poincare_plot.png").write_bytes(b"pp")
        # Bn plots are in order directories (typical for FC)
        for order in [4, 8, 16]:
            d = case_output / f"order_{order}"
            d.mkdir()
            (d / "bn_error_3d_plot.pdf").write_bytes(f"bn_{order}".encode())
            (d / "bn_error_3d_plot_initial.pdf").write_bytes(f"bni_{order}".encode())
            (d / "biot_savart_optimized.json").write_text("{}")

        _write_autopilot_submission(
            case_id="fc_test",
            results_dict=self._make_results_dict(),
            case_cfg=None,
            case_config_dict=self._make_case_config(fourier_continuation=True),
            walltime=10.0,
            repo_root=tmp_path,
            case_output_dir=case_output,
        )

        sub_dir = tmp_path / "submissions" / "LandremanPaul2021_QA" / "auto" / "fc_test"
        # Top-level plots should come from highest order dir
        assert (sub_dir / "bn_error_3d_plot.pdf").exists()
        assert (sub_dir / "bn_error_3d_plot.pdf").read_bytes() == b"bn_16"
        assert (sub_dir / "poincare_plot.png").exists()
        # Order directories should be copied
        assert (sub_dir / "order_4" / "bn_error_3d_plot.pdf").exists()
        assert (sub_dir / "order_8" / "bn_error_3d_plot.pdf").exists()
        assert (sub_dir / "order_16" / "bn_error_3d_plot.pdf").exists()

    def test_fc_no_order_dirs_falls_back(self, tmp_path):
        """FC config with no order dirs on disk still works."""
        case_output = tmp_path / "case_output"
        case_output.mkdir()
        (case_output / "bn_error_3d_plot.pdf").write_bytes(b"top")

        _write_autopilot_submission(
            case_id="fc_fallback",
            results_dict=self._make_results_dict(),
            case_cfg=None,
            case_config_dict=self._make_case_config(fourier_continuation=True),
            walltime=10.0,
            repo_root=tmp_path,
            case_output_dir=case_output,
        )

        sub_dir = tmp_path / "submissions" / "LandremanPaul2021_QA" / "auto" / "fc_fallback"
        assert (sub_dir / "bn_error_3d_plot.pdf").read_bytes() == b"top"
