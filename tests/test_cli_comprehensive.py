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
