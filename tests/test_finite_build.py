"""Unit tests for finite-build coil geometry generation."""

import numpy as np
import pytest

from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries

from stellcoilbench.finite_build import (
    _compute_cross_section_frame,
    sweep_rectangular_cross_section,
    finite_build_coils_to_vtk,
)


class TestComputeCrossSectionFrame:
    """Tests for _compute_cross_section_frame."""

    def test_tangent_z_reference_orthogonal(self):
        """Frame vectors should be orthonormal."""
        tangent = np.array([1.0, 0.0, 0.0])
        normal, binormal = _compute_cross_section_frame(tangent)
        assert np.isclose(np.dot(tangent, normal), 0.0)
        assert np.isclose(np.dot(tangent, binormal), 0.0)
        assert np.isclose(np.dot(normal, binormal), 0.0)
        assert np.isclose(np.linalg.norm(normal), 1.0)
        assert np.isclose(np.linalg.norm(binormal), 1.0)

    def test_tangent_parallel_to_z_uses_x_fallback(self):
        """When tangent is parallel to Z, use X-axis as reference."""
        tangent = np.array([0.0, 0.0, 1.0])
        normal, binormal = _compute_cross_section_frame(tangent)
        assert np.isclose(np.dot(tangent, normal), 0.0)
        assert np.isclose(np.dot(tangent, binormal), 0.0)
        assert np.isclose(np.linalg.norm(normal), 1.0)
        assert np.isclose(np.linalg.norm(binormal), 1.0)


class TestSweepRectangularCrossSection:
    """Tests for sweep_rectangular_cross_section."""

    def test_circle_sweep_produces_torus(self):
        """Sweeping a rectangle along a circle should produce a toroidal surface."""
        n = 32
        R = 2.0
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        gamma = np.column_stack([
            R * np.cos(theta),
            R * np.sin(theta),
            np.zeros(n),
        ])
        gammadash = np.column_stack([
            -R * np.sin(theta) * (2 * np.pi / n) * n,
            R * np.cos(theta) * (2 * np.pi / n) * n,
            np.zeros(n),
        ])
        width, height = 0.1, 0.1
        vertices, faces = sweep_rectangular_cross_section(
            gamma, gammadash, width=width, height=height
        )
        # n+1 cross-sections (first point appended to close loop) × 4 corners
        assert vertices.shape[0] == (n + 1) * 4
        assert faces.shape[0] >= n * 8  # n segments × 4 quads × 2 triangles per quad
        assert faces.shape[1] == 3
        # Check vertices are roughly at radius R
        radii = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2)
        assert np.all(radii > R - 0.1) and np.all(radii < R + 0.1)

    def test_gamma_gammadash_length_mismatch_raises(self):
        """Mismatched gamma and gammadash lengths should raise."""
        gamma = np.random.randn(10, 3)
        gammadash = np.random.randn(5, 3)
        with pytest.raises(ValueError, match="same length"):
            sweep_rectangular_cross_section(gamma, gammadash, 0.1, 0.1)


class TestFiniteBuildCoilsToVtk:
    """Tests for finite_build_coils_to_vtk."""

    @pytest.fixture
    def simple_coils(self):
        """Create a minimal coil set for testing."""
        base_curves = create_equally_spaced_curves(
            2, 1, stellsym=False,
            R0=1.7, R1=0.3, order=4, numquadpoints=64
        )
        base_currents = [Current(1e6), Current(-1e6)]
        return coils_via_symmetries(base_curves, base_currents, 1, False)

    def test_writes_vtk_file(self, simple_coils, tmp_path):
        """finite_build_coils_to_vtk should write a valid VTK file."""
        out_path = finite_build_coils_to_vtk(
            simple_coils,
            tmp_path / "finite_build_coils",
            width=0.02,
            height=0.02,
        )
        assert out_path.exists()
        assert out_path.suffix == ".vtk"
        content = out_path.read_text()
        assert "vtk DataFile" in content
        assert "POINTS" in content
        assert "CELLS" in content

    def test_empty_coils_raises(self, tmp_path):
        """Empty coil list should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            finite_build_coils_to_vtk([], tmp_path / "out")

    def test_stellaris_default_dimensions(self, simple_coils, tmp_path):
        """Default dimensions should use 5 cm."""
        out_path = finite_build_coils_to_vtk(
            simple_coils,
            tmp_path / "out",
            use_stellaris_default=True,
        )
        assert out_path.exists()
        # Verify file has reasonable size (vertices from 2 coils)
        content = out_path.read_text()
        assert "POINTS" in content

    def test_per_coil_dimensions(self, simple_coils, tmp_path):
        """width_per_coil and height_per_coil should apply per coil."""
        out_path = finite_build_coils_to_vtk(
            simple_coils,
            tmp_path / "out",
            width_per_coil=[0.01, 0.02],
            height_per_coil=[0.01, 0.02],
        )
        assert out_path.exists()

    def test_per_coil_wrong_length_raises(self, simple_coils, tmp_path):
        """Wrong length for width_per_coil should raise."""
        with pytest.raises(ValueError, match="length equal to number of coils"):
            finite_build_coils_to_vtk(
                simple_coils,
                tmp_path / "out",
                width_per_coil=[0.01],
                height_per_coil=[0.01],
            )

    def test_use_parastell_false_uses_sweep(self, simple_coils, tmp_path):
        """With use_parastell=False, built-in sweep is used; output has no _parastell suffix."""
        out_path = finite_build_coils_to_vtk(
            simple_coils,
            tmp_path / "sweep_only",
            width=0.02,
            height=0.02,
            use_parastell=False,
        )
        assert out_path.exists()
        assert out_path.stem == "sweep_only"
        assert "_parastell" not in out_path.name
        content = out_path.read_text()
        assert "POINTS" in content
        assert "CELLS" in content
        assert "CELL_TYPES" in content
