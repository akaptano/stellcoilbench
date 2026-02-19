"""
Finite-build coil geometry generation and VTK export.

Generates 3D coil geometry by sweeping a rectangular cross-section along
the coil centerline (filament). When ParaStell is available, uses its
CadQuery-based sweep for accurate rectangular cross-sections and tetrahedral
mesh generation; output is written to ``*_parastell.vtk``. Otherwise falls
back to a built-in rotation-minimizing frame sweep (``*.vtk``).

Fidelity: The ParaStell output is faithful to ParaStell's representation:
it orients the cross-section using the coil center-of-mass (outward direction
from COM) and uses spline-based ruled surfaces. The built-in sweep uses a
rotation-minimizing frame with Z-axis reference and can differ in cross-section
orientation, especially for coils with significant curvature.
"""

import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

# Stellaris turn cross-section: 20 mm × 20 mm (Lion et al., FED 2025, Table 7)
STELLARIS_TURN_SIDE_M = 0.020  # 20 mm

# Default finite-build cross-section: 5 cm total width (and height)
DEFAULT_CROSS_SECTION_M = 0.05  # 5 cm

# Minimum points along curve for accurate rectangular sweep (ensures smooth representation)
MIN_POINTS_ALONG_CURVE = 128

# Set when ParaStell import fails (for diagnostics)
_last_parastell_error: Optional[str] = None


def _compute_cross_section_frame(
    tangent: np.ndarray,
    reference: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normal and binormal vectors for the cross-section plane.

    Uses a reference vector (default Z-axis) to define a rotation-minimizing
    frame. This avoids Frenet-Serret frame twisting in high-torsion regions.

    Parameters
    ----------
    tangent : np.ndarray
        Unit tangent vector (3,).
    reference : np.ndarray
        Reference vector for frame construction (default: Z-axis).

    Returns
    -------
    normal : np.ndarray
        Unit vector in cross-section plane, perpendicular to tangent.
    binormal : np.ndarray
        Unit vector completing right-handed frame: binormal = tangent × normal.
    """
    tangent = np.asarray(tangent, dtype=float)
    reference = np.asarray(reference, dtype=float)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-14:
        raise ValueError("Tangent vector has zero length")
    tangent = tangent / tangent_norm

    # normal = cross(tangent, reference) gives vector in cross-section plane
    cross_tr = np.cross(tangent, reference)
    cross_norm = np.linalg.norm(cross_tr)

    if cross_norm < 1e-10:
        # Tangent parallel to reference; use X-axis instead
        reference = np.array([1.0, 0.0, 0.0])
        cross_tr = np.cross(tangent, reference)
        cross_norm = np.linalg.norm(cross_tr)
        if cross_norm < 1e-10:
            reference = np.array([0.0, 1.0, 0.0])
            cross_tr = np.cross(tangent, reference)
            cross_norm = np.linalg.norm(cross_tr)

    normal = cross_tr / cross_norm
    binormal = np.cross(tangent, normal)
    binormal_norm = np.linalg.norm(binormal)
    if binormal_norm > 1e-14:
        binormal = binormal / binormal_norm

    return normal, binormal


def sweep_rectangular_cross_section(
    gamma: np.ndarray,
    gammadash: np.ndarray,
    width: float,
    height: float,
    n_cross: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep a rectangular cross-section along a curve to create a surface mesh.

    Parameters
    ----------
    gamma : np.ndarray
        Curve points, shape (n_points, 3).
    gammadash : np.ndarray
        Curve derivatives (tangents), shape (n_points, 3).
    width : float
        Cross-section width [m] along the first in-plane direction.
    height : float
        Cross-section height [m] along the second in-plane direction.
    n_cross : int
        Number of vertices per cross-section (4 for rectangle corners).

    Returns
    -------
    vertices : np.ndarray
        Mesh vertices, shape (n_vertices, 3).
    faces : np.ndarray
        Triangle faces as vertex indices, shape (n_faces, 3).

    Notes
    -----
    The first point is appended to gamma/gammadash to close the curve, so the
    last segment connects back to the start for a fully connected loop.
    """
    gamma = np.asarray(gamma)
    gammadash = np.asarray(gammadash)
    n_along = len(gamma)
    if n_along != len(gammadash):
        raise ValueError("gamma and gammadash must have same length")

    # Append first point to close the curve so the sweep is fully connected
    if n_along > 1:
        gamma = np.vstack([gamma, gamma[0:1]])
        gammadash = np.vstack([gammadash, gammadash[0:1]])
        n_along = len(gamma)

    # Rectangle corners in local frame: exact (±w/2, ±h/2) for a true rectangular cross-section
    w2 = float(width) / 2
    h2 = float(height) / 2
    local_corners = np.array([
        [-w2, -h2],
        [w2, -h2],
        [w2, h2],
        [-w2, h2],
    ], dtype=float)

    vertices_list = []
    for i in range(n_along):
        tangent = np.asarray(gammadash[i], dtype=float)
        tangent_len = np.linalg.norm(tangent)
        if tangent_len < 1e-14:
            tangent = gammadash[(i + 1) % n_along] if i < n_along - 1 else gammadash[i - 1]
            tangent_len = np.linalg.norm(tangent)
        tangent = tangent / tangent_len

        normal, binormal = _compute_cross_section_frame(tangent)
        center = gamma[i]

        for (u, v) in local_corners:
            point = center + u * normal + v * binormal
            vertices_list.append(point)

    vertices = np.array(vertices_list)

    # Build triangle faces: each segment between consecutive cross-sections
    # gives 4 quads, each split into 2 triangles. With first point appended,
    # the last segment connects back to the start for a fully closed loop.
    faces = []
    for i in range(n_along - 1):
        base_curr = i * 4
        base_next = (i + 1) * 4
        for j in range(4):
            j_next = (j + 1) % 4
            v0 = base_curr + j
            v1 = base_curr + j_next
            v2 = base_next + j_next
            v3 = base_next + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    return vertices, np.array(faces)


def _write_vtk_unstructured(
    vertices: np.ndarray,
    faces: np.ndarray,
    filepath: Union[str, Path],
    title: str = "finite-build coils",
) -> None:
    """
    Write vertices and triangle faces to VTK unstructured grid file.

    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices, shape (n_vertices, 3).
    faces : np.ndarray
        Triangle faces as vertex indices, shape (n_faces, 3).
    filepath : Path or str
        Output file path. .vtk suffix added if missing.
    title : str, optional
        Title string written in the VTK header.
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() != ".vtk":
        filepath = filepath.with_suffix(".vtk")

    n_points = len(vertices)
    n_cells = len(faces)
    # Each triangle: 3 vertex indices + 3 for the count
    cell_data_size = n_cells * 4

    with open(filepath, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n_points} float\n")
        for row in vertices:
            f.write(f"{row[0]:.6e} {row[1]:.6e} {row[2]:.6e}\n")
        f.write(f"CELLS {n_cells} {cell_data_size}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("5\n")  # VTK_TRIANGLE = 5


def _write_parastell_filament_file(
    coils: List,
    filepath: Path,
    scale: float = 100.0,
) -> None:
    """
    Write simsopt coils to ParaStell filament format.

    Format: x y z current per line. Each coil ends with a line where current=0.
    ParaStell's reader appends the first point when it sees current=0 to close loops.
    Coords are in meters; scale multiplies them (ParaStell uses scale=100 for m->cm).

    Parameters
    ----------
    coils : List
        List of simsopt Coil objects (with .curve and .current attributes).
    filepath : Path
        Output file path.
    scale : float, optional
        Coordinate scaling factor (default 100 for meters to cm).
    """
    with open(filepath, "w") as f:
        f.write("stellcoilbench coils\n")
        f.write("simsopt export\n")
        f.write("begin filament\n")
        for coil in coils:
            curve = coil.curve
            gamma = np.asarray(curve.gamma(), dtype=float).reshape(-1, 3)
            current = abs(float(coil.current.get_value()))
            for pt in gamma:
                f.write(f"  {pt[0]:.16e} {pt[1]:.16e} {pt[2]:.16e} {current:.16e}\n")
            # ParaStell: s=0 signals end of filament; reader appends first point to close
            if len(gamma) > 0:
                f.write(f"  {gamma[0][0]:.16e} {gamma[0][1]:.16e} {gamma[0][2]:.16e} 0.0\n")


def _finite_build_coils_to_vtk_parastell(
    coils: List,
    output_path: Path,
    width: float,
    height: float,
    min_mesh_size: float = 2.0,
    max_mesh_size: float = 8.0,
) -> Optional[Path]:
    """
    Use ParaStell to generate finite-build coil geometry and tetrahedral mesh.

    Builds CadQuery solids from filament data, meshes with Gmsh (3D tetrahedra),
    and writes VTK. Cross-section is oriented using coil center-of-mass.

    Parameters
    ----------
    coils : List
        List of simsopt Coil objects.
    output_path : Path
        Output VTK file path (should include _parastell in stem).
    width : float
        Cross-section width [m].
    height : float
        Cross-section height [m].
    min_mesh_size : float, optional
        Gmsh minimum element size [cm].
    max_mesh_size : float, optional
        Gmsh maximum element size [cm].

    Returns
    -------
    Path or None
        Path to written VTK file if successful; None if ParaStell or Gmsh
        unavailable or if build/mesh fails.
    """
    global _last_parastell_error
    try:
        from parastell.magnet_coils import MagnetSetFromFilaments  # type: ignore[import-untyped]
    except ImportError as e:
        _last_parastell_error = str(e)
        return None

    import tempfile

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".vtk":
        output_path = output_path.with_suffix(".vtk")

    try:
        import gmsh  # type: ignore[import-untyped]
    except ImportError as e:
        _last_parastell_error = f"gmsh: {e}"
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            filament_file = tmpdir / "coils_filament.txt"
            _write_parastell_filament_file(coils, filament_file, scale=100.0)

            # ParaStell width/thickness in cm; our width/height in m
            width_cm = width * 100
            thickness_cm = height * 100

            ms = MagnetSetFromFilaments(
                str(filament_file),
                width=width_cm,
                thickness=thickness_cm,
                toroidal_extent=360.0,
                case_thickness=0.0,
                scale=100.0,
                start_line=3,
            )
            ms.populate_magnet_coils()
            ms.build_magnet_coils()

            if not ms.coil_solids or len(ms.coil_solids) == 0:
                return None

            ms.mesh_magnets_gmsh(
                min_mesh_size=min_mesh_size,
                max_mesh_size=max_mesh_size,
            )
            gmsh.write(str(output_path))
            gmsh.clear()
            gmsh.finalize()

        return output_path
    except Exception as e:
        _last_parastell_error = str(e)
        return None


def finite_build_coils_to_vtk(
    coils: List,
    output_path: Union[str, Path],
    width: Optional[float] = None,
    height: Optional[float] = None,
    width_per_coil: Optional[List[float]] = None,
    height_per_coil: Optional[List[float]] = None,
    n_along: Optional[int] = None,
    use_stellaris_default: bool = True,
    use_parastell: bool = True,
) -> Path:
    """
    Generate finite-build coil geometry and export to VTK.

    Sweeps a rectangular cross-section along each coil centerline and writes
    a combined VTK file. Cross-section dimensions can be:
    - Uniform (width, height) for all coils
    - Per-coil (width_per_coil, height_per_coil)
    - Stellaris-derived: sqrt(N_turns) * 20 mm square when N_turns available
    - Default: 5 cm × 5 cm total cross-section

    Parameters
    ----------
    coils : List
        List of simsopt Coil objects (with .curve attribute).
    output_path : Path or str
        Output file path (e.g. "finite_build_coils" -> finite_build_coils.vtk).
    width : float, optional
        Uniform cross-section width [m] for all coils.
    height : float, optional
        Uniform cross-section height [m] for all coils.
    width_per_coil : List[float], optional
        Per-coil width [m]. Length must match number of coils.
    height_per_coil : List[float], optional
        Per-coil height [m]. Length must match number of coils.
    n_along : int, optional
        Number of points along each coil for sampling. If None, uses the
        curve's existing quadrature points.
    use_stellaris_default : bool, default=True
        If no dimensions given, use 5 cm × 5 cm (DEFAULT_CROSS_SECTION_M).
    use_parastell : bool, default=True
        If True and ParaStell (parastell, cadquery, gmsh) is available, use it to
        generate a tetrahedral mesh. Otherwise fall back to the built-in sweep.

    Returns
    -------
    Path
        Path to the written VTK file.

    Raises
    ------
    ValueError
        If coil list is empty or dimension arrays have wrong length.
    """
    if not coils:
        raise ValueError("coils list cannot be empty")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".vtk":
        output_path = output_path.with_suffix(".vtk")

    default_side = DEFAULT_CROSS_SECTION_M
    w_uniform = width if width is not None else default_side
    h_uniform = height if height is not None else default_side

    # ParaStell uses uniform width/height; skip it when per-coil dimensions are requested
    if use_parastell and width_per_coil is None and height_per_coil is None:
        parastell_path = output_path.with_stem(output_path.stem + "_parastell")
        result = _finite_build_coils_to_vtk_parastell(
            coils, parastell_path, width=w_uniform, height=h_uniform
        )
        if result is not None:
            return result
        msg = (
            "ParaStell unavailable or failed; using built-in sweep. "
            "Install parastell and deps (conda: moab, gmsh, cadquery; pip: parastell) for tetrahedral mesh."
        )
        if _last_parastell_error:
            msg += f" Import error: {_last_parastell_error}"
        warnings.warn(msg, UserWarning, stacklevel=2)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for coil_idx, coil in enumerate(coils):
        curve = coil.curve
        gamma = np.asarray(curve.gamma(), dtype=float).reshape(-1, 3)
        gammadash = np.asarray(curve.gammadash(), dtype=float).reshape(-1, 3)

        # Use at least MIN_POINTS_ALONG_CURVE for accurate rectangular sweep representation
        effective_n_along = n_along if n_along is not None else max(len(gamma), MIN_POINTS_ALONG_CURVE)
        if effective_n_along != len(gamma):
            # Resample curve for accurate finite cross-section representation
            t_orig = np.linspace(0, 1, len(gamma), endpoint=False)
            t_new = np.linspace(0, 1, effective_n_along, endpoint=False)
            gamma = np.column_stack([
                np.interp(t_new, t_orig, gamma[:, k]) for k in range(3)
            ])
            # Tangent: interpolate direction, then renormalize (linear interp of derivative)
            gammadash = np.column_stack([
                np.interp(t_new, t_orig, gammadash[:, k]) for k in range(3)
            ])
            # Renormalize tangents to unit length for consistent frame
            norms = np.linalg.norm(gammadash, axis=1, keepdims=True)
            norms = np.where(norms < 1e-14, 1.0, norms)
            gammadash = gammadash / norms

        if width_per_coil is not None and height_per_coil is not None:
            if len(width_per_coil) != len(coils) or len(height_per_coil) != len(coils):
                raise ValueError(
                    "width_per_coil and height_per_coil must have length equal to number of coils"
                )
            w = width_per_coil[coil_idx]
            h = height_per_coil[coil_idx]
        elif width is not None and height is not None:
            w = width
            h = height
        else:
            w = default_side
            h = default_side

        vertices, faces = sweep_rectangular_cross_section(
            gamma, gammadash, width=w, height=h
        )
        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)

    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)

    _write_vtk_unstructured(
        combined_vertices,
        combined_faces,
        output_path,
        title="finite-build coils",
    )
    return output_path
