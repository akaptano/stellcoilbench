"""
Unit tests for cli.py helpers and commands.
"""
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import typer

from stellcoilbench.cli import (
    NumpyJSONEncoder,
    _compute_reactor_scale_metrics,
    _detect_github_username,
    _detect_hardware,
    _get_version_info,
    _zip_submission_directory,
    app,
    update_db_cmd,
    generate_submission,
    run_case,
    submit_case,
    REACTOR_REFERENCE,
)
from stellcoilbench.coil_optimization import ARIES_CS_MINOR_RADIUS


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def _install_stub_modules(monkeypatch, metrics=None, surface="input.TestSurface"):
    if metrics is None:
        metrics = {"final_normalized_squared_flux": 0.001}

    def load_case_config(_path):
        return types.SimpleNamespace(surface_params={"surface": surface})

    def evaluate_case(case_cfg, results_dict):
        return metrics

    def optimize_coils(**kwargs):
        coils_out_path = kwargs.get("coils_out_path")
        if coils_out_path:
            Path(coils_out_path).write_text("{}")
        return {"ok": True}

    eval_mod = types.ModuleType("stellcoilbench.evaluate")
    eval_mod.load_case_config = load_case_config
    eval_mod.evaluate_case = evaluate_case

    coil_mod = types.ModuleType("stellcoilbench.coil_optimization")
    coil_mod.optimize_coils = optimize_coils

    monkeypatch.setitem(sys.modules, "stellcoilbench.evaluate", eval_mod)
    monkeypatch.setitem(sys.modules, "stellcoilbench.coil_optimization", coil_mod)


def test_numpy_json_encoder_handles_numpy_types():
    payload = {
        "i": np.int64(3),
        "f": np.float64(1.25),
        "b": np.bool_(True),
        "a": np.array([1, 2, 3]),
    }
    dumped = json.dumps(payload, cls=NumpyJSONEncoder)
    loaded = json.loads(dumped)
    assert loaded["i"] == 3
    assert loaded["f"] == 1.25
    assert loaded["b"] is True
    assert loaded["a"] == [1, 2, 3]


def test_numpy_json_encoder_handles_array_protocol():
    class _ArrayLike:
        def __array__(self):
            return np.array([4, 5, 6])

    payload = {"arr": _ArrayLike()}
    dumped = json.dumps(payload, cls=NumpyJSONEncoder)
    loaded = json.loads(dumped)
    assert loaded["arr"] == [4, 5, 6]


def test_numpy_json_encoder_array_protocol_error():
    class _BadArray:
        def __array__(self):
            raise TypeError("bad array")

    with pytest.raises(TypeError):
        json.dumps({"arr": _BadArray()}, cls=NumpyJSONEncoder)


def test_detect_github_username_from_remote_url(monkeypatch):
    def fake_run(cmd, **kwargs):
        # Check if this is the remote URL command
        if isinstance(cmd, list) and "remote" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="https://github.com/bob/repo.git\n")
        # For other commands, return error
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert _detect_github_username() == "bob"


def test_detect_github_username_from_remote_url_ssh(monkeypatch):
    def fake_run(cmd, **kwargs):
        # Check if this is the remote URL command
        if isinstance(cmd, list) and "remote" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="git@github.com:alice/repo.git\n")
        # For other commands, return error
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert _detect_github_username() == "alice"


def test_detect_github_username_from_env(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setenv("GITHUB_ACTOR", "env_user")
    assert _detect_github_username() == "env_user"


def test_detect_github_username_empty(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    monkeypatch.delenv("GITHUB_USER", raising=False)
    assert _detect_github_username() == ""


def test_zip_submission_directory_creates_zip_and_removes_dir(tmp_path):
    submission_dir = tmp_path / "run"
    submission_dir.mkdir()
    (submission_dir / "results.json").write_text("{}")
    (submission_dir / "nested").mkdir()
    (submission_dir / "nested" / "file.txt").write_text("data")
    (submission_dir / "plot.pdf").write_text("pdf content")

    zip_path = _zip_submission_directory(submission_dir)

    assert zip_path.exists()
    assert zip_path == submission_dir / "all_files.zip"
    # Directory should still exist (for PDFs)
    assert submission_dir.exists()
    # PDF should still be there
    assert (submission_dir / "plot.pdf").exists()
    # Non-PDF files should be removed
    assert not (submission_dir / "results.json").exists()
    assert not (submission_dir / "nested").exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "results.json" in zf.namelist()
        assert "nested/file.txt" in zf.namelist()
        assert "plot.pdf" not in zf.namelist()  # PDFs are not zipped


def test_zip_submission_directory_missing_dir(tmp_path):
    submission_dir = tmp_path / "missing"
    zip_path = _zip_submission_directory(submission_dir)
    assert zip_path == submission_dir / "all_files.zip"


def test_zip_submission_directory_empty_dir(tmp_path):
    submission_dir = tmp_path / "empty"
    submission_dir.mkdir()
    zip_path = _zip_submission_directory(submission_dir)
    assert zip_path == submission_dir / "all_files.zip"
    assert not zip_path.exists()


def test_detect_hardware_reports_cpu_gpu_and_ram(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "sysctl":
            return _FakeCompletedProcess(returncode=0, stdout="Test CPU\n")
        if cmd[0] == "nvidia-smi":
            return _FakeCompletedProcess(returncode=0, stdout="GPU1\nGPU2\n")
        return _FakeCompletedProcess(returncode=1, stdout="")

    class _Mem:
        total = 8 * 1024**3

    class _Psutil:
        @staticmethod
        def virtual_memory():
            return _Mem()

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.processor", lambda: "Fallback CPU")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setitem(sys.modules, "psutil", _Psutil())

    hardware = _detect_hardware()
    assert "CPU: Test CPU" in hardware
    assert "GPU: GPU1, GPU2" in hardware
    assert "RAM: 8.0GB" in hardware


def test_detect_hardware_linux_cpu_model(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "lscpu":
            return _FakeCompletedProcess(
                returncode=0,
                stdout="Model name: Fancy CPU\n",
            )
        if cmd[0] == "nvidia-smi":
            return _FakeCompletedProcess(returncode=1, stdout="")
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.processor", lambda: "")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("subprocess.run", fake_run)

    hardware = _detect_hardware()
    assert "CPU: Fancy CPU" in hardware


def test_run_case_writes_results_json(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch, metrics={"final_normalized_squared_flux": 0.123}, surface="input.TestSurface")
    case_path = tmp_path / "case.yaml"
    case_path.write_text("description: test\nsurface_params:\n  surface: input.TestSurface\n")
    submissions_dir = tmp_path / "submissions"
    monkeypatch.setattr("stellcoilbench.cli._detect_github_username", lambda: "testuser")

    run_case(case_path=case_path, submissions_dir=submissions_dir, results_out=None)

    # Results should be in submissions/TestSurface/testuser/<datetime>/results.json
    results_files = list(submissions_dir.rglob("results.json"))
    assert len(results_files) == 1
    results_path = results_files[0]
    assert "TestSurface" in str(results_path)
    assert "testuser" in str(results_path)
    data = json.loads(results_path.read_text())
    # Results are now wrapped in a structured format with metrics, metadata, version_info
    assert "metrics" in data
    assert "version_info" in data
    assert "reactor_scale_metrics" in data
    assert data["metrics"]["final_normalized_squared_flux"] == 0.123


def test_run_case_ensures_json_extension(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch, surface="input.TestSurface")
    case_path = tmp_path / "case.yaml"
    case_path.write_text("description: test\nsurface_params:\n  surface: input.TestSurface\n")
    submissions_dir = tmp_path / "submissions"
    results_out = tmp_path / "metrics.txt"
    monkeypatch.setattr("stellcoilbench.cli._detect_github_username", lambda: "testuser")

    run_case(case_path=case_path, submissions_dir=submissions_dir, results_out=results_out)

    assert not results_out.exists()
    assert (tmp_path / "metrics.json").exists()


def test_generate_submission_writes_results(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch, metrics={"final_normalized_squared_flux": 0.007})
    monkeypatch.chdir(tmp_path)

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    case_yaml = case_dir / "case.yaml"
    case_yaml.write_text("description: test\n")
    (case_dir / "coils.json").write_text("{}")

    metadata_yaml = tmp_path / "metadata.yaml"
    metadata_yaml.write_text(
        "method_name: demo\n"
        "method_version: v1\n"
        "contact: test@example.com\n"
        "hardware: CPU\n"
        "notes: note\n"
    )

    generate_submission(
        case_path=case_dir,
        metadata_path=metadata_yaml,
        coils_path=None,
        submission_out=None,
    )

    submission_path = tmp_path / "submissions" / "demo" / "v1" / "results.json"
    assert submission_path.exists()
    data = json.loads(submission_path.read_text())
    assert data["metadata"]["method_name"] == "demo"
    assert data["metrics"]["final_normalized_squared_flux"] == 0.007


def test_generate_submission_missing_coils_file(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case.yaml").write_text("description: test\n")
    metadata_yaml = tmp_path / "metadata.yaml"
    metadata_yaml.write_text("method_name: demo\nmethod_version: v1\n")

    with pytest.raises(typer.Exit):
        generate_submission(
            case_path=case_dir,
            metadata_path=metadata_yaml,
            coils_path=None,
            submission_out=None,
        )


def test_submit_case_creates_submission(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch, metrics={"final_normalized_squared_flux": 0.004})
    monkeypatch.chdir(tmp_path)

    case_path = tmp_path / "case.yaml"
    case_path.write_text("description: test\nsurface_params:\n  surface: input.TestSurface\n")

    monkeypatch.setattr("stellcoilbench.cli._detect_github_username", lambda: "user1")
    monkeypatch.setattr("stellcoilbench.cli._detect_hardware", lambda: "CPU: Test")
    monkeypatch.setattr("stellcoilbench.cli._zip_submission_directory", lambda path: path.with_suffix(".zip"))

    submissions_dir = tmp_path / "submissions"
    submit_case(
        case_path=case_path,
        method_name="method1",
        notes="",
        submissions_dir=submissions_dir,
    )

    results_files = list(submissions_dir.rglob("results.json"))
    assert len(results_files) == 1
    results_data = json.loads(results_files[0].read_text())
    assert results_data["metadata"]["method_name"] == "method1"
    assert results_data["metadata"]["contact"] == "user1"

    case_copies = list(submissions_dir.rglob("case.yaml"))
    assert len(case_copies) == 1
    case_data = case_copies[0].read_text()
    assert "source_case_file: case.yaml" in case_data


def test_submit_case_unknown_user_and_hardware(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch, surface="wout.TestSurface")
    monkeypatch.chdir(tmp_path)

    case_path = tmp_path / "case.yaml"
    case_path.write_text("description: test\nsurface_params:\n  surface: wout.TestSurface\n")

    monkeypatch.setattr("stellcoilbench.cli._detect_github_username", lambda: "")
    monkeypatch.setattr("stellcoilbench.cli._detect_hardware", lambda: "")
    monkeypatch.setattr("stellcoilbench.cli._zip_submission_directory", lambda path: path.with_suffix(".zip"))

    submissions_dir = tmp_path / "submissions"
    submit_case(
        case_path=case_path,
        method_name="method1",
        notes="",
        submissions_dir=submissions_dir,
    )

    results_files = list(submissions_dir.rglob("results.json"))
    assert len(results_files) == 1
    # Current structure: submissions/surface/user/timestamp/
    assert "unknown_user" in str(results_files[0].parent)
    assert "TestSurface" in str(results_files[0].parent.parent)


def test_update_db_cmd_invokes_update_database(tmp_path, monkeypatch):
    calls = {}

    def fake_update_database(repo_root, submissions_root, docs_dir):
        calls["repo_root"] = repo_root
        calls["submissions_root"] = submissions_root
        calls["docs_dir"] = docs_dir

    monkeypatch.setattr("stellcoilbench.update_db.update_database", fake_update_database)
    update_db_cmd(submissions_dir=tmp_path / "subs", docs_dir=tmp_path / "docs")
    assert calls["docs_dir"] == tmp_path / "docs"


def test_main_calls_app(monkeypatch):
    import stellcoilbench.cli as cli_module

    called = {"count": 0}

    def fake_app():
        called["count"] += 1

    monkeypatch.setattr(cli_module, "app", fake_app)
    cli_module.main()
    assert called["count"] == 1


# ---------------------------------------------------------------------------
# Tests for _compute_reactor_scale_metrics
# ---------------------------------------------------------------------------

def _make_metrics(minor_radius=0.2, target_B=1.0, major_radius=None, **overrides):
    """Helper to build a metrics dict with cached thresholds and device params.

    Uses minor_radius for L_scale (ARIES_CS_MINOR_RADIUS / minor_radius).
    major_radius defaults to ~4.5 * minor_radius (ARIES-CS aspect ratio) if not set.
    """
    if major_radius is None:
        major_radius = 4.5 * minor_radius  # typical aspect ratio
    a0 = ARIES_CS_MINOR_RADIUS / minor_radius
    m = {
        "target_B_field": target_B,
        "_cached_thresholds": {
            "major_radius": major_radius,
            "minor_radius": minor_radius,
            "a0": a0,
        },
    }
    m.update(overrides)
    return m


def test_reactor_scale_returns_error_when_params_missing():
    result = _compute_reactor_scale_metrics({})
    assert "error" in result


def test_reactor_scale_already_reactor_scaled_skips_scaling():
    """When already_reactor_scaled=True, raw metrics are used as reactor-scale (L_scale=1, B_scale=1)."""
    metrics = _make_metrics(
        minor_radius=0.2, target_B=1.0,
        final_min_cc_separation=0.5,
        final_min_cs_separation=1.5,
        final_total_length=200.0,
        final_max_curvature=0.5,
    )
    result = _compute_reactor_scale_metrics(metrics, already_reactor_scaled=True)
    assert result["scaling_factors"]["length_scale"] == 1.0
    assert result["scaling_factors"]["B_field_scale"] == 1.0
    assert result["scaling_factors"].get("already_reactor_scaled") is True
    assert result["reactor_scale_min_cc_separation"] == 0.5
    assert result["reactor_scale_min_cs_separation"] == 1.5
    assert result["reactor_scale_total_length"] == 200.0
    assert result["reactor_scale_max_curvature"] == 0.5


def test_reactor_scale_length_scaling():
    """Lengths [m] should scale as L_scale = ARIES_CS_MINOR_RADIUS / minor_radius."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_min_cc_separation=0.1,
        final_min_cs_separation=0.2,
        final_total_length=5.0,
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert result["reactor_scale_min_cc_separation"] == pytest.approx(0.1 * L_scale)
    assert result["reactor_scale_min_cs_separation"] == pytest.approx(0.2 * L_scale)
    assert result["reactor_scale_total_length"] == pytest.approx(5.0 * L_scale)


def test_reactor_scale_curvature_scaling():
    """Curvature [1/m] scales as 1/L, MSC [1/m²] scales as 1/L²."""
    minor_radius = 0.34  # L_scale = 1.7/0.34 = 5.0
    target_B = 2.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_curvature=3.0,
        final_average_curvature=1.5,
        final_mean_squared_curvature=4.0,
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert result["reactor_scale_max_curvature"] == pytest.approx(3.0 / L_scale)
    assert result["reactor_scale_average_curvature"] == pytest.approx(1.5 / L_scale)
    assert result["reactor_scale_mean_squared_curvature"] == pytest.approx(4.0 / L_scale**2)


def test_reactor_scale_force_scaling():
    """Force/length [MN/m] scales as B²·L / 1e6 (dF/dℓ = I×B, I ∝ B·L)."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale = B_scale**2 * L_scale / 1e6

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_max_coil_force=1e5,
        final_avg_max_coil_force=5e4,
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert result["reactor_scale_max_max_coil_force"] == pytest.approx(1e5 * force_scale)
    assert result["reactor_scale_avg_max_coil_force"] == pytest.approx(5e4 * force_scale)


def test_reactor_scale_torque_scaling():
    """Torque/length [MN] scales as B²·L² / 1e6 (dτ/dℓ = r × dF/dℓ, r ∝ L)."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    torque_scale = B_scale**2 * L_scale**2 / 1e6

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_max_coil_torque=1e6,
        final_avg_max_coil_torque=5e5,
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert result["reactor_scale_max_max_coil_torque"] == pytest.approx(1e6 * torque_scale)
    assert result["reactor_scale_avg_max_coil_torque"] == pytest.approx(5e5 * torque_scale)


def test_reactor_scale_n_turns_per_coil():
    """N_turns_per_coil[i] = max(N_force_i, N_jc_i) per unique coil.

    N_force_i = ceil(reactor_force[i] / 0.5)
    N_jc_i comes from the REBCO Jc model (Stellaris parameters).
    """
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale  # N/m → N/m at reactor scale

    # Three unique coils with different device-scale max forces [N/m].
    # Choose values so that reactor-scale forces [MN/m] are 2.3, 0.4, 0.7
    desired_MN = [2.3, 0.4, 0.7]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN]

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
    )
    result = _compute_reactor_scale_metrics(metrics)

    # Check per-coil reactor forces
    for i, d in enumerate(desired_MN):
        assert result["reactor_scale_force_per_coil_MN_per_m"][i] == pytest.approx(d)

    # Force-based N_turns: ceil(2.3/0.5)=5, ceil(0.4/0.5)=1, ceil(0.7/0.5)=2
    assert result["N_turns_force"] == [5, 1, 2]
    # Jc-based N_turns also present
    assert "N_turns_jc" in result
    assert len(result["N_turns_jc"]) == 3
    # Final N_turns is element-wise max of force and Jc
    for i in range(3):
        assert result["N_turns_per_coil"][i] == max(
            result["N_turns_force"][i], result["N_turns_jc"][i]
        )
    assert result["force_limit_MN_per_m"] == 0.5
    # Jc model details are stored
    assert "jc_model" in result
    assert "NI_reactor" in result["jc_model"]
    assert "I_turn" in result["jc_model"]
    assert "B_peak_estimate" in result["jc_model"]


def test_reactor_scale_n_turns_minimum_one():
    """N_turns_per_coil is at least 1 even when force is below the limit."""
    metrics = _make_metrics(
        minor_radius=0.2, target_B=1.0,
        final_max_max_coil_force=1e-3,
        final_max_force_per_coil=[1e-3, 1e-4],
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert result["N_turns_per_coil"] == [1, 1]


def test_reactor_scale_n_turns_from_currents_only():
    """N_turns finite-build runs when only currents/lengths present (e.g. Zenodo coils).

    When final_max_force_per_coil is missing, Jc model uses final_current_per_coil
    to compute N_turns_jc; n_turns_force=1; total_superconductor_length_km is computed.
    """
    metrics = _make_metrics(
        minor_radius=0.2, target_B=1.0,
        final_total_length=40.0,
        final_current_per_coil=[1e5, 1e5, 1e5, 1e5],  # 100 kA per coil
        final_length_per_coil=[10.0, 10.0, 10.0, 10.0],
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert "N_turns_per_coil" in result
    assert "N_turns_jc" in result
    assert "total_superconductor_length_km" in result
    assert result["N_turns_force"] == [1, 1, 1, 1]
    # Jc-based N_turns should be > 1 for 100 kA coils at reactor scale
    assert all(n >= 1 for n in result["N_turns_per_coil"])


# ---------------------------------------------------------------------------
# Tests for per-turn force and torque
# ---------------------------------------------------------------------------

def test_per_turn_max_force():
    """per_turn_max_force = max_i(reactor_force_i / N_turns_i)."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    desired_MN = [2.5, 0.3]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN]

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
    )
    result = _compute_reactor_scale_metrics(metrics)

    n_turns = result["N_turns_per_coil"]
    rs_forces = result["reactor_scale_force_per_coil_MN_per_m"]
    expected_per_turn = [f / n for f, n in zip(rs_forces, n_turns)]
    assert result["per_turn_max_force"] == pytest.approx(max(expected_per_turn))


def test_per_turn_max_torque_with_per_coil():
    """per_turn_max_torque uses per-coil torque when available."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale
    torque_scale_raw = B_scale**2 * L_scale**2

    desired_MN_force = [1.0, 1.0]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN_force]
    # Two coils with different device-scale torques
    device_torques = [1e4, 2e4]

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
        final_max_max_coil_torque=max(device_torques),
        final_max_torque_per_coil=device_torques,
    )
    result = _compute_reactor_scale_metrics(metrics)

    n_turns = result["N_turns_per_coil"]
    reactor_torques = [t * torque_scale_raw / 1e6 for t in device_torques]
    expected_per_turn = [t / n for t, n in zip(reactor_torques, n_turns)]
    assert result["per_turn_max_torque"] == pytest.approx(max(expected_per_turn))


def test_per_turn_max_torque_fallback():
    """Without per-coil torque, falls back to max_torque / min(N_turns)."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    desired_MN_force = [1.0, 2.0]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN_force]

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
        final_max_max_coil_torque=5e4,  # overall max, no per-coil list
    )
    result = _compute_reactor_scale_metrics(metrics)

    max_tau_rs = result["reactor_scale_max_max_coil_torque"]
    min_n = min(result["N_turns_per_coil"])
    assert result["per_turn_max_torque"] == pytest.approx(max_tau_rs / min_n)


def test_total_superconductor_length():
    """Total SC length = Σ N_turns_i * reactor_scale_length_i, in km."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    # Two coils with different device-scale forces
    desired_MN = [2.3, 0.4]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN]
    device_lengths = [3.0, 4.0]

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
        final_length_per_coil=device_lengths,
        final_total_length=sum(device_lengths),
    )
    result = _compute_reactor_scale_metrics(metrics)

    # N_turns = max(force, Jc) — verify the total SC formula
    n_turns = result["N_turns_per_coil"]
    expected_km = sum(n * ln * L_scale for n, ln in zip(n_turns, device_lengths)) / 1e3
    assert result["total_superconductor_length_km"] == pytest.approx(expected_km)


def test_total_superconductor_length_fallback():
    """Without per-coil lengths, use uniform-length fallback."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius

    # Two coils with device total length 10.0 m → avg 5.0 m each
    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_max_coil_force=1e-3,
        final_max_force_per_coil=[1e-3, 1e-3],
        final_total_length=10.0,
    )
    result = _compute_reactor_scale_metrics(metrics)
    n_turns = result["N_turns_per_coil"]
    # Fallback: avg_len = 10.0 * L_scale / 2 = 5.0 * L_scale
    expected_km = sum(n * 5.0 * L_scale for n in n_turns) / 1e3
    assert result["total_superconductor_length_km"] == pytest.approx(expected_km)


def test_reactor_scale_squared_flux_scaling():
    """SquaredFlux [T²m²] = ½∫(B·n̂)²dS scales as B²·L² (NOT B²·L⁴)."""
    minor_radius = 0.453  # L_scale = 1.7/0.453 ≈ 3.75
    target_B = 3.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    flux_scale = B_scale**2 * L_scale**2

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_squared_flux=1e-4,
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert result["reactor_scale_squared_flux"] == pytest.approx(1e-4 * flux_scale)


def test_reactor_scale_arclength_variation_scaling():
    """ArclengthVariation [m²] (variance of arclengths) scales as L²."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_arclength_variation=0.01,
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert result["reactor_scale_arclength_variation"] == pytest.approx(0.01 * L_scale**2)


# ---------------------------------------------------------------------------
# Tests for REBCO Jc model and Jc-based N_turns
# ---------------------------------------------------------------------------

def test_rebco_jc_20K_values():
    """Validate the REBCO Jc model at 20 K against Stellaris Table 8 data."""
    from stellcoilbench.cli import _rebco_jc_tape_stack

    # At B=20T: j_crit ≈ 2450 A/mm² = 2.45e9 A/m²
    jc_20 = _rebco_jc_tape_stack(20.0, T_op=20.0)
    assert 2.3e9 < jc_20 < 2.6e9

    # At B=25T: j_crit ≈ 2200 A/mm² = 2.2e9 A/m²
    jc_25 = _rebco_jc_tape_stack(25.0, T_op=20.0)
    assert 2.0e9 < jc_25 < 2.4e9

    # j_crit should decrease monotonically with B
    assert jc_20 > jc_25

    # At B=0: j_crit ≈ 5000 A/mm²
    jc_0 = _rebco_jc_tape_stack(0.0, T_op=20.0)
    assert abs(jc_0 - 5.0e9) < 1e6  # ≈ 5000 A/mm²


def test_rebco_jc_temperature_scaling():
    """Jc should decrease with higher temperature."""
    from stellcoilbench.cli import _rebco_jc_tape_stack

    jc_20K = _rebco_jc_tape_stack(15.0, T_op=20.0)
    jc_40K = _rebco_jc_tape_stack(15.0, T_op=40.0)
    assert jc_20K > jc_40K > 0
    # Linear scaling: jc(40K)/jc(20K) ≈ (1-40/92)/(1-20/92) ≈ 0.565/0.783 ≈ 0.72
    ratio = jc_40K / jc_20K
    assert 0.65 < ratio < 0.80


def test_rebco_jc_negative_B():
    """Negative B should be treated as B=0."""
    from stellcoilbench.cli import _rebco_jc_tape_stack

    jc_neg = _rebco_jc_tape_stack(-5.0, T_op=20.0)
    jc_zero = _rebco_jc_tape_stack(0.0, T_op=20.0)
    assert jc_neg == jc_zero


def test_n_turns_jc_with_currents():
    """N_turns_jc should increase for coils requiring more ampere-turns."""
    from stellcoilbench.cli import _compute_N_turns_critical_current

    minor_radius = 0.2
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    # Two coils: one with 10× more current → needs ~10× more turns
    result = _compute_N_turns_critical_current(
        per_coil_forces=[1e5, 1e5],        # Same forces
        per_coil_currents=[1e4, 1e5],       # 10x current difference
        per_coil_lengths=[5.0, 5.0],
        L_scale=L_scale, B_scale=5.7, target_B=1.0,
    )
    n = result["N_turns_jc"]
    assert n[1] >= 5 * n[0]  # Much more turns for higher current
    assert all(isinstance(x, int) and x >= 1 for x in n)
    # I_turn should be lead-limited or Jc-limited
    assert all(0 < it <= 50e3 for it in result["I_turn"])


def test_n_turns_jc_no_currents_fallback():
    """Without per-coil currents, use force-based current estimate."""
    from stellcoilbench.cli import _compute_N_turns_critical_current

    minor_radius = 0.2
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    result = _compute_N_turns_critical_current(
        per_coil_forces=[1e5, 5e5],
        per_coil_currents=None,
        per_coil_lengths=None,
        L_scale=L_scale, B_scale=5.7, target_B=1.0,
    )
    n = result["N_turns_jc"]
    # Coil with 5× more force estimated as 5× more current → ~5× more turns
    assert n[1] >= 3 * n[0]
    assert all(isinstance(x, int) and x >= 1 for x in n)


def test_n_turns_max_of_force_and_jc():
    """N_turns_per_coil should be element-wise max(N_force, N_jc)."""
    minor_radius = 0.2
    target_B = 1.0

    # Use a very high force to ensure N_force dominates for one coil
    # and moderate force for another where N_jc might dominate.
    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_max_coil_force=1e7,
        final_max_force_per_coil=[1e7, 1e3],  # coil 0: very high force
    )
    result = _compute_reactor_scale_metrics(metrics)
    for i in range(2):
        assert result["N_turns_per_coil"][i] >= result["N_turns_force"][i]
        assert result["N_turns_per_coil"][i] >= result["N_turns_jc"][i]
        assert result["N_turns_per_coil"][i] == max(
            result["N_turns_force"][i], result["N_turns_jc"][i]
        )


# ---------------------------------------------------------------------------
# Tests for finite-build (winding-pack) extent
# ---------------------------------------------------------------------------

def test_winding_pack_width_formula():
    """Winding-pack side length: w = sqrt(N_turns) * 20 mm."""
    from stellcoilbench.cli import STELLARIS_A_TURN
    import numpy as np

    turn_side = np.sqrt(STELLARIS_A_TURN)   # 0.020 m
    assert turn_side == pytest.approx(0.020)

    # Stellaris Table 8 validation:
    # Coil 0: N=324 → w = 18 × 20 mm = 360 mm = 0.360 m
    assert np.sqrt(324) * turn_side == pytest.approx(0.360)
    # Coil 5: N=225 → w = 15 × 20 mm = 300 mm = 0.300 m
    assert np.sqrt(225) * turn_side == pytest.approx(0.300)


def test_winding_pack_in_reactor_metrics():
    """_compute_reactor_scale_metrics stores winding-pack width per coil."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    # Desired reactor-scale forces in MN/m
    desired_MN = [2.5, 0.3]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN]

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
    )
    result = _compute_reactor_scale_metrics(metrics)

    n_turns = result["N_turns_per_coil"]
    wp_widths = result["winding_pack_width_per_coil"]
    assert len(wp_widths) == len(n_turns)

    import numpy as np
    from stellcoilbench.cli import STELLARIS_A_TURN
    turn_side = np.sqrt(STELLARIS_A_TURN)

    for i, (n, w) in enumerate(zip(n_turns, wp_widths)):
        expected = float(np.sqrt(n) * turn_side)
        assert w == pytest.approx(expected), f"Coil {i}: expected {expected}, got {w}"

    # max_winding_pack_width == max of per-coil widths
    assert result["max_winding_pack_width"] == pytest.approx(max(wp_widths))


def test_winding_pack_single_turn():
    """A coil with N_turns=1 has winding-pack width = 20 mm."""
    metrics = _make_metrics(
        minor_radius=0.2, target_B=1.0,
        final_max_max_coil_force=1e-3,
        final_max_force_per_coil=[1e-3],
    )
    result = _compute_reactor_scale_metrics(metrics)
    # N_turns should be 1 (force below limit and Jc gives at least 1)
    assert all(n >= 1 for n in result["N_turns_per_coil"])
    # Winding pack width for N=1 → sqrt(1)*0.020 = 0.020 m
    import numpy as np
    from stellcoilbench.cli import STELLARIS_A_TURN
    n = result["N_turns_per_coil"][0]
    expected_w = float(np.sqrt(n) * np.sqrt(STELLARIS_A_TURN))
    assert result["winding_pack_width_per_coil"][0] == pytest.approx(expected_w)
    assert result["max_winding_pack_width"] == pytest.approx(expected_w)


# ---------------------------------------------------------------------------
# Tests for finite-build coil-coil clearance
# ---------------------------------------------------------------------------

def test_finite_build_cc_clearance_positive():
    """Clearance = d_cc_min - w_max; positive when coils don't overlap."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius

    # Device-scale cc separation of 0.2 m → reactor-scale = 0.2 * L_scale
    # Forces are low so N_turns will be small → w_max small → clearance positive
    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_min_cc_separation=0.2,
        final_max_max_coil_force=1e-3,
        final_max_force_per_coil=[1e-3, 1e-3],
    )
    result = _compute_reactor_scale_metrics(metrics)

    d_cc = result["reactor_scale_min_cc_separation"]
    w_max = result["max_winding_pack_width"]
    assert d_cc == pytest.approx(0.2 * L_scale)
    assert w_max < d_cc  # no overlap
    assert result["finite_build_cc_clearance"] == pytest.approx(d_cc - w_max)
    assert result["finite_build_cc_clearance"] > 0


def test_finite_build_cc_clearance_negative():
    """Clearance is negative when the winding pack exceeds the gap."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    # Very small device-scale separation → small reactor-scale d_cc
    # Very large forces → many turns → large winding pack
    small_gap = 0.001  # 1 mm device scale
    big_force = 50.0 * 1e6 / force_scale_raw  # 50 MN/m reactor → many turns

    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B,
        final_min_cc_separation=small_gap,
        final_max_max_coil_force=big_force,
        final_max_force_per_coil=[big_force, big_force],
    )
    result = _compute_reactor_scale_metrics(metrics)

    d_cc = result["reactor_scale_min_cc_separation"]
    w_max = result["max_winding_pack_width"]
    assert w_max > d_cc  # overlap!
    assert result["finite_build_cc_clearance"] < 0


def test_finite_build_cc_clearance_absent_without_cc_sep():
    """If min_cc_separation is not in metrics, no clearance is computed."""
    metrics = _make_metrics(
        minor_radius=0.2, target_B=1.0,
        final_max_max_coil_force=1e-3,
        final_max_force_per_coil=[1e-3],
    )
    # No final_min_cc_separation in metrics
    result = _compute_reactor_scale_metrics(metrics)
    assert "finite_build_cc_clearance" not in result


# ---------------------------------------------------------------------------
# Tests for _get_version_info
# ---------------------------------------------------------------------------

def test_get_version_info_git_exception(monkeypatch):
    """_get_version_info handles git command failure gracefully."""
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr("subprocess.run", fake_run)
    info = _get_version_info()
    assert info["stellcoilbench_commit"] == "unknown"


def test_get_version_info_simsopt_not_installed(monkeypatch):
    """_get_version_info handles missing simsopt."""
    # Make git calls succeed first
    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=0, stdout="abc123\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    # Force simsopt import to fail
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "simsopt":
            raise ImportError("no simsopt")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    info = _get_version_info()
    assert info["simsopt_version"] == "not installed"


def test_get_version_info_simsopt_no_file(monkeypatch):
    """_get_version_info returns early when simsopt has no __file__."""
    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=0, stdout="abc123\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    # Create a fake simsopt module with no __file__
    fake_simsopt = types.ModuleType("simsopt")
    fake_simsopt.__version__ = "1.0.0"
    # Ensure __file__ is None
    fake_simsopt.__file__ = None
    monkeypatch.setitem(sys.modules, "simsopt", fake_simsopt)

    info = _get_version_info()
    assert info["simsopt_version"] == "1.0.0"
    # Should not have simsopt_commit since __file__ was None
    assert "simsopt_branch" not in info


def test_get_version_info_simsopt_editable_install(monkeypatch, tmp_path):
    """_get_version_info detects simsopt installed from source with .git dir."""
    # Set up a fake directory structure with .git
    simsopt_pkg = tmp_path / "simsopt"
    simsopt_pkg.mkdir()
    (tmp_path / ".git").mkdir()  # parent has .git
    init_file = simsopt_pkg / "__init__.py"
    init_file.write_text("")

    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        if "rev-parse" in cmd and "HEAD" in cmd and "--abbrev-ref" not in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="deadbeef\n")
        elif "--abbrev-ref" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="main\n")
        elif "remote" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="https://github.com/user/simsopt.git\n")
        return _FakeCompletedProcess(returncode=0, stdout="abc\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    fake_simsopt = types.ModuleType("simsopt")
    fake_simsopt.__version__ = "0.1.dev100+gabcdef"
    fake_simsopt.__file__ = str(init_file)
    monkeypatch.setitem(sys.modules, "simsopt", fake_simsopt)

    info = _get_version_info()
    assert info["simsopt_version"] == "0.1.dev100+gabcdef"
    assert info["simsopt_commit"] == "deadbeef"
    assert info["simsopt_branch"] == "main"
    assert info["simsopt_remote"] == "https://github.com/user/simsopt.git"


def test_get_version_info_simsopt_git_exception(monkeypatch, tmp_path):
    """_get_version_info handles git failure inside simsopt source dir."""
    simsopt_pkg = tmp_path / "simsopt"
    simsopt_pkg.mkdir()
    (tmp_path / ".git").mkdir()
    init_file = simsopt_pkg / "__init__.py"
    init_file.write_text("")

    call_idx = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_idx["n"] += 1
        # First two calls are for stellcoilbench git (HEAD, branch)
        if call_idx["n"] <= 2:
            return _FakeCompletedProcess(returncode=0, stdout="abc\n")
        # Subsequent calls (simsopt git) should fail
        raise OSError("git failed")

    monkeypatch.setattr("subprocess.run", fake_run)

    fake_simsopt = types.ModuleType("simsopt")
    fake_simsopt.__version__ = "1.0.0"
    fake_simsopt.__file__ = str(init_file)
    monkeypatch.setitem(sys.modules, "simsopt", fake_simsopt)

    info = _get_version_info()
    assert info["simsopt_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Tests for NumpyJSONEncoder - simsopt object handling
# ---------------------------------------------------------------------------

def test_numpy_json_encoder_simsopt_object():
    """NumpyJSONEncoder converts simsopt objects to string."""
    class _FakeSimsoptObj:
        __module__ = "simsopt.geo.surface"
        def __str__(self):
            return "FakeSurface()"

    payload = {"obj": _FakeSimsoptObj()}
    dumped = json.dumps(payload, cls=NumpyJSONEncoder)
    loaded = json.loads(dumped)
    assert loaded["obj"] == "FakeSurface()"


def test_numpy_json_encoder_unknown_type_raises():
    """NumpyJSONEncoder raises TypeError for unrecognized types."""
    class _Unknown:
        pass

    with pytest.raises(TypeError):
        json.dumps({"obj": _Unknown()}, cls=NumpyJSONEncoder)


# ---------------------------------------------------------------------------
# Tests for _detect_hardware - exception paths
# ---------------------------------------------------------------------------

def test_detect_hardware_cpu_exception(monkeypatch):
    """_detect_hardware handles CPU detection exception."""
    monkeypatch.setattr("platform.processor", lambda: "")
    monkeypatch.setattr("platform.machine", lambda: "")

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("command not found")

    monkeypatch.setattr("subprocess.run", fake_run)
    # Remove psutil if present
    monkeypatch.delitem(sys.modules, "psutil", raising=False)

    hardware = _detect_hardware()
    # Should still return something (platform.platform() fallback or partial)
    assert isinstance(hardware, str)


def test_detect_hardware_gpu_exception(monkeypatch):
    """_detect_hardware handles nvidia-smi exception."""
    monkeypatch.setattr("platform.processor", lambda: "TestCPU")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("platform.system", lambda: "Linux")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "nvidia-smi":
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2)
        if cmd[0] == "lscpu":
            return _FakeCompletedProcess(returncode=1, stdout="")
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)

    hardware = _detect_hardware()
    assert "CPU: TestCPU" in hardware
    assert "GPU" not in hardware


def test_detect_hardware_psutil_import_error(monkeypatch):
    """_detect_hardware handles missing psutil."""
    monkeypatch.setattr("platform.processor", lambda: "TestCPU")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("platform.system", lambda: "Other")

    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)

    # Ensure psutil import fails
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    hardware = _detect_hardware()
    assert "CPU: TestCPU" in hardware
    assert "RAM" not in hardware


def test_detect_hardware_detailed_cpu_exception(monkeypatch):
    """_detect_hardware handles detailed CPU command timeout."""
    monkeypatch.setattr("platform.processor", lambda: "TestCPU")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "sysctl":
            raise subprocess.TimeoutExpired(cmd="sysctl", timeout=2)
        if cmd[0] == "nvidia-smi":
            return _FakeCompletedProcess(returncode=1, stdout="")
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)

    hardware = _detect_hardware()
    assert "CPU: TestCPU" in hardware


# ---------------------------------------------------------------------------
# Tests for post_process command
# ---------------------------------------------------------------------------

def test_post_process_success(tmp_path, monkeypatch):
    """post_process calls run_post_processing and prints results."""
    coils_file = tmp_path / "coils.json"
    coils_file.write_text("{}")
    output_dir = tmp_path / "output"

    fake_results = {
        "BdotN": 1.23e-4,
        "BdotN_over_B": 5.67e-5,
        "quasisymmetry_average": 0.01,
    }

    pp_mod = types.ModuleType("stellcoilbench.post_processing")
    pp_mod.run_post_processing = lambda **kwargs: fake_results
    monkeypatch.setitem(sys.modules, "stellcoilbench.post_processing", pp_mod)

    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(app, [
        "post-process",
        str(coils_file),
        "--output-dir", str(output_dir),
    ])
    assert result.exit_code == 0
    assert "Post-processing complete!" in result.output
    assert "B·n on plasma surface" in result.output
    assert "quasisymmetry" in result.output


def test_post_process_no_qs(tmp_path, monkeypatch):
    """post_process works when quasisymmetry key is absent."""
    coils_file = tmp_path / "coils.json"
    coils_file.write_text("{}")
    output_dir = tmp_path / "output"

    fake_results = {"some_metric": 42}

    pp_mod = types.ModuleType("stellcoilbench.post_processing")
    pp_mod.run_post_processing = lambda **kwargs: fake_results
    monkeypatch.setitem(sys.modules, "stellcoilbench.post_processing", pp_mod)

    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(app, [
        "post-process",
        str(coils_file),
        "--output-dir", str(output_dir),
    ])
    assert result.exit_code == 0
    assert "Post-processing complete!" in result.output
    assert "quasisymmetry" not in result.output


def test_post_process_exception(tmp_path, monkeypatch):
    """post_process handles exceptions from run_post_processing."""
    coils_file = tmp_path / "coils.json"
    coils_file.write_text("{}")
    output_dir = tmp_path / "output"

    def boom(**kwargs):
        raise RuntimeError("post-processing failed")

    pp_mod = types.ModuleType("stellcoilbench.post_processing")
    pp_mod.run_post_processing = boom
    monkeypatch.setitem(sys.modules, "stellcoilbench.post_processing", pp_mod)

    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(app, [
        "post-process",
        str(coils_file),
        "--output-dir", str(output_dir),
    ])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Tests for surface name parsing edge cases (submit_case / run_case)
# ---------------------------------------------------------------------------

def test_run_case_wout_surface_prefix(tmp_path, monkeypatch):
    """run_case strips 'wout.' prefix from surface name for directory."""
    _install_stub_modules(monkeypatch, metrics={"final_normalized_squared_flux": 0.1}, surface="wout.TestSurface")
    case_path = tmp_path / "case.yaml"
    case_path.write_text("description: test\nsurface_params:\n  surface: wout.TestSurface\n")
    submissions_dir = tmp_path / "submissions"
    monkeypatch.setattr("stellcoilbench.cli._detect_github_username", lambda: "testuser")

    run_case(case_path=case_path, submissions_dir=submissions_dir, results_out=None)

    results_files = list(submissions_dir.rglob("results.json"))
    assert len(results_files) == 1
    # Should use "TestSurface" not "wout.TestSurface"
    assert "TestSurface" in str(results_files[0])
    assert "wout." not in str(results_files[0])


def test_run_case_surface_with_extension(tmp_path, monkeypatch):
    """run_case strips file extensions like .focus from surface name."""
    _install_stub_modules(monkeypatch, metrics={"final_normalized_squared_flux": 0.05},
                          surface="c09r00_NCSX.focus")
    case_path = tmp_path / "case.yaml"
    case_path.write_text("description: test\nsurface_params:\n  surface: c09r00_NCSX.focus\n")
    submissions_dir = tmp_path / "submissions"
    monkeypatch.setattr("stellcoilbench.cli._detect_github_username", lambda: "testuser")

    run_case(case_path=case_path, submissions_dir=submissions_dir, results_out=None)

    results_files = list(submissions_dir.rglob("results.json"))
    assert len(results_files) == 1
    # Should use "c09r00_NCSX" not "c09r00_NCSX.focus"
    assert "c09r00_NCSX" in str(results_files[0])
    assert ".focus" not in str(results_files[0])


def test_submit_case_surface_with_extension(tmp_path, monkeypatch):
    """submit_case strips file extensions from surface name in directory path."""
    _install_stub_modules(monkeypatch, metrics={"final_normalized_squared_flux": 0.004},
                          surface="plasma.focus")
    monkeypatch.chdir(tmp_path)

    case_path = tmp_path / "case.yaml"
    case_path.write_text("description: test\nsurface_params:\n  surface: plasma.focus\n")

    monkeypatch.setattr("stellcoilbench.cli._detect_github_username", lambda: "user1")
    monkeypatch.setattr("stellcoilbench.cli._detect_hardware", lambda: "CPU: Test")
    monkeypatch.setattr("stellcoilbench.cli._zip_submission_directory", lambda path: path.with_suffix(".zip"))

    submissions_dir = tmp_path / "submissions"
    submit_case(
        case_path=case_path,
        method_name="method1",
        notes="",
        submissions_dir=submissions_dir,
    )

    results_files = list(submissions_dir.rglob("results.json"))
    assert len(results_files) == 1
    # Directory should use "plasma" not "plasma.focus"
    assert "/plasma/" in str(results_files[0])


def test_submit_case_relative_path_fallback(tmp_path, monkeypatch):
    """submit_case handles ValueError when computing relative path for case.yaml."""
    _install_stub_modules(monkeypatch, metrics={"final_normalized_squared_flux": 0.004},
                          surface="input.Test")
    # Set cwd to a different root so relative_to raises ValueError
    other_root = tmp_path / "other_root"
    other_root.mkdir()
    monkeypatch.chdir(other_root)

    case_path = tmp_path / "case.yaml"
    case_path.write_text("description: test\nsurface_params:\n  surface: input.Test\n")

    monkeypatch.setattr("stellcoilbench.cli._detect_github_username", lambda: "user1")
    monkeypatch.setattr("stellcoilbench.cli._detect_hardware", lambda: "CPU: Test")
    monkeypatch.setattr("stellcoilbench.cli._zip_submission_directory", lambda path: path.with_suffix(".zip"))

    submissions_dir = tmp_path / "submissions"
    submit_case(
        case_path=case_path,
        method_name="method1",
        notes="",
        submissions_dir=submissions_dir,
    )

    results_files = list(submissions_dir.rglob("results.json"))
    assert len(results_files) == 1
