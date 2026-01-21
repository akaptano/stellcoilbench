"""
Unit tests for cli.py helpers and commands.
"""
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import typer

from stellcoilbench.cli import (
    NumpyJSONEncoder,
    _detect_github_username,
    _detect_hardware,
    _zip_submission_directory,
    update_db_cmd,
    generate_submission,
    run_case,
    submit_case,
)


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


def test_detect_github_username_from_git_config(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=0, stdout="alice\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert _detect_github_username() == "alice"


def test_detect_github_username_from_remote_url(monkeypatch):
    calls = {"count": 0}

    def fake_run(cmd, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeCompletedProcess(returncode=1, stdout="")
        return _FakeCompletedProcess(returncode=0, stdout="https://github.com/bob/repo.git\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert _detect_github_username() == "bob"


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

    zip_path = _zip_submission_directory(submission_dir)

    assert zip_path.exists()
    assert not submission_dir.exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "results.json" in zf.namelist()
        assert "nested/file.txt" in zf.namelist()


def test_zip_submission_directory_missing_dir(tmp_path):
    submission_dir = tmp_path / "missing"
    zip_path = _zip_submission_directory(submission_dir)
    assert zip_path == submission_dir.parent / "missing.zip"


def test_zip_submission_directory_empty_dir(tmp_path):
    submission_dir = tmp_path / "empty"
    submission_dir.mkdir()
    zip_path = _zip_submission_directory(submission_dir)
    assert zip_path == submission_dir.parent / "empty.zip"
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
    _install_stub_modules(monkeypatch, metrics={"final_normalized_squared_flux": 0.123})
    case_path = tmp_path / "case.yaml"
    case_path.write_text("description: test\n")
    coils_out_dir = tmp_path / "out"

    run_case(case_path=case_path, coils_out_dir=coils_out_dir, results_out=None)

    results_path = coils_out_dir / "results.json"
    assert results_path.exists()
    data = json.loads(results_path.read_text())
    assert data["final_normalized_squared_flux"] == 0.123


def test_run_case_ensures_json_extension(tmp_path, monkeypatch):
    _install_stub_modules(monkeypatch)
    case_path = tmp_path / "case.yaml"
    case_path.write_text("description: test\n")
    coils_out_dir = tmp_path / "out"
    results_out = tmp_path / "metrics.txt"

    run_case(case_path=case_path, coils_out_dir=coils_out_dir, results_out=results_out)

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
    # wout. prefix should be stripped from surface name
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
