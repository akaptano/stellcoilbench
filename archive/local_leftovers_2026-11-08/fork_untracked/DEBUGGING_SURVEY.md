# StellCoilBench Debugging Survey

**Date:** 2026-03-08  
**Scope:** Comprehensive code quality and potential bug survey across `src/`, `tools/`, `knowledge/`, CI workflows, and tests.

---

## Executive Summary

- **Tests:** 670 passed, 3 skipped. All passing.
- **Lint:** Ruff checks pass.
- **Findings:** Several lower-severity issues and robustness improvements identified; no critical bugs requiring immediate fix.

---

## 1. Overly Broad Exception Handling

| File | Line(s) | Issue |
|------|---------|-------|
| `tools/ci_filter_cases.py` | 38, 52, 83 | `except Exception` swallows all exceptions (including `KeyboardInterrupt`, `SystemExit`). Can mask real bugs. |
| `tools/ci_autopilot_runner.py` | 36, 49 | Same pattern. |
| `src/stellcoilbench/cli/list_cases_cmd.py` | 49, 65 | `except Exception` when loading YAML—masks parse/IO errors. |
| `src/stellcoilbench/post_processing/_simple_run.py` | 50 | `except Exception` for `sys.prefix` / env setup—defensible (best-effort) but could hide `ImportError`. |
| `src/stellcoilbench/finite_build/_parastell.py` | 181, 191, 298, 304 | Broad `except Exception` in optional ParaStell code. |
| `src/stellcoilbench/coil_optimization/_structural_mesh.py` | 148, 181, 219, 328, 366 | Same in mesh/cad logic. |
| `src/stellcoilbench/coil_optimization/_scipy_optimizer.py` | 579 | Taylor test failure path. |

**Recommendation:** Replace `except Exception` with more specific types (e.g. `OSError`, `ValueError`, `yaml.YAMLError`) where possible to avoid masking unexpected failures.

---

## 2. Pytest Mark Warnings

```
tests/test_structural_objective.py:424  PytestUnknownMarkWarning: Unknown pytest.mark.benchmark
tests/test_structural_objective.py:616  PytestUnknownMarkWarning: Unknown pytest.mark.benchmark
```

**Recommendation:** Register the `benchmark` mark in `pytest.ini` or `pyproject.toml`:

```ini
[pytest]
markers =
    benchmark: profiling/timing tests (deselected by default)
```

---

## 3. File Handle Usage (Potential Leaks)

| File | Line | Issue |
|------|------|-------|
| `tools/install_parastell_in_vmec.sh` | 45–47 | `open(p).read()` and `open(p, 'w').write(s)` without context managers. On exception, handles may leak. |

**Recommendation:** Use `with open(p) as f: s = f.read()` and `with open(p, 'w') as f: f.write(s)` in the embedded Python snippet.

---

## 4. CI / Workflow Considerations

- **`update-db-self-hosted.yml`** references `stellcoilbench_ci` and `stellcoilbench_fork`; ensure self-hosted runner env matches.
- **`ci_filter_cases.py --github-output`**: If `GITHUB_OUTPUT` is unset (e.g. local run), script still prints JSON to stdout; CI always sets it.
- **GITHUB_OUTPUT values**: `cases_to_run_json` is written as `json.dumps(to_run)`. Paths with newlines or `=` could theoretically break parsing; unlikely in practice.

---

## 5. `run_ci_case` Flow Check

In `cli/submit_run.py`, `run_ci_case`:

- On **success**: `_write_ci_summary` is called in the `try` block; flow reaches line 325.
- On **caught exception** `(OSError, RuntimeError, ValueError, KeyError)`: `_write_ci_summary` is called in `except`; flow reaches line 325.
- On **validation failure**: `_write_ci_summary` is called, then `typer.Exit(1)` before line 325.
- On **other exception types** (e.g. `TypeError`, `AttributeError`): Exception propagates; line 325 is never reached.

**Edge case:** If `_write_ci_summary` fails (e.g. disk full) inside the `except` block, the original exception is replaced by the summary write failure. Consider wrapping `_write_ci_summary` in a try/except and logging both if desired.

---

## 6. Division-by-Zero / Numeric Edge Cases

- **`finite_build/_core.py`**: Tangent/normal/binormal norms checked before division (`< 1e-14`, `< 1e-10`).
- **`update_db/_constraints.py`**: `len(exponents)` used only after `if not exponents: return`, so division is safe.
- **`post_processing/_finite_build_runner.py`**: `minor_radius` used in division; caller ensures it is set.
- **`coil_optimization/_optimization_loop.py`**: `target_B` used in division; tolerance check guards against zero.

No obvious division-by-zero bugs found in the main paths.

---

## 7. None / AttributeError Patterns

- **`submission_io.py`**: Uses `data.get("metadata") or {}`, `data.get("metrics") or {}`, etc., and skips entries with missing metrics.
- **`update_db/submission_io.py` build_leaderboard_json**: Handles `composite_score is None` and missing `score_primary` with fallbacks.
- **`_constraints.py`**: `_resolve_constraint_value` returns `None` when metric is absent; callers handle it.

Null-safety patterns are generally consistent.

---

## 8. Chained `.get()` Usage

- `post_processing/_coil_io.py:227`: `case_data.get("surface_params", {}).get(...)`—safe due to `{}` default.
- `update_db/submission_io.py:348`: `entry.get("metrics", {}).keys()`—safe.

---

## 9. Test Coverage Gaps (Suggested)

- No dedicated tests for `run_ci_case` exception paths (e.g. `_write_ci_summary` failure, `_append_failure_to_autopilot_failures` failure).
- `ci_filter_cases.py`: Broad `except Exception` paths not explicitly tested for behavior (e.g. malformed YAML).
- MPI-specific tests exist; CI runs without MPI by default.

---

## 10. Dependency Warnings (External)

- **ezdxf**: `addParseAction`, `oneOf`, `setResultsName`, `infixNotation` deprecations (from ezdxf, not stellcoilbench).
- **scipy.optimize**: L-BFGS-B `disp`/`iprint` deprecated (from simsopt, not stellcoilbench).

---

## Prioritized Action Items

| Priority | Item | Effort |
|----------|------|--------|
| Low | Register `pytest.mark.benchmark` | Trivial |
| Low | Narrow `except Exception` in `ci_filter_cases.py` and `list_cases_cmd.py` | Small |
| Low | Use `with open()` in `install_parastell_in_vmec.sh` | Trivial |
| Medium | Add tests for `run_ci_case` failure paths | Medium |
| Medium | Review broad `except Exception` in `_parastell.py`, `_structural_mesh.py` for logging/scope | Medium |

---

## Conclusion

The codebase is generally robust with good defensive patterns (e.g. `proc0_try`, explicit `None` checks, safe `.get()` chains). The main improvement opportunities are:

1. Tighter exception handling to avoid masking bugs.
2. Minor file-handle and pytest configuration fixes.
3. More targeted tests for CI/autopilot failure paths.

No critical bugs or obvious runtime failures were identified in the surveyed code paths.
