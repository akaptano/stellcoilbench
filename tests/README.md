# StellCoilBench Unit Tests

This directory contains unit tests for all StellCoilBench source code modules.

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run tests with verbose output:
```bash
pytest tests/ -v
```

Run a specific test file:
```bash
pytest tests/test_config_scheme.py
```

Run a specific test:
```bash
pytest tests/test_config_scheme.py::TestCaseConfig::test_from_dict_minimal
```

Run tests with coverage:
```bash
pytest tests/ --cov=stellcoilbench --cov-report=html
```

## Test Structure

Tests are organized by module:

- **`test_config_scheme.py`** - Tests for `CaseConfig` and `SubmissionMetadata` dataclasses
- **`test_validate_config.py`** - Tests for configuration validation functions
- **`test_evaluate.py`** - Tests for case evaluation and leaderboard building functions
- **`test_update_db.py`** - Tests for database update and leaderboard generation functions
- **`test_coil_optimization.py`** - Tests for coil optimization utility functions
- **`test_leaderboard_integration.py`** - Integration tests for full leaderboard generation pipeline (converted from `test_leaderboard.sh`)

## Test Coverage

The test suite covers:

- ✅ Configuration loading and validation
- ✅ Dataclass creation and serialization
- ✅ YAML parsing and validation
- ✅ Submission loading and processing
- ✅ Leaderboard generation and sorting
- ✅ Metric shorthand and definition functions
- ✅ File I/O operations
- ✅ Error handling

## Note on Integration Tests

Full coil optimization tests require simsopt and actual optimization runs, which are considered integration tests. The unit tests focus on functions that can be tested without running full optimizations.

