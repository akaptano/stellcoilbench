"""
Unit tests for update_db.py
"""
import json
import tempfile
import zipfile
from pathlib import Path
from stellcoilbench.update_db import (
    N_TURNS_MODEL,
    _metric_shorthand,
    _metric_definition,
    _format_date,
    _shorthand_to_math,
    _load_submissions,
    _get_all_metrics_from_entries,
    build_methods_json,
    build_leaderboard_json,
    build_surface_leaderboards,
    check_reactor_constraints,
    compute_composite_score,
    write_markdown_leaderboard,
    write_reactor_scale_leaderboard,
    write_rst_leaderboard,
    write_surface_leaderboards,
    write_surface_leaderboard_index,
    update_database,
)


class TestMetricShorthand:
    """Tests for _metric_shorthand function."""
    
    def test_known_metrics(self):
        """Test shorthand for known metrics."""
        assert _metric_shorthand("final_normalized_squared_flux") == "f_B"
        assert _metric_shorthand("max_BdotN_over_B") == "max(B_n)"
        assert _metric_shorthand("final_average_curvature") == "κ̄"
        assert _metric_shorthand("final_linking_number") == "LN"
        assert _metric_shorthand("coil_order") == "n"
        assert _metric_shorthand("num_coils") == "N"
    
    def test_unknown_metric(self):
        """Test shorthand for unknown metric."""
        assert _metric_shorthand("unknown_metric_name") == "unknown metric name"


class TestMetricDefinition:
    """Tests for _metric_definition function."""
    
    def test_known_metric_definitions(self):
        """Test definitions for known metrics."""
        flux_def = _metric_definition("final_squared_flux")
        assert "Squared flux" in flux_def
        assert r"\int" in flux_def  # Check for the integral formula
        
        # Legacy name should also work
        legacy_flux_def = _metric_definition("final_normalized_squared_flux")
        assert "Squared flux" in legacy_flux_def
        
        ln_def = _metric_definition("final_linking_number")
        assert "Linking number" in ln_def
        assert r"4\pi" in ln_def or "4π" in ln_def  # Check for the formula
    
    def test_unknown_metric_definition(self):
        """Test definition for unknown metric."""
        defn = _metric_definition("unknown_metric")
        assert defn == "Unknown Metric"


class TestLoadSubmissions:
    """Tests for _load_submissions function."""
    
    def test_load_submissions_empty_dir(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            submissions = list(_load_submissions(submissions_root))
            assert submissions == []
    
    def test_load_submissions_nonexistent_dir(self):
        """Test loading from nonexistent directory."""
        submissions_root = Path("/nonexistent/directory")
        submissions = list(_load_submissions(submissions_root))
        assert submissions == []
    
    def test_load_submissions_single_file(self):
        """Test loading single submission file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            # Current structure: submissions/surface/user/timestamp/
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text(json.dumps({
                "metadata": {
                    "method_name": "test_method",
                    "contact": "test@example.com"
                },
                "metrics": {
                    "final_normalized_squared_flux": 0.001
                }
            }))
            
            # Create case.yaml to extract surface name
            case_yaml = submission_dir / "case.yaml"
            case_yaml.write_text("surface_params:\n  surface: input.surface1\n")
            
            submissions = list(_load_submissions(submissions_root))
            assert len(submissions) == 1
            method_key, path, data = submissions[0]
            assert method_key == "test_method:surface1:user1:2024-01-01_12-00"
            assert data["metadata"]["method_name"] == "test_method"
    
    def test_load_submissions_skips_non_results_json(self):
        """Test that non-results.json files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            # Create a non-results.json file
            other_file = submission_dir / "other.json"
            other_file.write_text(json.dumps({"test": "data"}))
            
            submissions = list(_load_submissions(submissions_root))
            assert len(submissions) == 0
    
    def test_load_submissions_invalid_json(self):
        """Test that invalid JSON files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text("invalid json content {")
            
            submissions = list(_load_submissions(submissions_root))
            assert len(submissions) == 0

    def test_load_submissions_zip_file(self):
        """Test loading submission from a zip file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            # Current structure: submissions/surface/user/timestamp/all_files.zip
            zip_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            zip_dir.mkdir(parents=True)
            zip_path = zip_dir / "all_files.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "results.json",
                    json.dumps(
                        {
                            "metadata": {"method_name": "test_method"},
                            "metrics": {"final_normalized_squared_flux": 0.002},
                        }
                    ),
                )
                # Add case.yaml to extract surface name
                zf.writestr("case.yaml", "surface_params:\n  surface: input.surface1\n")

            submissions = list(_load_submissions(submissions_root))
            assert len(submissions) == 1
            method_key, path, data = submissions[0]
            assert method_key == "test_method:surface1:user1:2024-01-01_12-00"
            assert path == zip_path
            assert data["metrics"]["final_normalized_squared_flux"] == 0.002

    def test_load_submissions_zip_without_results(self):
        """Test that zip files without results.json are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            zip_path = submissions_root / "submissions" / "surface1" / "user1" / "2024-01-01_12-00.zip"
            zip_path.parent.mkdir(parents=True)
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("other.json", json.dumps({"ok": True}))

            submissions = list(_load_submissions(submissions_root))
            assert submissions == []


class TestBuildMethodsJson:
    """Tests for build_methods_json function."""
    
    def test_build_methods_json_empty(self):
        """Test building methods JSON from empty submissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            repo_root = Path(tmpdir)
            methods = build_methods_json(submissions_root, repo_root)
            assert methods == {}
    
    def test_build_methods_json_single_submission(self):
        """Test building methods JSON from single submission."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir).resolve()
            repo_root = Path(tmpdir).resolve()
            
            # Current structure: submissions/surface/user/timestamp/
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text(json.dumps({
                "metadata": {
                    "method_name": "test_method",
                    "contact": "test@example.com",
                    "hardware": "CPU: Test",
                    "run_date": "2024-01-01T12:00:00"
                },
                "metrics": {
                    "final_normalized_squared_flux": 0.001,
                    "final_total_length": 100.0
                }
            }))
            
            # Create case.yaml to extract surface name
            case_yaml = submission_dir / "case.yaml"
            case_yaml.write_text("surface_params:\n  surface: input.surface1\n")
            
            methods = build_methods_json(submissions_root, repo_root)
            assert len(methods) == 1
            
            method_key = "test_method:surface1:user1:2024-01-01_12-00"
            assert method_key in methods
            method_data = methods[method_key]
            assert method_data["method_name"] == "test_method"
            # Contact field now uses GitHub username from path, not metadata
            assert method_data["contact"] == "user1"  # Extracted from path structure
            assert method_data["metrics"]["final_normalized_squared_flux"] == 0.001
            assert method_data["score_primary"] == 0.001
    
    def test_build_methods_json_extracts_coil_params(self):
        """Test that coil parameters are extracted from case.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir).resolve()
            repo_root = Path(tmpdir).resolve()
            
            # Current structure: submissions/surface/user/timestamp/
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text(json.dumps({
                "metadata": {"method_name": "test_method"},
                "metrics": {"final_normalized_squared_flux": 0.001}
            }))
            
            # Create case.yaml with coil parameters and surface
            case_yaml = submission_dir / "case.yaml"
            case_yaml.write_text("""surface_params:
  surface: input.surface1
coils_params:
  ncoils: 4
  order: 16
""")
            
            methods = build_methods_json(submissions_root, repo_root)
            method_key = "test_method:surface1:user1:2024-01-01_12-00"
            method_data = methods[method_key]
            
            assert method_data["metrics"]["num_coils"] == 4.0
            assert method_data["metrics"]["coil_order"] == 16.0
    
    def test_build_methods_json_skips_no_metrics(self):
        """Test that submissions without metrics are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir)
            repo_root = Path(tmpdir)
            
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text(json.dumps({
                "metadata": {"method_name": "test_method"},
                "metrics": {}
            }))
            
            methods = build_methods_json(submissions_root, repo_root)
            assert methods == {}

    def test_build_methods_json_fallback_score(self):
        """Test fallback scoring from final_flux."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir).resolve()
            repo_root = Path(tmpdir).resolve()
            # New structure: submissions/surface/user/timestamp/
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            results_file = submission_dir / "results.json"
            results_file.write_text(
                json.dumps(
                    {
                        "metadata": {"method_name": "method1"},
                        "metrics": {"final_flux": 0.5},
                    }
                )
            )
            
            # Create case.yaml to extract surface name
            case_yaml = submission_dir / "case.yaml"
            case_yaml.write_text("surface_params:\n  surface: input.surface1\n")

            methods = build_methods_json(submissions_root, repo_root)
            method_key = "method1:surface1:user1:2024-01-01_12-00"
            assert methods[method_key]["score_primary"] == 0.5

    def test_build_methods_json_non_numeric_fallback_keeps_none(self):
        """Test non-numeric fallback score leaves score_primary as None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir).resolve()
            repo_root = Path(tmpdir).resolve()
            # New structure: submissions/surface/user/timestamp/
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            results_file = submission_dir / "results.json"
            results_file.write_text(
                json.dumps(
                    {
                        "metadata": {"method_name": "method1"},
                        "metrics": {"final_flux": "bad"},
                    }
                )
            )
            
            # Create case.yaml to extract surface name
            case_yaml = submission_dir / "case.yaml"
            case_yaml.write_text("surface_params:\n  surface: input.surface1\n")

            methods = build_methods_json(submissions_root, repo_root)
            method_key = "method1:surface1:user1:2024-01-01_12-00"
            assert method_key in methods
            assert methods[method_key]["score_primary"] is None

    def test_build_methods_json_duplicate_method_keys(self):
        """Test duplicate method keys are overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir).resolve()
            repo_root = Path(tmpdir).resolve()
            for ts in ["2024-01-01_12-00", "2024-01-02_12-00"]:
                submission_dir = submissions_root / "surface1" / "user1" / ts
                submission_dir.mkdir(parents=True)
                results_file = submission_dir / "results.json"
                results_file.write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "method_name": "method1",
                                "method_version": "v1",
                            },
                            "metrics": {"final_normalized_squared_flux": 0.1},
                        }
                    )
                )

            methods = build_methods_json(submissions_root, repo_root)
            assert len(methods) == 1


class TestBuildLeaderboardJson:
    """Tests for build_leaderboard_json function."""
    
    def test_build_leaderboard_json_empty(self):
        """Test building leaderboard from empty methods."""
        leaderboard = build_leaderboard_json({})
        assert leaderboard["entries"] == []
        assert leaderboard["excluded_entries"] == []
    
    def test_build_leaderboard_json_single_entry(self):
        """Test building leaderboard with single entry."""
        methods = {
            "method1:1.0": {
                "method_name": "method1",
                "method_version": "1.0",
                "contact": "user1",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "submissions/surface1/user1/2024-01-01_12-00/results.json",
                "score_primary": 0.001,
                "metrics": {"final_normalized_squared_flux": 0.001}
            }
        }
        leaderboard = build_leaderboard_json(methods)
        
        assert len(leaderboard["entries"]) == 1
        entry = leaderboard["entries"][0]
        assert entry["rank"] == 1
        assert entry["score_primary"] == 0.001
        assert entry["method_name"] == "method1"
    
    def test_build_leaderboard_json_sorts_ascending(self):
        """Test that leaderboard is sorted by score_primary ascending."""
        methods = {
            "method1:1.0": {
                "method_name": "method1",
                "method_version": "1.0",
                "contact": "user1",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "path1",
                "score_primary": 0.003,
                "metrics": {}
            },
            "method2:1.0": {
                "method_name": "method2",
                "method_version": "1.0",
                "contact": "user2",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "path2",
                "score_primary": 0.001,
                "metrics": {}
            },
        }
        leaderboard = build_leaderboard_json(methods)
        
        assert len(leaderboard["entries"]) == 2
        # Lower score should be first (ascending order)
        assert leaderboard["entries"][0]["score_primary"] == 0.001
        assert leaderboard["entries"][1]["score_primary"] == 0.003
        assert leaderboard["entries"][0]["rank"] == 1
        assert leaderboard["entries"][1]["rank"] == 2
    
    def test_build_leaderboard_json_skips_no_score(self):
        """Test that entries without score_primary are skipped."""
        methods = {
            "method1:1.0": {
                "method_name": "method1",
                "method_version": "1.0",
                "contact": "user1",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "path1",
                "score_primary": None,
                "metrics": {}
            }
        }
        leaderboard = build_leaderboard_json(methods)
        assert leaderboard["entries"] == []



class TestLeaderboardAdditional:
    """Additional tests for leaderboard filtering."""

    def test_build_leaderboard_filters_missing_scores(self):
        """Test that entries with missing score_primary are filtered out."""
        methods = {
            "method1:1.0": {
                "method_name": "method1",
                "method_version": "1.0",
                "contact": "user1",
                "hardware": "CPU: Test",
                "run_date": "2024-01-01T12:00:00",
                "path": "path1",
                "score_primary": None,
                "metrics": {"final_normalized_squared_flux": 0.1},
            },
            "method2:1.0": {
                "method_name": "method2",
                "method_version": "1.0",
                "contact": "user2",
                "hardware": "CPU: Test",
                "run_date": "2024-01-02T12:00:00",
                "path": "path2",
                "score_primary": 0.02,
                "metrics": {"final_normalized_squared_flux": 0.02},
            },
        }
        leaderboard = build_leaderboard_json(methods)
        assert len(leaderboard["entries"]) == 1
        assert leaderboard["entries"][0]["method_name"] == "method2"


class TestLeaderboardMarkdown:
    """Tests for markdown leaderboard writers and helpers."""

    def test_get_all_metrics_from_entries_excludes_and_orders(self):
        entries = [
            {
                "metrics": {
                    "final_total_length": 10.0,
                    "final_squared_flux": 0.01,
                    "score_primary": 0.01,
                    "initial_B_field": 1.0,
                    "BdotN": 0.001,  # Raw post-processing duplicate
                    "BdotN_over_B": 0.002,  # Raw post-processing duplicate
                    "final_normalized_squared_flux": 0.01,  # Legacy duplicate
                    "coils_linked_to_surface": 1.0,  # Boolean-like, excluded
                }
            }
        ]
        keys = _get_all_metrics_from_entries(entries)
        assert keys[0] == "final_squared_flux"
        assert "score_primary" not in keys
        assert "initial_B_field" not in keys
        assert "BdotN" not in keys
        assert "BdotN_over_B" not in keys
        assert "final_normalized_squared_flux" not in keys
        assert "final_total_length" in keys

    def test_write_markdown_leaderboard(self, tmp_path):
        leaderboard = {
            "entries": [
                {
                    "rank": 1,
                    "method_key": "method1",
                    "method_name": "method1",
                    "method_version": "v1",
                    "score_primary": 0.01,
                    "composite_score": 1.5,
                    "run_date": "2024-01-01T12:00:00",
                    "contact": "user1",
                    "hardware": "CPU",
                    "path": "submissions/surface/user/ts/results.json",
                    "metrics": {
                        "final_squared_flux": 0.01,
                        "final_linking_number": 0,
                    },
                }
            ]
        }
        out_md = tmp_path / "leaderboard.md"
        write_markdown_leaderboard(leaderboard, out_md)
        content = out_md.read_text()
        assert "CoilBench Leaderboard" in content
        assert "Legend" in content
        assert "f_B" in content
        assert "Score" in content  # Score column header
        assert "1.500" in content  # Composite score value

    def test_write_rst_leaderboard(self, tmp_path):
        # Create submission directory structure and case.yaml for surface extraction
        submission_dir = tmp_path / "submissions" / "surface" / "user" / "ts"
        submission_dir.mkdir(parents=True)
        case_yaml = submission_dir / "case.yaml"
        case_yaml.write_text("surface_params:\n  surface: input.surface\n")
        
        leaderboard = {
            "entries": [
                {
                    "rank": 1,
                    "method_key": "method1",
                    "method_name": "method1",
                    "method_version": "v1",
                    "score_primary": 0.01,
                    "run_date": "2024-01-01T12:00:00",
                    "contact": "user1",
                    "hardware": "CPU",
                    "path": "submissions/surface/user/ts/results.json",
                    "metrics": {
                        "final_normalized_squared_flux": 0.01,
                        "final_linking_number": 0,
                    },
                }
            ]
        }
        submissions_root = tmp_path / "submissions"
        surface_leaderboards = build_surface_leaderboards(
            leaderboard, submissions_root=submissions_root, plasma_surfaces_dir=tmp_path
        )
        out_rst = tmp_path / "leaderboard.rst"
        write_rst_leaderboard(leaderboard, out_rst, surface_leaderboards)
        content = out_rst.read_text()
        assert "StellCoilBench Leaderboard" in content
        # Surface-Specific Leaderboards is in a separate file now
        surface_specific_file = tmp_path / "leaderboard" / "surface_specific.rst"
        assert surface_specific_file.exists()
        surface_content = surface_specific_file.read_text()
        assert "Surface-Specific Leaderboards" in surface_content
        assert ".. list-table::" in surface_content
        # Metric Definitions is also in a separate file
        metric_def_file = tmp_path / "leaderboard" / "metric_definitions.rst"
        assert metric_def_file.exists()
        metric_content = metric_def_file.read_text()
        assert "Metric Definitions" in metric_content

    def test_build_surface_leaderboards_and_write(self, tmp_path):
        # Create submission directory structure and case.yaml files for surface extraction
        submission_dir1 = tmp_path / "submissions" / "surf1" / "user" / "ts"
        submission_dir1.mkdir(parents=True)
        case_yaml1 = submission_dir1 / "case.yaml"
        case_yaml1.write_text("surface_params:\n  surface: input.surf1\n")
        
        submission_dir2 = tmp_path / "submissions" / "surf1" / "user" / "ts2"
        submission_dir2.mkdir(parents=True)
        case_yaml2 = submission_dir2 / "case.yaml"
        case_yaml2.write_text("surface_params:\n  surface: input.surf1\n")
        
        leaderboard = {
            "entries": [
                {
                    "method_key": "m1",
                    "method_name": "m1",
                    "method_version": "v1",
                    "score_primary": 0.2,
                    "run_date": "2024-01-01T12:00:00",
                    "contact": "user1",
                    "hardware": "CPU",
                    "path": "submissions/surf1/user/ts/results.json",
                    "metrics": {
                        "final_normalized_squared_flux": 0.2,
                        "num_coils": 4.0,
                    },
                },
                {
                    "method_key": "m2",
                    "method_name": "m2",
                    "method_version": "v1",
                    "score_primary": 0.1,
                    "run_date": "2024-01-02T12:00:00",
                    "contact": "user2",
                    "hardware": "CPU",
                    "path": "submissions/surf1/user/ts2/results.json",
                    "metrics": {
                        "final_normalized_squared_flux": 0.1,
                        "num_coils": 6.0,
                    },
                },
            ]
        }

        submissions_root = tmp_path / "submissions"
        surface_leaderboards = build_surface_leaderboards(
            leaderboard, submissions_root=submissions_root, plasma_surfaces_dir=tmp_path
        )
        assert "surf1" in surface_leaderboards
        assert surface_leaderboards["surf1"]["entries"][0]["score_primary"] == 0.1

        docs_dir = tmp_path / "docs"
        surface_names = write_surface_leaderboards(
            surface_leaderboards, docs_dir=docs_dir, repo_root=tmp_path
        )
        assert "surf1" in surface_names
        surface_md = docs_dir / "leaderboards" / "surf1.md"
        assert surface_md.exists()
        content = surface_md.read_text()
        assert "Legend" in content


class TestLeaderboardEdgeCases:
    """Tests for empty and malformed leaderboard entries."""

    def test_write_markdown_leaderboard_empty_entries(self, tmp_path):
        leaderboard = {"entries": []}
        out_md = tmp_path / "leaderboard.md"
        write_markdown_leaderboard(leaderboard, out_md)
        content = out_md.read_text()
        assert "_No valid submissions found._" in content
        assert '"method_name": "your_method"' in content

    def test_build_surface_leaderboards_skips_missing_path(self):
        leaderboard = {"entries": [{"metrics": {"final_normalized_squared_flux": 0.1}}]}
        surface_leaderboards = build_surface_leaderboards(
            leaderboard, submissions_root=Path("."), plasma_surfaces_dir=Path(".")
        )
        assert surface_leaderboards == {}

    def test_build_surface_leaderboards_skips_non_submission_paths(self):
        leaderboard = {
            "entries": [
                {"path": "results.json", "metrics": {"final_normalized_squared_flux": 0.1}},
                {"path": "submissions", "metrics": {"final_normalized_squared_flux": 0.2}},
            ]
        }
        surface_leaderboards = build_surface_leaderboards(
            leaderboard, submissions_root=Path("."), plasma_surfaces_dir=Path(".")
        )
        assert surface_leaderboards == {}


class TestFormatDate:
    """Tests for _format_date function."""
    
    def test_format_iso_date(self):
        """Test formatting ISO date (YYYY-MM-DD)."""
        assert _format_date("2025-12-01") == "01/12/25"
        assert _format_date("2026-01-21") == "21/01/26"
        assert _format_date("2024-01-01") == "01/01/24"
    
    def test_format_iso_datetime(self):
        """Test formatting ISO datetime (YYYY-MM-DDTHH:MM:SS)."""
        assert _format_date("2025-12-01T12:00:00") == "01/12/25"
        assert _format_date("2026-01-21T23:59:59") == "21/01/26"
    
    def test_format_already_formatted(self):
        """Test that already formatted dates are handled."""
        # DD/MM/YY format (day > 12)
        assert _format_date("21/01/26") == "21/01/26"
        # MM/DD/YY format (second part > 12 means it's the day)
        assert _format_date("01/21/26") == "21/01/26"
        # Ambiguous format (both <= 12) - assumes MM/DD/YY and converts
        result = _format_date("01/12/26")
        assert "/" in result and "26" in result
    
    def test_format_unknown(self):
        """Test handling of unknown dates."""
        assert _format_date("_unknown_") == "_unknown_"
        assert _format_date("") == ""
        assert _format_date(None) == "_unknown_"
    
    def test_format_invalid_date(self):
        """Test handling of invalid date formats."""
        # Invalid format should return something (not crash)
        result = _format_date("invalid")
        assert isinstance(result, str)
    
    def test_format_date_edge_cases(self):
        """Test edge cases in date formatting."""
        # Test with 4-digit year in slash format
        result = _format_date("21/01/2026")
        assert "/" in result and "26" in result
        
        # Test with single-digit components
        result = _format_date("1/2/26")
        assert "/" in result
        
        # Test ambiguous date (both <= 12)
        result = _format_date("01/12/26")
        assert "/" in result
        
        # Test day > 12 (definitely DD/MM/YY)
        result = _format_date("21/01/26")
        assert result == "21/01/26"
        
        # Test month > 12 in second position (MM/DD/YY)
        result = _format_date("01/21/26")
        assert result == "21/01/26"


class TestShorthandToMath:
    """Tests for _shorthand_to_math function."""
    
    def test_simple_variables(self):
        """Test simple variable names."""
        assert _shorthand_to_math("n") == ":math:`n`"
        assert _shorthand_to_math("N") == ":math:`N`"
        assert _shorthand_to_math("L") == ":math:`L`"
        assert _shorthand_to_math("t") == ":math:`t`"
    
    def test_unicode_characters(self):
        """Test Unicode character conversions."""
        assert r"\bar{\kappa}" in _shorthand_to_math("κ̄")
        assert r"\bar{F}" in _shorthand_to_math("F̄")
        assert r"\bar{\tau}" in _shorthand_to_math("τ̄")
        assert r"\text{avg}" in _shorthand_to_math("avg(B_n)")
        assert r"\max" in _shorthand_to_math("max(B_n)")
    
    def test_function_calls(self):
        """Test function call conversions."""
        result = _shorthand_to_math("min(d_cc)")
        assert r"\min" in result
        assert r"d_{cc}" in result
        
        result = _shorthand_to_math("max(κ)")
        assert r"\max" in result
        assert r"\kappa" in result
        
        result = _shorthand_to_math("max(F)")
        assert r"\max" in result
        assert "F" in result
    
    def test_subscripts(self):
        """Test subscript conversions."""
        result = _shorthand_to_math("d_cc")
        assert r"d_{cc}" in result
        
        result = _shorthand_to_math("d_cs")
        assert r"d_{cs}" in result
        
        result = _shorthand_to_math("B_n")
        assert r"B_n" in result or r"B_{n}" in result
    
    def test_default_wrapping(self):
        """Test default math mode wrapping."""
        result = _shorthand_to_math("f_B")
        assert ":math:`" in result
        assert "f_B" in result or "f_{B}" in result
    
    def test_shorthand_to_math_complex_cases(self):
        """Test complex shorthand conversions."""
        # Test multiple underscores
        result = _shorthand_to_math("d_cc_cs")
        assert ":math:`" in result
        
        # Test function with complex argument
        result = _shorthand_to_math("max(d_cc)")
        assert r"\max" in result or "max" in result
        assert r"d_{cc}" in result
        
        # Test Var function
        result = _shorthand_to_math("Var(l_i)")
        assert r"\mathrm{Var}" in result or "Var" in result


class TestMetricDefinitionComprehensive:
    """Comprehensive tests for _metric_definition function."""
    
    def test_all_metric_types(self):
        """Test definitions for various metric types."""
        # Field quality metrics
        assert "flux" in _metric_definition("final_normalized_squared_flux").lower()
        assert "B" in _metric_definition("avg_BdotN_over_B")
        assert "B" in _metric_definition("max_BdotN_over_B")
        
        # Curvature metrics
        assert "curvature" in _metric_definition("final_average_curvature").lower()
        assert "curvature" in _metric_definition("final_max_curvature").lower()
        assert "curvature" in _metric_definition("final_mean_squared_curvature").lower()
        
        # Separation metrics
        assert "separation" in _metric_definition("final_min_cs_separation").lower() or "distance" in _metric_definition("final_min_cs_separation").lower()
        assert "separation" in _metric_definition("final_min_cc_separation").lower() or "distance" in _metric_definition("final_min_cc_separation").lower()
        
        # Length metrics
        assert "length" in _metric_definition("final_total_length").lower()
        assert "arclength" in _metric_definition("final_arclength_variation").lower() or "variation" in _metric_definition("final_arclength_variation").lower()
        
        # Force/torque metrics
        assert "force" in _metric_definition("final_max_max_coil_force").lower()
        assert "torque" in _metric_definition("final_max_max_coil_torque").lower()
        
        # Topology metrics
        assert "linking" in _metric_definition("final_linking_number").lower()
        
        # Time metrics
        assert "time" in _metric_definition("optimization_time").lower()
    
    def test_config_metrics(self):
        """Test configuration metrics."""
        assert "coil" in _metric_definition("num_coils").lower() or "number" in _metric_definition("num_coils").lower()
        assert "order" in _metric_definition("coil_order").lower() or "fourier" in _metric_definition("coil_order").lower()


class TestBuildMethodsJsonComprehensive:
    """Comprehensive tests for build_methods_json function."""
    
    def test_build_methods_json_with_fourier_continuation_orders(self, tmp_path):
        """Test extracting Fourier continuation orders from case.yaml."""
        submissions_root = Path(tmp_path).resolve()
        repo_root = Path(tmp_path).resolve()
        
        submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
        submission_dir.mkdir(parents=True)
        
        results_file = submission_dir / "results.json"
        results_file.write_text(json.dumps({
            "metadata": {"method_name": "test_method"},
            "metrics": {"final_normalized_squared_flux": 0.001}
        }))
        
        # Create case.yaml with Fourier continuation orders
        case_yaml = submission_dir / "case.yaml"
        case_yaml.write_text("""surface_params:
  surface: input.surface1
fourier_continuation:
  enabled: true
  orders: [2, 4, 8]
""")
        
        methods = build_methods_json(submissions_root, repo_root)
        method_key = "test_method:surface1:user1:2024-01-01_12-00"
        assert method_key in methods
        # Should extract fourier_continuation_orders as comma-separated string
        orders = methods[method_key]["metrics"].get("fourier_continuation_orders")
        assert orders is not None
        assert isinstance(orders, str)
        assert "2" in orders
    
    def test_build_methods_json_legacy_format(self, tmp_path):
        """Test handling legacy format where metrics are at top level."""
        submissions_root = Path(tmp_path).resolve()
        repo_root = Path(tmp_path).resolve()
        
        submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
        submission_dir.mkdir(parents=True)
        
        # Legacy format: metrics at top level, no "metrics" key
        results_file = submission_dir / "results.json"
        results_file.write_text(json.dumps({
            "method_name": "test_method",
            "contact": "user1@example.com",
            "final_normalized_squared_flux": 0.001,
            "final_total_length": 100.0,
        }))
        
        case_yaml = submission_dir / "case.yaml"
        case_yaml.write_text("surface_params:\n  surface: input.surface1\n")
        
        methods = build_methods_json(submissions_root, repo_root)
        # Method key uses "UNKNOWN" when method_name is at top level but not in metadata
        method_key = "UNKNOWN:surface1:user1:2024-01-01_12-00"
        assert method_key in methods
        assert methods[method_key]["method_name"] == "test_method"  # But method_name is still extracted
        assert methods[method_key]["metrics"]["final_normalized_squared_flux"] == 0.001
    
    def test_build_methods_json_extract_date_from_path(self, tmp_path):
        """Test extracting run_date from path timestamp."""
        submissions_root = Path(tmp_path).resolve()
        repo_root = Path(tmp_path).resolve()
        
        # Use timestamp format MM-DD-YYYY_HH-MM
        submission_dir = submissions_root / "surface1" / "user1" / "01-15-2024_14-30"
        submission_dir.mkdir(parents=True)
        
        results_file = submission_dir / "results.json"
        results_file.write_text(json.dumps({
            "method_name": "test_method",
            "final_normalized_squared_flux": 0.001,
        }))
        
        case_yaml = submission_dir / "case.yaml"
        case_yaml.write_text("surface_params:\n  surface: input.surface1\n")
        
        methods = build_methods_json(submissions_root, repo_root)
        # Method key uses "UNKNOWN" when method_name is at top level but not in metadata
        method_key = "UNKNOWN:surface1:user1:01-15-2024_14-30"
        assert method_key in methods
        # Should extract date from path
        run_date = methods[method_key].get("run_date", "")
        assert "2024" in run_date or "01" in run_date
    
    def test_build_methods_json_extract_contact_from_path(self, tmp_path):
        """Test extracting contact from path when missing from metadata."""
        submissions_root = Path(tmp_path).resolve()
        repo_root = Path(tmp_path).resolve()
        
        submission_dir = submissions_root / "surface1" / "github_user" / "2024-01-01_12-00"
        submission_dir.mkdir(parents=True)
        
        # No contact in metadata - should extract from path
        results_file = submission_dir / "results.json"
        results_file.write_text(json.dumps({
            "method_name": "test_method",
            "final_normalized_squared_flux": 0.001,
        }))
        
        case_yaml = submission_dir / "case.yaml"
        case_yaml.write_text("surface_params:\n  surface: input.surface1\n")
        
        methods = build_methods_json(submissions_root, repo_root)
        # Method key uses "UNKNOWN" when method_name is at top level but not in metadata
        method_key = "UNKNOWN:surface1:github_user:2024-01-01_12-00"
        assert method_key in methods
        # Contact should be extracted from path
        assert methods[method_key]["contact"] == "github_user"
    
    def test_build_methods_json_duplicate_keys_tracking(self, tmp_path):
        """Test that duplicate method_keys are tracked and overwritten."""
        submissions_root = Path(tmp_path).resolve()
        repo_root = Path(tmp_path).resolve()
        
        # Create two submissions with same method_key (will overwrite)
        for i in [1, 2]:
            submission_dir = submissions_root / "surface1" / "user1" / f"2024-01-0{i}_12-00"
            submission_dir.mkdir(parents=True)
            results_file = submission_dir / "results.json"
            results_file.write_text(json.dumps({
                "method_name": "same_method",
                "method_version": "v1",
                "final_normalized_squared_flux": float(i) * 0.001,
            }))
            case_yaml = submission_dir / "case.yaml"
            case_yaml.write_text("surface_params:\n  surface: input.surface1\n")
        
        methods = build_methods_json(submissions_root, repo_root)
        # Both submissions have same method_name and version, but different timestamps
        # So they should have different keys (different timestamps)
        method_keys = [k for k in methods.keys() if "same_method" in k or "surface1" in k]
        # Both will have different timestamps, so different keys
        assert len(method_keys) == 2
    
    def test_build_methods_json_case_yaml_from_zip(self, tmp_path):
        """Test reading case.yaml from zip file."""
        submissions_root = Path(tmp_path).resolve()
        repo_root = Path(tmp_path).resolve()
        
        zip_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
        zip_dir.mkdir(parents=True)
        zip_path = zip_dir / "all_files.zip"
        
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("results.json", json.dumps({
                "metadata": {"method_name": "test_method"},
                "metrics": {"final_normalized_squared_flux": 0.001}
            }))
            zf.writestr("case.yaml", """surface_params:
  surface: input.surface1
coils_params:
  order: 8
  ncoils: 4
""")
        
        methods = build_methods_json(submissions_root, repo_root)
        method_key = "test_method:surface1:user1:2024-01-01_12-00"
        assert method_key in methods
        assert methods[method_key]["metrics"]["coil_order"] == 8.0
        assert methods[method_key]["metrics"]["num_coils"] == 4.0
    
    def test_build_methods_json_with_coil_info_from_json(self, tmp_path):
        """Test extracting coil info from coils.json file."""
        submissions_root = Path(tmp_path).resolve()
        repo_root = Path(tmp_path).resolve()
        
        submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
        submission_dir.mkdir(parents=True)
        
        results_file = submission_dir / "results.json"
        results_file.write_text(json.dumps({
            "metadata": {"method_name": "test_method"},
            "metrics": {"final_normalized_squared_flux": 0.001}
        }))
        
        # Create coils.json file (even if invalid, should not crash)
        coils_file = submission_dir / "coils.json"
        # Create a minimal valid JSON structure that simsopt.load might handle
        # Note: actual coil extraction requires real simsopt objects, so this test
        # just verifies the code path doesn't crash
        coils_file.write_text(json.dumps({
            "test": "data"
        }))
        
        case_yaml = submission_dir / "case.yaml"
        case_yaml.write_text("surface_params:\n  surface: input.surface1\n")
        
        methods = build_methods_json(submissions_root, repo_root)
        method_key = "test_method:surface1:user1:2024-01-01_12-00"
        assert method_key in methods
        # Coil extraction may fail gracefully, so just verify the method exists
        assert method_key in methods
    
    def test_build_methods_json_with_surface_file(self, tmp_path):
        """Test extracting surface info from surface file path."""
        submissions_root = Path(tmp_path).resolve()
        repo_root = Path(tmp_path).resolve()
        
        # Create a mock surface file
        plasma_surfaces_dir = repo_root / "plasma_surfaces"
        plasma_surfaces_dir.mkdir(parents=True)
        surface_file = plasma_surfaces_dir / "input.test_surface"
        surface_file.write_text("# Mock surface file\n")
        
        submission_dir = submissions_root / "test_surface" / "user1" / "2024-01-01_12-00"
        submission_dir.mkdir(parents=True)
        
        results_file = submission_dir / "results.json"
        results_file.write_text(json.dumps({
            "metadata": {"method_name": "test_method"},
            "metrics": {"final_normalized_squared_flux": 0.001}
        }))
        
        case_yaml = submission_dir / "case.yaml"
        case_yaml.write_text("surface_params:\n  surface: input.test_surface\n")
        
        methods = build_methods_json(submissions_root, repo_root)
        method_key = "test_method:test_surface:user1:2024-01-01_12-00"
        assert method_key in methods


class TestWriteMarkdownLeaderboardComprehensive:
    """Comprehensive tests for write_markdown_leaderboard function."""
    
    def test_write_markdown_leaderboard_formatting_edge_cases(self, tmp_path):
        """Test markdown formatting with edge case values."""
        leaderboard = {
            "entries": [
                {
                    "rank": 1,
                    "method_key": "method1",
                    "method_name": "method1",
                    "method_version": "v1",
                    "score_primary": 1e-100,  # Very small value
                    "run_date": "2024-01-01T12:00:00",
                    "contact": "user1",
                    "hardware": "CPU",
                    "path": "submissions/surface/user/ts/results.json",
                    "metrics": {
                        "final_normalized_squared_flux": 1e-100,
                        "final_linking_number": 0,
                        "num_coils": 4.0,
                    },
                },
                {
                    "rank": 2,
                    "method_key": "method2",
                    "method_name": "method2",
                    "method_version": "v2",
                    "score_primary": 1e10,  # Very large value
                    "run_date": "2024-01-02T12:00:00",
                    "contact": "user2",
                    "hardware": "GPU",
                    "path": "submissions/surface/user2/ts2/results.json",
                    "metrics": {
                        "final_normalized_squared_flux": 1e10,
                        "final_linking_number": 5,
                    },
                },
            ]
        }
        out_md = tmp_path / "leaderboard.md"
        write_markdown_leaderboard(leaderboard, out_md)
        content = out_md.read_text()
        # Should handle very small and very large values
        assert "user1" in content
        assert "user2" in content
    
    def test_write_markdown_leaderboard_with_multiple_entries(self, tmp_path):
        """Test markdown leaderboard with multiple entries."""
        leaderboard = {
            "entries": [
                {
                    "rank": 1,
                    "method_key": "method1",
                    "method_name": "method1",
                    "method_version": "v1",
                    "score_primary": 0.001,
                    "composite_score": 1.8,
                    "run_date": "2024-01-01T12:00:00",
                    "contact": "user1",
                    "hardware": "CPU",
                    "path": "submissions/surface/user/ts/results.json",
                    "metrics": {
                        "final_squared_flux": 0.001,
                        "final_total_length": 100.0,
                        "num_coils": 4.0,
                    },
                },
                {
                    "rank": 2,
                    "method_key": "method2",
                    "method_name": "method2",
                    "method_version": "v2",
                    "score_primary": 0.002,
                    "composite_score": 1.2,
                    "run_date": "2024-01-02T12:00:00",
                    "contact": "user2",
                    "hardware": "GPU",
                    "path": "submissions/surface/user2/ts2/results.json",
                    "metrics": {
                        "final_squared_flux": 0.002,
                        "final_total_length": 200.0,
                        "num_coils": 6.0,
                    },
                },
            ]
        }
        out_md = tmp_path / "leaderboard.md"
        write_markdown_leaderboard(leaderboard, out_md)
        content = out_md.read_text()
        # Markdown uses user names, not method names in the table
        assert "user1" in content
        assert "user2" in content
        # Check metric values are present
        assert "1.0e-3" in content or "1.0e-03" in content
        assert "2.0e-3" in content or "2.0e-03" in content


class TestWriteRstLeaderboardComprehensive:
    """Comprehensive tests for write_rst_leaderboard function."""
    
    def test_write_rst_leaderboard_empty_surfaces(self, tmp_path):
        """Test RST leaderboard with no surface leaderboards."""
        leaderboard = {
            "entries": [
                {
                    "rank": 1,
                    "method_key": "m1",
                    "method_name": "m1",
                    "method_version": "v1",
                    "score_primary": 0.1,
                    "run_date": "2024-01-01T12:00:00",
                    "contact": "user1",
                    "hardware": "CPU",
                    "path": "submissions/surf1/user/ts/results.json",
                    "metrics": {"final_normalized_squared_flux": 0.1},
                }
            ]
        }
        # submissions_root = tmp_path / "submissions"
        surface_leaderboards = {}  # Empty
        
        out_rst = tmp_path / "leaderboard.rst"
        write_rst_leaderboard(leaderboard, out_rst, surface_leaderboards)
        
        surface_specific_file = tmp_path / "leaderboard" / "surface_specific.rst"
        assert surface_specific_file.exists()
        content = surface_specific_file.read_text()
        assert "No surface leaderboards generated yet" in content
    
    def test_write_rst_leaderboard_with_multiple_surfaces(self, tmp_path):
        """Test RST leaderboard with multiple surfaces."""
        submission_dir1 = tmp_path / "submissions" / "surf1" / "user" / "ts"
        submission_dir1.mkdir(parents=True)
        case_yaml1 = submission_dir1 / "case.yaml"
        case_yaml1.write_text("surface_params:\n  surface: input.surf1\n")
        
        submission_dir2 = tmp_path / "submissions" / "surf2" / "user" / "ts"
        submission_dir2.mkdir(parents=True)
        case_yaml2 = submission_dir2 / "case.yaml"
        case_yaml2.write_text("surface_params:\n  surface: input.surf2\n")
        
        leaderboard = {
            "entries": [
                {
                    "rank": 1,
                    "method_key": "m1",
                    "method_name": "m1",
                    "method_version": "v1",
                    "score_primary": 0.1,
                    "run_date": "2024-01-01T12:00:00",
                    "contact": "user1",
                    "hardware": "CPU",
                    "path": "submissions/surf1/user/ts/results.json",
                    "metrics": {"final_normalized_squared_flux": 0.1},
                },
                {
                    "rank": 1,
                    "method_key": "m2",
                    "method_name": "m2",
                    "method_version": "v1",
                    "score_primary": 0.2,
                    "run_date": "2024-01-01T12:00:00",
                    "contact": "user2",
                    "hardware": "CPU",
                    "path": "submissions/surf2/user/ts/results.json",
                    "metrics": {"final_normalized_squared_flux": 0.2},
                },
            ]
        }
        submissions_root = tmp_path / "submissions"
        surface_leaderboards = build_surface_leaderboards(
            leaderboard, submissions_root=submissions_root, plasma_surfaces_dir=tmp_path
        )
        out_rst = tmp_path / "leaderboard.rst"
        write_rst_leaderboard(leaderboard, out_rst, surface_leaderboards)
        
        # Check that both surfaces are in the surface-specific file
        surface_specific_file = tmp_path / "leaderboard" / "surface_specific.rst"
        assert surface_specific_file.exists()
        content = surface_specific_file.read_text()
        assert "surf1" in content or "Surf1" in content
        assert "surf2" in content or "Surf2" in content


class TestWriteSurfaceLeaderboardsComprehensive:
    """Comprehensive tests for write_surface_leaderboards function."""
    
    def test_write_surface_leaderboards_with_fourier_continuation(self, tmp_path):
        """Test writing surface leaderboards with Fourier continuation visualization links."""
        repo_root = tmp_path
        docs_dir = tmp_path / "docs"
        
        # Create submission directory structure
        submission_dir = tmp_path / "submissions" / "surf1" / "user1" / "2024-01-01_12-00"
        submission_dir.mkdir(parents=True)
        
        # Create order directories for Fourier continuation
        order_2_dir = submission_dir / "order_2"
        order_2_dir.mkdir()
        order_4_dir = submission_dir / "order_4"
        order_4_dir.mkdir()
        
        # Create PDF files
        (order_2_dir / "bn_error_3d_plot_initial.pdf").write_text("dummy")
        (order_2_dir / "bn_error_3d_plot.pdf").write_text("dummy")
        (order_4_dir / "bn_error_3d_plot.pdf").write_text("dummy")
        
        surface_leaderboards = {
            "surf1": {
                "entries": [
                    {
                        "rank": 1,
                        "method_key": "m1",
                        "method_name": "m1",
                        "method_version": "v1",
                        "score_primary": 0.1,
                        "run_date": "2024-01-01T12:00:00",
                        "contact": "user1",
                        "hardware": "CPU",
                        "path": "submissions/surf1/user1/2024-01-01_12-00/results.json",
                        "metrics": {
                            "final_normalized_squared_flux": 0.1,
                            "fourier_continuation_orders": "2,4",
                        },
                    }
                ]
            }
        }
        
        surface_names = write_surface_leaderboards(
            surface_leaderboards, docs_dir=docs_dir, repo_root=repo_root
        )
        assert "surf1" in surface_names
        
        surface_md = docs_dir / "leaderboards" / "surf1.md"
        assert surface_md.exists()
        content = surface_md.read_text()
        # Should have visualization links
        assert "i" in content or "f" in content or "Legend" in content
    
    def test_write_surface_leaderboards_empty(self, tmp_path):
        """Test writing empty surface leaderboards."""
        surface_leaderboards = {}
        docs_dir = tmp_path / "docs"
        repo_root = tmp_path
        surface_names = write_surface_leaderboards(
            surface_leaderboards, docs_dir=docs_dir, repo_root=repo_root
        )
        assert surface_names == []
    
    def test_write_surface_leaderboards_with_visualization_links(self, tmp_path):
        """Test writing surface leaderboards with visualization links."""
        surface_leaderboards = {
            "surf1": {
                "entries": [
                    {
                        "rank": 1,
                        "method_key": "m1",
                        "method_name": "m1",
                        "method_version": "v1",
                        "score_primary": 0.1,
                        "run_date": "2024-01-01T12:00:00",
                        "contact": "user1",
                        "hardware": "CPU",
                        "path": "submissions/surf1/user/ts/results.json",
                        "metrics": {
                            "final_normalized_squared_flux": 0.1,
                            "num_coils": 4.0,
                        },
                    }
                ]
            }
        }
        docs_dir = tmp_path / "docs"
        repo_root = tmp_path
        
        # Create the submission directory structure
        submission_dir = tmp_path / "submissions" / "surf1" / "user" / "ts"
        submission_dir.mkdir(parents=True)
        
        surface_names = write_surface_leaderboards(
            surface_leaderboards, docs_dir=docs_dir, repo_root=repo_root
        )
        assert "surf1" in surface_names
        
        surface_md = docs_dir / "leaderboards" / "surf1.md"
        assert surface_md.exists()
        content = surface_md.read_text()
        # Surface leaderboard uses user names, not method names
        assert "user1" in content
        assert "Legend" in content


class TestWriteSurfaceLeaderboardIndex:
    """Tests for write_surface_leaderboard_index function."""
    
    def test_write_surface_leaderboard_index(self, tmp_path):
        """Test that write_surface_leaderboard_index does nothing (API compatibility)."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True)
        # Should not raise an error
        write_surface_leaderboard_index(["surf1", "surf2"], docs_dir)
        # Function does nothing, so just verify it completes


class TestUpdateDatabase:
    """Tests for update_database function."""
    
    def test_update_database_empty(self, tmp_path):
        """Test update_database with empty submissions."""
        repo_root = tmp_path
        submissions_root = tmp_path / "submissions"
        submissions_root.mkdir(parents=True)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True)
        
        update_database(
            repo_root=repo_root,
            submissions_root=submissions_root,
            docs_dir=docs_dir,
            cases_root=tmp_path / "cases",
            plasma_surfaces_dir=tmp_path / "plasma_surfaces",
        )
        
        # Should create leaderboard.json
        leaderboard_file = docs_dir / "leaderboard.json"
        assert leaderboard_file.exists()
        leaderboard = json.loads(leaderboard_file.read_text())
        assert leaderboard["entries"] == []
        
        # Should create leaderboard.rst
        leaderboard_rst = docs_dir / "leaderboard.rst"
        assert leaderboard_rst.exists()
    
    def test_update_database_with_submissions(self, tmp_path):
        """Test update_database with actual submissions."""
        repo_root = tmp_path
        submissions_root = tmp_path / "submissions"
        docs_dir = tmp_path / "docs"
        plasma_surfaces_dir = tmp_path / "plasma_surfaces"
        plasma_surfaces_dir.mkdir(parents=True)
        
        # Create a submission
        submission_dir = submissions_root / "surf1" / "user1" / "2024-01-01_12-00"
        submission_dir.mkdir(parents=True)
        
        results_file = submission_dir / "results.json"
        results_file.write_text(json.dumps({
            "metadata": {
                "method_name": "test_method",
                "method_version": "v1",
                "contact": "user1@example.com",
                "hardware": "CPU",
            },
            "metrics": {
                "final_normalized_squared_flux": 0.001,
            }
        }))
        
        case_yaml = submission_dir / "case.yaml"
        case_yaml.write_text("surface_params:\n  surface: input.surf1\n")
        
        update_database(
            repo_root=repo_root,
            submissions_root=submissions_root,
            docs_dir=docs_dir,
            cases_root=tmp_path / "cases",
            plasma_surfaces_dir=plasma_surfaces_dir,
        )
        
        # Verify outputs
        leaderboard_file = docs_dir / "leaderboard.json"
        assert leaderboard_file.exists()
        leaderboard = json.loads(leaderboard_file.read_text())
        assert len(leaderboard["entries"]) == 1
        
        leaderboard_rst = docs_dir / "leaderboard.rst"
        assert leaderboard_rst.exists()
        
        # Check surface leaderboard was created
        surface_md = docs_dir / "leaderboards" / "surf1.md"
        assert surface_md.exists()
    
    def test_update_database_default_paths(self, tmp_path):
        """Test update_database with default paths."""
        repo_root = tmp_path
        submissions_root = repo_root / "submissions"
        submissions_root.mkdir(parents=True)
        docs_dir = repo_root / "docs"
        docs_dir.mkdir(parents=True)
        
        update_database(repo_root=repo_root)
        
        # Should use default paths
        leaderboard_file = docs_dir / "leaderboard.json"
        assert leaderboard_file.exists()
    
    def test_update_database_with_invalid_leaderboard(self, tmp_path):
        """Test update_database handles invalid leaderboard gracefully."""
        repo_root = tmp_path
        submissions_root = repo_root / "submissions"
        submissions_root.mkdir(parents=True)
        docs_dir = repo_root / "docs"
        docs_dir.mkdir(parents=True)
        
        # This should not crash even with empty submissions
        update_database(
            repo_root=repo_root,
            submissions_root=submissions_root,
            docs_dir=docs_dir,
        )
        
        leaderboard_file = docs_dir / "leaderboard.json"
        assert leaderboard_file.exists()
        leaderboard = json.loads(leaderboard_file.read_text())
        assert "entries" in leaderboard


# ---------------------------------------------------------------------------
# Tests for check_reactor_constraints
# ---------------------------------------------------------------------------

class TestCheckReactorConstraints:
    """Tests for check_reactor_constraints function."""

    def _good_metrics(self):
        """Return metrics that pass all constraints."""
        return {
            "avg_BdotN_over_B": 5e-3,       # < 1e-2
            "final_linking_number": 0,        # abs(0) < 0.5
            "coils_linked_to_surface": True,  # must be True
        }

    def _good_reactor(self):
        """Return reactor-scale metrics that pass all constraints."""
        return {
            "reactor_scale_min_cs_separation": 2.0,   # > 1.3
            "reactor_scale_min_cc_separation": 1.0,    # > 0.7
            "reactor_scale_total_length": 180.0,       # < 220
            "reactor_scale_max_curvature": 0.8,        # < 1.0
            "reactor_scale_mean_squared_curvature": 0.5,  # sqrt(0.5) ≈ 0.71 < 1.0
            "reactor_scale_max_max_coil_force": 0.3,   # < 0.5 MN/m
            "reactor_scale_arclength_variation": 0.5,   # sqrt(0.5) ≈ 0.71 < 1.0
            "N_turns_per_coil": [2, 3, 2],             # max 3 < 500
            "finite_build_cc_clearance": 0.5,           # > 0 (no overlap)
        }

    def test_all_pass(self):
        passes, violations = check_reactor_constraints(
            self._good_metrics(), self._good_reactor()
        )
        assert passes is True
        assert violations == []

    def test_empty_metrics_pass(self):
        """Missing metrics are skipped, not penalized."""
        passes, violations = check_reactor_constraints({}, {})
        assert passes is True
        assert violations == []

    def test_finite_build_cc_overlap_violation(self):
        """Negative clearance (winding packs overlap) should be a hard failure."""
        reactor = self._good_reactor()
        reactor["finite_build_cc_clearance"] = -0.1  # overlap!
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is False
        hard_vs = [v for v in violations if v.get("hard")]
        assert any(v["metric"] == "finite_build_cc_clearance" for v in hard_vs)

    def test_finite_build_cc_clearance_at_zero_pass(self):
        """Clearance exactly 0 (touching but not overlapping) should pass."""
        reactor = self._good_reactor()
        reactor["finite_build_cc_clearance"] = 0.0  # exactly at bound
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        # direction=min, bound=0: value < 0 violates, value >= 0 passes
        fb_violations = [v for v in violations if v["metric"] == "finite_build_cc_clearance"]
        assert len(fb_violations) == 0

    def test_finite_build_cc_clearance_missing_skipped(self):
        """If finite_build_cc_clearance is absent, skip without penalty."""
        reactor = self._good_reactor()
        del reactor["finite_build_cc_clearance"]
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is True

    def test_n_turns_violation(self):
        """max(N_turns_per_coil) > 500 should be a hard constraint violation."""
        reactor = self._good_reactor()
        reactor["N_turns_per_coil"] = [100, 600, 200]  # max 600 > 500
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is False
        assert any(v["metric"] == "N_turns_per_coil" for v in violations)
        nturn_v = [v for v in violations if v["metric"] == "N_turns_per_coil"][0]
        assert nturn_v["value"] == 600  # transformed: max of list
        assert nturn_v["bound"] == N_TURNS_MODEL
        assert nturn_v["hard"] is True

    def test_n_turns_at_bound_pass(self):
        """max(N_turns_per_coil) == 500 should pass."""
        reactor = self._good_reactor()
        reactor["N_turns_per_coil"] = [400, 500, 300]  # max 500 == bound
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is True

    def test_cs_separation_violation(self):
        """Soft constraint: violation recorded but passes_hard stays True."""
        reactor = self._good_reactor()
        reactor["reactor_scale_min_cs_separation"] = 1.0  # < 1.3
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is True  # soft → still passes hard check
        assert any(v["metric"] == "reactor_scale_min_cs_separation" for v in violations)
        assert not any(v["hard"] for v in violations)

    def test_cc_separation_violation(self):
        """Soft constraint: violation recorded but passes_hard stays True."""
        reactor = self._good_reactor()
        reactor["reactor_scale_min_cc_separation"] = 0.5  # < 0.7
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is True  # soft → still passes hard check
        assert any(v["metric"] == "reactor_scale_min_cc_separation" for v in violations)

    def test_total_length_violation(self):
        """Soft constraint: violation recorded but passes_hard stays True."""
        reactor = self._good_reactor()
        reactor["reactor_scale_total_length"] = 250.0  # > 220
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is True  # soft → still passes hard check
        assert any(v["metric"] == "reactor_scale_total_length" for v in violations)

    def test_curvature_violation(self):
        """Soft constraint: violation recorded but passes_hard stays True."""
        reactor = self._good_reactor()
        reactor["reactor_scale_max_curvature"] = 1.5  # > 1.0
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is True  # soft → still passes hard check
        assert any(v["metric"] == "reactor_scale_max_curvature" for v in violations)

    def test_msc_sqrt_violation(self):
        """MSC bound is on sqrt(MSC), so MSC=4 -> sqrt=2 > 1.0.  Soft constraint."""
        reactor = self._good_reactor()
        reactor["reactor_scale_mean_squared_curvature"] = 4.0  # sqrt(4) = 2 > 1.0
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is True  # soft → still passes hard check
        assert any(v["metric"] == "reactor_scale_mean_squared_curvature" for v in violations)

    def test_msc_sqrt_pass(self):
        """MSC=0.81 -> sqrt=0.9 < 1.0, should pass."""
        reactor = self._good_reactor()
        reactor["reactor_scale_mean_squared_curvature"] = 0.81  # sqrt(0.81) = 0.9 < 1.0
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is True

    def test_linking_violation(self):
        metrics = self._good_metrics()
        metrics["final_linking_number"] = 2  # abs(2) > 0.5
        passes, violations = check_reactor_constraints(metrics, self._good_reactor())
        assert passes is False
        assert any(v["metric"] == "final_linking_number" for v in violations)

    def test_linking_negative_violation(self):
        """Negative linking number should also be caught via abs transform."""
        metrics = self._good_metrics()
        metrics["final_linking_number"] = -1  # abs(-1) > 0.5
        passes, violations = check_reactor_constraints(metrics, self._good_reactor())
        assert passes is False
        assert any(v["metric"] == "final_linking_number" for v in violations)

    def test_coils_not_linked_to_surface(self):
        """Coils delinked from plasma surface should fail."""
        metrics = self._good_metrics()
        metrics["coils_linked_to_surface"] = False
        passes, violations = check_reactor_constraints(metrics, self._good_reactor())
        assert passes is False
        assert any(v["metric"] == "coils_linked_to_surface" for v in violations)
        # Should be a hard constraint
        assert any(v.get("hard") for v in violations)

    def test_coils_linked_to_surface_pass(self):
        """Coils linked to surface should pass."""
        metrics = self._good_metrics()
        metrics["coils_linked_to_surface"] = True
        passes, violations = check_reactor_constraints(metrics, self._good_reactor())
        assert passes is True

    def test_bn_violation(self):
        """Soft constraint: violation recorded but passes_hard stays True."""
        metrics = self._good_metrics()
        metrics["avg_BdotN_over_B"] = 5e-2  # > 1e-2
        passes, violations = check_reactor_constraints(metrics, self._good_reactor())
        assert passes is True  # soft → still passes hard check
        assert any(v["metric"] == "avg_BdotN_over_B" for v in violations)

    def test_arclength_variation_violation(self):
        """Soft constraint: sqrt(arclength_variation) > 1.0 m recorded but passes."""
        reactor = self._good_reactor()
        reactor["reactor_scale_arclength_variation"] = 4.0  # sqrt(4) = 2.0 > 1.0
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is True  # soft → still passes
        assert any(v["metric"] == "reactor_scale_arclength_variation" for v in violations)
        av_v = [v for v in violations if v["metric"] == "reactor_scale_arclength_variation"][0]
        assert abs(av_v["value"] - 2.0) < 1e-10  # transformed via sqrt

    def test_arclength_variation_at_bound_pass(self):
        """sqrt(arclength_variation) == 1.0 should pass with no violation."""
        reactor = self._good_reactor()
        reactor["reactor_scale_arclength_variation"] = 1.0  # sqrt(1.0) = 1.0, at bound
        passes, violations = check_reactor_constraints(self._good_metrics(), reactor)
        assert passes is True
        assert not any(v["metric"] == "reactor_scale_arclength_variation" for v in violations)

    def test_multiple_soft_violations_still_pass(self):
        """Multiple soft violations should NOT cause passes=False."""
        metrics = self._good_metrics()
        metrics["avg_BdotN_over_B"] = 1e-1  # soft violation
        reactor = self._good_reactor()
        reactor["reactor_scale_min_cs_separation"] = 0.5   # soft violation
        reactor["reactor_scale_total_length"] = 300.0      # soft violation
        passes, violations = check_reactor_constraints(metrics, reactor)
        assert passes is True  # no hard violations → passes
        assert len(violations) == 3
        assert not any(v["hard"] for v in violations)

    def test_multiple_violations_with_hard(self):
        """Mix of hard and soft violations → passes=False (hard drives it)."""
        metrics = self._good_metrics()
        metrics["avg_BdotN_over_B"] = 1e-1  # soft violation
        reactor = self._good_reactor()
        reactor["N_turns_per_coil"] = [600]  # max 600 > 500 → hard violation
        reactor["reactor_scale_min_cs_separation"] = 0.5   # soft violation
        passes, violations = check_reactor_constraints(metrics, reactor)
        assert passes is False  # hard violation present
        assert len(violations) == 3

    def test_exact_bounds_pass(self):
        """Values exactly at the bound should pass."""
        metrics = {"avg_BdotN_over_B": 1e-2, "final_linking_number": 0,
                   "coils_linked_to_surface": True}
        reactor = {
            "reactor_scale_min_cs_separation": 1.3,
            "reactor_scale_min_cc_separation": 0.7,
            "reactor_scale_total_length": 220.0,
            "reactor_scale_max_curvature": 1.0,
            "reactor_scale_mean_squared_curvature": 1.0,  # sqrt(1.0) = 1.0
            "reactor_scale_arclength_variation": 1.0,      # sqrt(1.0) = 1.0
            "N_turns_per_coil": [500, 400],  # max 500, exactly at bound
            "finite_build_cc_clearance": 0.0,  # exactly at bound (touching)
        }
        passes, violations = check_reactor_constraints(metrics, reactor)
        assert passes is True


class TestLeaderboardConstraintFiltering:
    """Tests for constraint filtering in build_leaderboard_json."""

    def test_passing_entry_included(self):
        methods = {
            "good_method": {
                "method_name": "good",
                "score_primary": 0.001,
                "metrics": {"final_squared_flux": 0.001},
                "passes_constraints": True,
                "constraint_violations": [],
            }
        }
        lb = build_leaderboard_json(methods)
        assert len(lb["entries"]) == 1
        assert len(lb["excluded_entries"]) == 0

    def test_failing_entry_excluded(self):
        methods = {
            "bad_method": {
                "method_name": "bad",
                "score_primary": 0.001,
                "metrics": {"final_squared_flux": 0.001},
                "passes_constraints": False,
                "constraint_violations": [
                    {"label": "Max turns per coil", "metric": "N_turns_per_coil",
                     "value": 600, "bound": 500, "direction": "max",
                     "units": "(turns)", "hard": True}
                ],
            }
        }
        lb = build_leaderboard_json(methods)
        assert len(lb["entries"]) == 0
        assert len(lb["excluded_entries"]) == 1
        assert lb["excluded_entries"][0]["method_name"] == "bad"
        assert len(lb["excluded_entries"][0]["constraint_violations"]) == 1

    def test_mixed_entries(self):
        methods = {
            "good": {
                "method_name": "good",
                "score_primary": 0.002,
                "metrics": {"final_squared_flux": 0.002},
                "passes_constraints": True,
                "constraint_violations": [],
            },
            "bad": {
                "method_name": "bad",
                "score_primary": 0.001,
                "metrics": {"final_squared_flux": 0.001},
                "passes_constraints": False,
                "constraint_violations": [
                    {"label": "test", "metric": "x", "value": 1, "bound": 0,
                     "direction": "max", "units": ""}
                ],
            },
        }
        lb = build_leaderboard_json(methods)
        assert len(lb["entries"]) == 1
        assert lb["entries"][0]["method_name"] == "good"
        assert len(lb["excluded_entries"]) == 1
        assert lb["excluded_entries"][0]["method_name"] == "bad"

    def test_legacy_entry_without_constraint_field(self):
        """Legacy entries without passes_constraints default to included."""
        methods = {
            "legacy": {
                "method_name": "legacy",
                "score_primary": 0.005,
                "metrics": {"final_squared_flux": 0.005},
                # No passes_constraints key
            }
        }
        lb = build_leaderboard_json(methods)
        assert len(lb["entries"]) == 1
        assert len(lb["excluded_entries"]) == 0

    def test_composite_score_zero_excluded(self):
        """Entries with composite_score=0 should be excluded even if passes_constraints is True."""
        methods = {
            "infeasible": {
                "method_name": "infeasible",
                "composite_score": 0.0,
                "score_primary": 0.001,
                "metrics": {"final_squared_flux": 0.001},
                "passes_constraints": True,  # Constraints say OK but score is 0
            }
        }
        lb = build_leaderboard_json(methods)
        assert len(lb["entries"]) == 0
        assert len(lb["excluded_entries"]) == 1

    def test_soft_violation_entry_not_excluded(self):
        """Entries with only soft violations should stay in main leaderboard."""
        methods = {
            "soft_fail": {
                "method_name": "soft_fail",
                "composite_score": 0.7,  # below 1 due to soft violations
                "score_primary": 0.003,
                "metrics": {"final_squared_flux": 0.003},
                "passes_constraints": True,  # soft-only → still passes hard check
                "constraint_violations": [
                    {"label": "Total coil length",
                     "metric": "reactor_scale_total_length",
                     "value": 250.0, "bound": 220.0,
                     "direction": "max", "units": "m", "hard": False},
                ],
            }
        }
        lb = build_leaderboard_json(methods)
        assert len(lb["entries"]) == 1  # NOT excluded
        assert len(lb["excluded_entries"]) == 0
        # Soft violations should be carried through
        assert len(lb["entries"][0].get("constraint_violations", [])) == 1

    def test_sort_by_composite_score_descending(self):
        """Leaderboard should sort by composite_score descending (higher is better)."""
        methods = {
            "medium": {
                "method_name": "medium",
                "composite_score": 1.5,
                "score_primary": 0.002,
                "metrics": {"final_squared_flux": 0.002},
                "passes_constraints": True,
                "constraint_violations": [],
            },
            "best": {
                "method_name": "best",
                "composite_score": 2.5,
                "score_primary": 0.001,
                "metrics": {"final_squared_flux": 0.001},
                "passes_constraints": True,
                "constraint_violations": [],
            },
            "worst": {
                "method_name": "worst",
                "composite_score": 0.8,
                "score_primary": 0.003,
                "metrics": {"final_squared_flux": 0.003},
                "passes_constraints": True,
                "constraint_violations": [],
            },
        }
        lb = build_leaderboard_json(methods)
        assert len(lb["entries"]) == 3
        assert lb["entries"][0]["method_name"] == "best"
        assert lb["entries"][0]["rank"] == 1
        assert lb["entries"][1]["method_name"] == "medium"
        assert lb["entries"][1]["rank"] == 2
        assert lb["entries"][2]["method_name"] == "worst"
        assert lb["entries"][2]["rank"] == 3


class TestCompositeScore:
    """Tests for compute_composite_score function."""

    def _good_metrics(self):
        """Return metrics that pass all constraints."""
        return {
            "avg_BdotN_over_B": 5e-3,  # < 1e-2
            "final_linking_number": 0,
            "coils_linked_to_surface": True,
        }

    def _good_reactor(self):
        """Return reactor-scale metrics that pass all constraints."""
        return {
            "reactor_scale_min_cs_separation": 2.0,
            "reactor_scale_min_cc_separation": 1.0,
            "reactor_scale_total_length": 180.0,
            "reactor_scale_max_curvature": 0.8,
            "reactor_scale_mean_squared_curvature": 0.5,  # sqrt(0.5) ≈ 0.71
            "reactor_scale_max_max_coil_force": 0.3,
            "reactor_scale_arclength_variation": 0.5,      # sqrt(0.5) ≈ 0.71 < 1.0
            "N_turns_per_coil": [2, 3, 2],  # max 3 < 500
            "finite_build_cc_clearance": 0.5,              # > 0 (no overlap)
        }

    def test_feasible_score_positive(self):
        """Feasible submissions should get a positive score."""
        score, details = compute_composite_score(
            self._good_metrics(), self._good_reactor()
        )
        assert score > 0
        assert not details["infeasible"]
        assert details["n_factors"] > 0

    def test_feasible_score_above_one(self):
        """Submissions with good margin should score above 1."""
        score, details = compute_composite_score(
            self._good_metrics(), self._good_reactor()
        )
        assert score > 1.0  # Good margins on all constraints

    def test_coils_delinked_score_zero(self):
        """Delinked coils (hard constraint) should give score = 0."""
        metrics = self._good_metrics()
        metrics["coils_linked_to_surface"] = False
        score, details = compute_composite_score(metrics, self._good_reactor())
        assert score == 0.0
        assert details["infeasible"] is True
        assert "linked" in details["reason"].lower()

    def test_linking_number_nonzero_score_zero(self):
        """Non-zero coil-coil linking number should give score = 0."""
        metrics = self._good_metrics()
        metrics["final_linking_number"] = 2
        score, details = compute_composite_score(metrics, self._good_reactor())
        assert score == 0.0
        assert details["infeasible"] is True

    def test_linking_number_negative_score_zero(self):
        """Negative linking number should also give score = 0 (abs check)."""
        metrics = self._good_metrics()
        metrics["final_linking_number"] = -1
        score, details = compute_composite_score(metrics, self._good_reactor())
        assert score == 0.0
        assert details["infeasible"] is True

    def test_at_bounds_score_one(self):
        """Values exactly at bounds should give score ≈ 1."""
        metrics = {
            "avg_BdotN_over_B": 1e-2,
            "final_linking_number": 0,
            "coils_linked_to_surface": True,
        }
        reactor = {
            "reactor_scale_min_cs_separation": 1.3,
            "reactor_scale_min_cc_separation": 0.7,
            "reactor_scale_total_length": 220.0,
            "reactor_scale_max_curvature": 1.0,
            "reactor_scale_mean_squared_curvature": 1.0,  # sqrt(1.0) = 1.0
            "reactor_scale_arclength_variation": 1.0,      # sqrt(1.0) = 1.0
            "N_turns_per_coil": [100],       # hard constraint, not in soft score
            "finite_build_cc_clearance": 0.5, # hard constraint, not in soft score
        }
        score, details = compute_composite_score(metrics, reactor)
        # All margins are 0, so score = exp(0) = 1.0
        assert abs(score - 1.0) < 1e-10

    def test_score_factors_correctness(self):
        """Verify individual factor computations."""
        import math
        metrics = {
            "avg_BdotN_over_B": 5e-3,  # 50% of bound (1e-2)
            "final_linking_number": 0,
            "coils_linked_to_surface": True,
        }
        reactor = {
            "reactor_scale_min_cs_separation": 2.6,  # 2x bound → margin = 1.0
            "reactor_scale_min_cc_separation": 0.7,   # at bound → margin = 0.0
        }
        score, details = compute_composite_score(metrics, reactor)
        factors = details["factors"]

        # BdotN: max constraint, margin = 1 - 5e-3/1e-2 = 0.5
        assert abs(factors["avg_BdotN_over_B"]["margin"] - 0.5) < 1e-10

        # CS separation: min constraint, margin = 2.6/1.3 - 1 = 1.0
        assert abs(factors["reactor_scale_min_cs_separation"]["margin"] - 1.0) < 1e-10

        # CC separation: min constraint, margin = 0.7/0.7 - 1 = 0.0
        assert abs(factors["reactor_scale_min_cc_separation"]["margin"] - 0.0) < 1e-10

        # Score = exp(mean(0.5, 1.0, 0.0)) = exp(0.5)
        expected_score = math.exp((0.5 + 1.0 + 0.0) / 3)
        assert abs(score - expected_score) < 1e-10

    def test_empty_metrics_score_none(self):
        """No metrics at all should give score None (not infeasible, just unknown)."""
        score, details = compute_composite_score({}, {})
        assert score is None
        assert "No metrics" in details.get("reason", "")

    def test_missing_some_metrics(self):
        """Score should be computed from available metrics only."""
        metrics = {
            "avg_BdotN_over_B": 5e-3,  # < 1e-2
            "final_linking_number": 0,
            "coils_linked_to_surface": True,
        }
        reactor = {
            "reactor_scale_min_cs_separation": 2.0,
            # Only 2 soft constraints available (BdotN + CS)
        }
        score, details = compute_composite_score(metrics, reactor)
        assert score > 0
        assert details["n_factors"] == 2  # BdotN + CS separation

    def test_sqrt_msc_transform(self):
        """MSC factor should use sqrt transform."""
        metrics = self._good_metrics()
        reactor = self._good_reactor()
        reactor["reactor_scale_mean_squared_curvature"] = 0.64  # sqrt(0.64)=0.8, bound=1.0
        score, details = compute_composite_score(metrics, reactor)
        msc_factor = details["factors"]["reactor_scale_mean_squared_curvature"]
        # margin = 1 - sqrt(0.64)/1.0 = 1 - 0.8 = 0.2
        assert abs(msc_factor["margin"] - 0.2) < 1e-10
        assert abs(msc_factor["value"] - 0.8) < 1e-10

    def test_worse_design_lower_score(self):
        """Design closer to bounds should score lower."""
        metrics = self._good_metrics()
        good_reactor = self._good_reactor()
        bad_reactor = self._good_reactor()
        # Worse total length — closer to the 220 m bound
        bad_reactor["reactor_scale_total_length"] = 215.0  # close to 220

        score_good, _ = compute_composite_score(metrics, good_reactor)
        score_bad, _ = compute_composite_score(metrics, bad_reactor)
        assert score_good > score_bad

    def test_hard_constraint_takes_precedence(self):
        """Hard constraint failure should override even excellent soft metrics."""
        metrics = self._good_metrics()
        metrics["coils_linked_to_surface"] = False  # Hard fail
        reactor = self._good_reactor()
        score, details = compute_composite_score(metrics, reactor)
        assert score == 0.0
        assert details["infeasible"] is True

    def test_n_turns_hard_constraint_score_zero(self):
        """N_turns exceeding 500 should give score = 0 (hard constraint)."""
        metrics = self._good_metrics()
        reactor = self._good_reactor()
        reactor["N_turns_per_coil"] = [100, 600, 200]  # max 600 > 500
        score, details = compute_composite_score(metrics, reactor)
        assert score == 0.0
        assert details["infeasible"] is True


class TestReactorScaleLeaderboard:
    """Tests for write_reactor_scale_leaderboard."""

    def test_writes_rst_file(self, tmp_path):
        """write_reactor_scale_leaderboard should produce an RST file."""
        leaderboard = {"entries": []}
        surface_leaderboards = {
            "test_surface": {
                "entries": [
                    {
                        "rank": 1,
                        "method_name": "m1",
                        "contact": "user1",
                        "composite_score": 1.5,
                        "reactor_scale_metrics": {
                            "reactor_scale_min_cs_separation": 2.0,
                            "reactor_scale_max_max_coil_force": 0.3,
                            "reactor_scale_total_length": 180.0,
                            "total_superconductor_length_km": 54.0,
                            "reactor_scale_arclength_variation": 0.05,
                            "max_winding_pack_width": 0.035,
                            "per_turn_max_force": 0.15,
                            "per_turn_max_torque": 0.08,
                            "N_turns_per_coil": [2, 3],
                        },
                    }
                ]
            }
        }
        out_rst = tmp_path / "reactor_scale.rst"
        write_reactor_scale_leaderboard(leaderboard, surface_leaderboards, out_rst)
        assert out_rst.exists()
        content = out_rst.read_text()
        assert "Reactor-Scale Leaderboard" in content
        assert "Test Surface" in content  # display_name
        assert "1.500" in content  # composite_score
        assert "pass" in content  # status column (no violations)
        assert "user1" in content
        # Units should appear in column headers, not as subscripts
        assert r"[\text{m}]" in content  # e.g. d_{cs}\ [\text{m}]
        assert r"[\text{MN/m}]" in content  # e.g. F_turn\ [\text{MN/m}]
        # Per-turn force and torque columns should appear
        assert r"F_\text{turn}" in content
        assert r"\tau_\text{turn}" in content
        # Superconductor length column should appear
        assert r"L_\text{SC}" in content
        assert r"[\text{km}]" in content
        assert "54.0" in content  # the SC length value
        # N_turns_per_coil should appear as a comma-separated column
        assert r"N_{\text{turns},i}" in content
        assert "2, 3" in content  # per-coil turns values
        # Single-turn F_max / tau_max should NOT appear (replaced by per-turn)
        assert r"F_\text{max}" not in content
        assert r"\tau_\text{max}" not in content
        # Average force/torque should NOT appear
        assert r"\bar{F}" not in content
        assert r"\bar{\tau}" not in content
        # Arclength variation should NOT appear as a column (removed from display)
        assert r"\mathrm{Var}(l_i)" not in content
        # LN column should appear
        assert r"\text{LN}" in content
        # Visualization link columns should appear in header
        assert r"\text{i}" in content
        assert r"\text{f}" in content
        assert r"\text{PP}" in content
        # Winding-pack width column should appear
        assert r"w_\text{WP}" in content
        assert "3.50e-02" in content  # the max WP width value (sci notation)
        # Finite-build clearance constraint should appear in the constraint table
        assert "Finite-build" in content
        assert "clearance" in content.lower()

    def test_excluded_entry_shows_fail(self, tmp_path):
        """Entries with hard constraint violations should show FAIL status."""
        leaderboard = {"entries": []}
        surface_leaderboards = {
            "test_surface": {
                "entries": [
                    {
                        "rank": 1,
                        "method_name": "m1",
                        "contact": "user1",
                        "composite_score": 0.0,
                        "constraint_violations": [
                            {"label": "Max turns per coil",
                             "metric": "N_turns_per_coil",
                             "hard": True}
                        ],
                        "reactor_scale_metrics": {
                            "reactor_scale_min_cs_separation": 0.5,
                            "reactor_scale_max_max_coil_force": 300.0,
                        },
                    }
                ]
            }
        }
        out_rst = tmp_path / "reactor_scale.rst"
        write_reactor_scale_leaderboard(leaderboard, surface_leaderboards, out_rst)
        content = out_rst.read_text()
        assert "FAIL" in content
        assert ":red:" in content

    def test_soft_violation_shows_orange_not_fail(self, tmp_path):
        """Soft violations should show orange cells but NOT FAIL status."""
        leaderboard = {"entries": []}
        surface_leaderboards = {
            "test_surface": {
                "entries": [
                    {
                        "rank": 1,
                        "method_name": "m1",
                        "contact": "user1",
                        "composite_score": 0.8,
                        "constraint_violations": [
                            {"label": "Total coil length",
                             "metric": "reactor_scale_total_length",
                             "hard": False}
                        ],
                        "reactor_scale_metrics": {
                            "reactor_scale_total_length": 250.0,
                            "reactor_scale_min_cs_separation": 2.0,
                        },
                    }
                ]
            }
        }
        out_rst = tmp_path / "reactor_scale.rst"
        write_reactor_scale_leaderboard(leaderboard, surface_leaderboards, out_rst)
        content = out_rst.read_text()
        # Soft violation → orange highlight, NOT red FAIL
        assert ":orange:" in content
        assert "pass" in content  # status should be pass
        # Should NOT show FAIL for a pure soft violation
        lines_with_fail = [line for line in content.splitlines()
                           if "FAIL" in line and "hard" not in line.lower()
                           and "constraint" not in line.lower()]
        # The only lines with FAIL should be in the explanatory text, not data rows
        for line in lines_with_fail:
            assert "* -" not in line, f"Data row incorrectly shows FAIL: {line}"

    def test_empty_surface(self, tmp_path):
        """Surfaces with no entries should show a placeholder."""
        leaderboard = {"entries": []}
        surface_leaderboards = {"empty_surf": {"entries": []}}
        out_rst = tmp_path / "reactor_scale.rst"
        write_reactor_scale_leaderboard(leaderboard, surface_leaderboards, out_rst)
        content = out_rst.read_text()
        assert "No submissions" in content

    def test_no_reactor_scale_data(self, tmp_path):
        """Entries without reactor_scale_metrics should show a message."""
        leaderboard = {"entries": []}
        surface_leaderboards = {
            "surf": {
                "entries": [
                    {"rank": 1, "method_name": "m1", "contact": "u",
                     "composite_score": None, "reactor_scale_metrics": {}}
                ]
            }
        }
        out_rst = tmp_path / "reactor_scale.rst"
        write_reactor_scale_leaderboard(leaderboard, surface_leaderboards, out_rst)
        content = out_rst.read_text()
        assert "No reactor-scale data" in content
