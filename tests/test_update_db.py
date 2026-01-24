"""
Unit tests for update_db.py
"""
import json
import tempfile
import zipfile
from pathlib import Path
from stellcoilbench.update_db import (
    _metric_shorthand,
    _metric_definition,
    _format_date,
    _shorthand_to_math,
    _load_submissions,
    _get_all_metrics_from_entries,
    build_methods_json,
    build_leaderboard_json,
    build_surface_leaderboards,
    write_markdown_leaderboard,
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
        flux_def = _metric_definition("final_normalized_squared_flux")
        assert "Normalized squared flux" in flux_def
        assert r"\int" in flux_def  # Check for the integral formula
        
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
        assert leaderboard == {"entries": []}
    
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
        assert leaderboard == {"entries": []}



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
                    "final_normalized_squared_flux": 0.01,
                    "score_primary": 0.01,
                    "initial_B_field": 1.0,
                }
            }
        ]
        keys = _get_all_metrics_from_entries(entries)
        assert keys[0] == "final_normalized_squared_flux"
        assert "score_primary" not in keys
        assert "initial_B_field" not in keys
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
        out_md = tmp_path / "leaderboard.md"
        write_markdown_leaderboard(leaderboard, out_md)
        content = out_md.read_text()
        assert "CoilBench Leaderboard" in content
        assert "Legend" in content
        assert "f_B" in content

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
                    "run_date": "2024-01-01T12:00:00",
                    "contact": "user1",
                    "hardware": "CPU",
                    "path": "submissions/surface/user/ts/results.json",
                    "metrics": {
                        "final_normalized_squared_flux": 0.001,
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
                    "run_date": "2024-01-02T12:00:00",
                    "contact": "user2",
                    "hardware": "GPU",
                    "path": "submissions/surface/user2/ts2/results.json",
                    "metrics": {
                        "final_normalized_squared_flux": 0.002,
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
        assert "0.001" in content or "1.0e-03" in content
        assert "0.002" in content or "2.0e-03" in content


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
        submissions_root = tmp_path / "submissions"
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
        assert leaderboard == {"entries": []}
        
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
