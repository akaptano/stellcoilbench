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
    _load_submissions,
    _get_all_metrics_from_entries,
    build_methods_json,
    build_leaderboard_json,
    build_surface_leaderboards,
    write_markdown_leaderboard,
    write_rst_leaderboard,
    write_surface_leaderboards,
)


class TestMetricShorthand:
    """Tests for _metric_shorthand function."""
    
    def test_known_metrics(self):
        """Test shorthand for known metrics."""
        assert _metric_shorthand("final_normalized_squared_flux") == "f_B"
        assert _metric_shorthand("max_BdotN_over_B") == "max⟨Bn⟩/⟨B⟩"
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
            zip_path = submissions_root / "submissions" / "surface1" / "user1" / "2024-01-01_12-00.zip"
            zip_path.parent.mkdir(parents=True)
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
            
            methods = build_methods_json(submissions_root, repo_root)
            assert len(methods) == 1
            
            method_key = "test_method:surface1:user1:2024-01-01_12-00"
            assert method_key in methods
            method_data = methods[method_key]
            assert method_data["method_name"] == "test_method"
            assert method_data["contact"] == "test@example.com"
            assert method_data["metrics"]["final_normalized_squared_flux"] == 0.001
            assert method_data["score_primary"] == 0.001
    
    def test_build_methods_json_extracts_coil_params(self):
        """Test that coil parameters are extracted from case.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir).resolve()
            repo_root = Path(tmpdir).resolve()
            
            submission_dir = submissions_root / "surface1" / "user1" / "2024-01-01_12-00"
            submission_dir.mkdir(parents=True)
            
            results_file = submission_dir / "results.json"
            results_file.write_text(json.dumps({
                "metadata": {"method_name": "test_method"},
                "metrics": {"final_normalized_squared_flux": 0.001}
            }))
            
            # Create case.yaml with coil parameters
            case_yaml = submission_dir / "case.yaml"
            case_yaml.write_text("""coils_params:
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

            methods = build_methods_json(submissions_root, repo_root)
            method_key = "method1:surface1:user1:2024-01-01_12-00"
            assert methods[method_key]["score_primary"] == 0.5

    def test_build_methods_json_non_numeric_fallback_keeps_none(self):
        """Test non-numeric fallback score leaves score_primary as None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_root = Path(tmpdir).resolve()
            repo_root = Path(tmpdir).resolve()
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
        surface_leaderboards = build_surface_leaderboards(
            leaderboard, submissions_root=tmp_path, plasma_surfaces_dir=tmp_path
        )
        out_rst = tmp_path / "leaderboard.rst"
        write_rst_leaderboard(leaderboard, out_rst, surface_leaderboards)
        content = out_rst.read_text()
        assert "StellCoilBench Leaderboard" in content
        assert "Surface-Specific Leaderboards" in content
        assert ".. list-table::" in content
        assert "Metric Definitions" in content

    def test_build_surface_leaderboards_and_write(self, tmp_path):
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

        surface_leaderboards = build_surface_leaderboards(
            leaderboard, submissions_root=tmp_path, plasma_surfaces_dir=tmp_path
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
