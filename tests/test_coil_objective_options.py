"""
Unit tests for coil objective terms options and threshold determination.

Tests verify that coil_objective_terms options correctly determine thresholds
for Jccdist, Jcsdist, Jforce, and Jtorque objects.
"""
import pytest
from stellcoilbench.evaluate import load_case_config


def _determine_thresholds(coil_objective_terms, cc_threshold, cs_threshold, force_threshold, torque_threshold):
    """
    Extract threshold determination logic from optimize_coils_loop for testing.
    
    This mirrors the logic in coil_optimization.py lines 448-484.
    """
    # Default to using thresholds (for backward compatibility with default behavior)
    cc_thresh = cc_threshold
    cs_thresh = cs_threshold
    force_thresh = force_threshold
    torque_thresh = torque_threshold
    
    # Check if l1 (no threshold) or l1_threshold is specified
    # Only adjust thresholds if the term is explicitly specified in coil_objective_terms
    if coil_objective_terms:
        coil_coil_dist_option = coil_objective_terms.get("coil_coil_distance")
        if coil_coil_dist_option == "l1":
            cc_thresh = 0.0
        elif coil_coil_dist_option == "l1_threshold":
            cc_thresh = cc_threshold
        
        coil_surf_dist_option = coil_objective_terms.get("coil_surface_distance")
        if coil_surf_dist_option == "l1":
            cs_thresh = 0.0
        elif coil_surf_dist_option == "l1_threshold":
            cs_thresh = cs_threshold
        
        coil_force_option = coil_objective_terms.get("coil_coil_force")
        if coil_force_option == "lp":
            force_thresh = 0.0
        elif coil_force_option == "lp_threshold":
            force_thresh = force_threshold
        
        coil_torque_option = coil_objective_terms.get("coil_coil_torque")
        if coil_torque_option == "lp":
            torque_thresh = 0.0
        elif coil_torque_option == "lp_threshold":
            torque_thresh = torque_threshold
    
    return cc_thresh, cs_thresh, force_thresh, torque_thresh


class TestThresholdDetermination:
    """Test threshold determination logic for coil objective terms."""
    
    def test_default_thresholds_when_none_specified(self):
        """Test that default thresholds are used when coil_objective_terms is None."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            None, cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 1.0
        assert cs_thresh == 1.5
        assert force_thresh == 2.0
        assert torque_thresh == 2.5
    
    def test_default_thresholds_when_empty_dict(self):
        """Test that default thresholds are used when coil_objective_terms is empty."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {}, cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 1.0
        assert cs_thresh == 1.5
        assert force_thresh == 2.0
        assert torque_thresh == 2.5
    
    def test_coil_coil_distance_l1_zero_threshold(self):
        """Test that l1 option sets cc_threshold to 0.0."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {"coil_coil_distance": "l1"},
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 0.0
        assert cs_thresh == 1.5  # Unchanged
        assert force_thresh == 2.0  # Unchanged
        assert torque_thresh == 2.5  # Unchanged
    
    def test_coil_coil_distance_l1_threshold_uses_threshold(self):
        """Test that l1_threshold option uses actual threshold."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {"coil_coil_distance": "l1_threshold"},
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 1.0
        assert cs_thresh == 1.5  # Unchanged
        assert force_thresh == 2.0  # Unchanged
        assert torque_thresh == 2.5  # Unchanged
    
    def test_coil_surface_distance_l1_zero_threshold(self):
        """Test that l1 option sets cs_threshold to 0.0."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {"coil_surface_distance": "l1"},
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 1.0  # Unchanged
        assert cs_thresh == 0.0
        assert force_thresh == 2.0  # Unchanged
        assert torque_thresh == 2.5  # Unchanged
    
    def test_coil_surface_distance_l1_threshold_uses_threshold(self):
        """Test that l1_threshold option uses actual threshold."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {"coil_surface_distance": "l1_threshold"},
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 1.0  # Unchanged
        assert cs_thresh == 1.5
        assert force_thresh == 2.0  # Unchanged
        assert torque_thresh == 2.5  # Unchanged
    
    def test_coil_coil_force_lp_zero_threshold(self):
        """Test that lp option sets force_threshold to 0.0."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {"coil_coil_force": "lp"},
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 1.0  # Unchanged
        assert cs_thresh == 1.5  # Unchanged
        assert force_thresh == 0.0
        assert torque_thresh == 2.5  # Unchanged
    
    def test_coil_coil_force_lp_threshold_uses_threshold(self):
        """Test that lp_threshold option uses actual threshold."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {"coil_coil_force": "lp_threshold"},
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 1.0  # Unchanged
        assert cs_thresh == 1.5  # Unchanged
        assert force_thresh == 2.0
        assert torque_thresh == 2.5  # Unchanged
    
    def test_coil_coil_torque_lp_zero_threshold(self):
        """Test that lp option sets torque_threshold to 0.0."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {"coil_coil_torque": "lp"},
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 1.0  # Unchanged
        assert cs_thresh == 1.5  # Unchanged
        assert force_thresh == 2.0  # Unchanged
        assert torque_thresh == 0.0
    
    def test_coil_coil_torque_lp_threshold_uses_threshold(self):
        """Test that lp_threshold option uses actual threshold."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {"coil_coil_torque": "lp_threshold"},
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 1.0  # Unchanged
        assert cs_thresh == 1.5  # Unchanged
        assert force_thresh == 2.0  # Unchanged
        assert torque_thresh == 2.5
    
    def test_all_l1_options_together(self):
        """Test all l1 options together set all thresholds to 0.0."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {
                "coil_coil_distance": "l1",
                "coil_surface_distance": "l1",
                "coil_coil_force": "lp",
                "coil_coil_torque": "lp",
            },
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 0.0
        assert cs_thresh == 0.0
        assert force_thresh == 0.0
        assert torque_thresh == 0.0
    
    def test_all_threshold_options_together(self):
        """Test all threshold options together use actual thresholds."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {
                "coil_coil_distance": "l1_threshold",
                "coil_surface_distance": "l1_threshold",
                "coil_coil_force": "lp_threshold",
                "coil_coil_torque": "lp_threshold",
            },
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 1.0
        assert cs_thresh == 1.5
        assert force_thresh == 2.0
        assert torque_thresh == 2.5
    
    def test_mixed_options(self):
        """Test mixed l1 and threshold options."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {
                "coil_coil_distance": "l1",  # Zero threshold
                "coil_surface_distance": "l1_threshold",  # Use threshold
                "coil_coil_force": "lp",  # Zero threshold
                "coil_coil_torque": "lp_threshold",  # Use threshold
            },
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        assert cc_thresh == 0.0
        assert cs_thresh == 1.5
        assert force_thresh == 0.0
        assert torque_thresh == 2.5
    
    def test_other_options_dont_affect_thresholds(self):
        """Test that other options (like l2) don't affect thresholds."""
        cc_thresh, cs_thresh, force_thresh, torque_thresh = _determine_thresholds(
            {
                "coil_coil_distance": "l2",  # Not l1 or l1_threshold
                "coil_surface_distance": "l2_threshold",  # Not l1 or l1_threshold
                "total_length": "l2_threshold",  # Different term
            },
            cc_threshold=1.0, cs_threshold=1.5, force_threshold=2.0, torque_threshold=2.5
        )
        # Should keep default thresholds since l1/l1_threshold not specified
        assert cc_thresh == 1.0
        assert cs_thresh == 1.5
        assert force_thresh == 2.0
        assert torque_thresh == 2.5


class TestCoilObjectiveTermsLoading:
    """Test that case.yaml files with different coil_objective_terms options load correctly."""
    
    def test_load_case_without_coil_objective_terms(self, tmp_path):
        """Test loading case.yaml without coil_objective_terms (should be None, no constraints included)."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        
        config = load_case_config(case_yaml)
        assert config.coil_objective_terms is None
    
    def test_load_case_with_empty_coil_objective_terms(self, tmp_path):
        """Test loading case.yaml with empty coil_objective_terms dict (no constraints included)."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
coil_objective_terms: {}
""")
        
        config = load_case_config(case_yaml)
        assert config.coil_objective_terms == {}
    
    def test_load_case_with_l1_options(self, tmp_path):
        """Test loading case.yaml with l1 options."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
coil_objective_terms:
  coil_coil_distance: l1
  coil_surface_distance: l1
  coil_coil_force: lp
  coil_coil_torque: lp
""")
        
        config = load_case_config(case_yaml)
        assert config.coil_objective_terms["coil_coil_distance"] == "l1"
        assert config.coil_objective_terms["coil_surface_distance"] == "l1"
        assert config.coil_objective_terms["coil_coil_force"] == "lp"
        assert config.coil_objective_terms["coil_coil_torque"] == "lp"
    
    def test_load_case_with_l1_threshold_options(self, tmp_path):
        """Test loading case.yaml with l1_threshold options."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
coil_objective_terms:
  coil_coil_distance: l1_threshold
  coil_surface_distance: l1_threshold
  coil_coil_force: lp_threshold
  coil_coil_torque: lp_threshold
""")
        
        config = load_case_config(case_yaml)
        assert config.coil_objective_terms["coil_coil_distance"] == "l1_threshold"
        assert config.coil_objective_terms["coil_surface_distance"] == "l1_threshold"
        assert config.coil_objective_terms["coil_coil_force"] == "lp_threshold"
        assert config.coil_objective_terms["coil_coil_torque"] == "lp_threshold"
    
    def test_load_case_with_all_options(self, tmp_path):
        """Test loading case.yaml with all coil objective term options."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
coil_objective_terms:
  total_length: l2_threshold
  coil_coil_distance: l1
  coil_surface_distance: l1_threshold
  coil_curvature: lp_threshold
  coil_curvature_p: 2
  coil_mean_squared_curvature: l2
  linking_number: l2
  coil_coil_force: lp
  coil_coil_force_p: 2.5
  coil_coil_torque: lp_threshold
  coil_coil_torque_p: 3.0
""")
        
        config = load_case_config(case_yaml)
        assert config.coil_objective_terms["total_length"] == "l2_threshold"
        assert config.coil_objective_terms["coil_coil_distance"] == "l1"
        assert config.coil_objective_terms["coil_surface_distance"] == "l1_threshold"
        assert config.coil_objective_terms["coil_curvature"] == "lp_threshold"
        assert config.coil_objective_terms["coil_curvature_p"] == 2
        assert config.coil_objective_terms["coil_mean_squared_curvature"] == "l2"
        assert config.coil_objective_terms["linking_number"] == "l2"
        assert config.coil_objective_terms["coil_coil_force"] == "lp"
        assert config.coil_objective_terms["coil_coil_force_p"] == 2.5
        assert config.coil_objective_terms["coil_coil_torque"] == "lp_threshold"
        assert config.coil_objective_terms["coil_coil_torque_p"] == 3.0


class TestAllCoilObjectiveTermOptions:
    """Test all valid options for each coil objective term type."""
    
    @pytest.fixture
    def base_case_yaml(self, tmp_path):
        """Create a base case.yaml file."""
        case_yaml = tmp_path / "case.yaml"
        case_yaml.write_text("""description: Test case
surface_params:
  surface: input.test
coils_params:
  ncoils: 4
  order: 16
optimizer_params:
  algorithm: l-bfgs
""")
        return case_yaml
    
    def test_total_length_l2(self, base_case_yaml):
        """Test total_length with l2 option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  total_length: l2
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["total_length"] == "l2"
    
    def test_total_length_l2_threshold(self, base_case_yaml):
        """Test total_length with l2_threshold option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  total_length: l2_threshold
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["total_length"] == "l2_threshold"
    
    def test_coil_coil_distance_l1(self, base_case_yaml):
        """Test coil_coil_distance with l1 option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_coil_distance: l1
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_coil_distance"] == "l1"
    
    def test_coil_coil_distance_l1_threshold(self, base_case_yaml):
        """Test coil_coil_distance with l1_threshold option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_coil_distance: l1_threshold
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_coil_distance"] == "l1_threshold"
    
    
    def test_coil_surface_distance_l1(self, base_case_yaml):
        """Test coil_surface_distance with l1 option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_surface_distance: l1
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_surface_distance"] == "l1"
    
    def test_coil_surface_distance_l1_threshold(self, base_case_yaml):
        """Test coil_surface_distance with l1_threshold option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_surface_distance: l1_threshold
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_surface_distance"] == "l1_threshold"
    
    
    def test_coil_curvature_lp(self, base_case_yaml):
        """Test coil_curvature with lp option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_curvature: lp
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_curvature"] == "lp"
    
    def test_coil_curvature_lp_threshold(self, base_case_yaml):
        """Test coil_curvature with lp_threshold option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_curvature: lp_threshold
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_curvature"] == "lp_threshold"
    
    def test_coil_curvature_with_p(self, base_case_yaml):
        """Test coil_curvature with p parameter."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_curvature: lp_threshold
  coil_curvature_p: 3.0
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_curvature"] == "lp_threshold"
        assert config.coil_objective_terms["coil_curvature_p"] == 3.0
    
    def test_coil_mean_squared_curvature_l2(self, base_case_yaml):
        """Test coil_mean_squared_curvature with l2 option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_mean_squared_curvature: l2
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_mean_squared_curvature"] == "l2"
    
    def test_coil_mean_squared_curvature_l2_threshold(self, base_case_yaml):
        """Test coil_mean_squared_curvature with l2_threshold option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_mean_squared_curvature: l2_threshold
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_mean_squared_curvature"] == "l2_threshold"
    
    def test_coil_mean_squared_curvature_l1(self, base_case_yaml):
        """Test coil_mean_squared_curvature with l1 option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_mean_squared_curvature: l1
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_mean_squared_curvature"] == "l1"
    
    def test_linking_number_empty(self, base_case_yaml):
        """Test linking_number with empty string (includes linking number, defaults to l2)."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  linking_number: ""
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["linking_number"] == ""
    
    def test_linking_number_l2(self, base_case_yaml):
        """Test linking_number with l2 option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  linking_number: l2
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["linking_number"] == "l2"
    
    def test_linking_number_l2_threshold(self, base_case_yaml):
        """Test linking_number with l2_threshold option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  linking_number: l2_threshold
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["linking_number"] == "l2_threshold"
    
    def test_coil_coil_force_lp(self, base_case_yaml):
        """Test coil_coil_force with lp option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_coil_force: lp
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_coil_force"] == "lp"
    
    def test_coil_coil_force_lp_threshold(self, base_case_yaml):
        """Test coil_coil_force with lp_threshold option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_coil_force: lp_threshold
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_coil_force"] == "lp_threshold"
    
    def test_coil_coil_force_with_p(self, base_case_yaml):
        """Test coil_coil_force with p parameter."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_coil_force: lp_threshold
  coil_coil_force_p: 2.5
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_coil_force"] == "lp_threshold"
        assert config.coil_objective_terms["coil_coil_force_p"] == 2.5
    
    def test_coil_coil_torque_lp(self, base_case_yaml):
        """Test coil_coil_torque with lp option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_coil_torque: lp
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_coil_torque"] == "lp"
    
    def test_coil_coil_torque_lp_threshold(self, base_case_yaml):
        """Test coil_coil_torque with lp_threshold option."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_coil_torque: lp_threshold
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_coil_torque"] == "lp_threshold"
    
    def test_coil_coil_torque_with_p(self, base_case_yaml):
        """Test coil_coil_torque with p parameter."""
        base_case_yaml.write_text(base_case_yaml.read_text() + """coil_objective_terms:
  coil_coil_torque: lp_threshold
  coil_coil_torque_p: 3.0
""")
        config = load_case_config(base_case_yaml)
        assert config.coil_objective_terms["coil_coil_torque"] == "lp_threshold"
        assert config.coil_objective_terms["coil_coil_torque_p"] == 3.0

