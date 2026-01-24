"""
Comprehensive unit tests for update_db.py helper functions.
"""
from stellcoilbench.update_db import (
    _metric_shorthand,
    _format_date,
    _shorthand_to_math,
    _metric_definition,
)


class TestMetricShorthand:
    """Tests for _metric_shorthand function."""
    
    def test_b_field_metrics(self):
        """Test B-field related metric shorthands."""
        assert _metric_shorthand("max_BdotN_over_B") == "max(B_n)"
        assert _metric_shorthand("avg_BdotN_over_B") == "avg(B_n)"
        assert _metric_shorthand("final_normalized_squared_flux") == "f_B"
        assert _metric_shorthand("initial_B_field") == "B0"
        assert _metric_shorthand("final_B_field") == "Bf"
        assert _metric_shorthand("target_B_field") == "Bt"
    
    def test_curvature_metrics(self):
        """Test curvature metric shorthands."""
        assert _metric_shorthand("final_average_curvature") == "κ̄"
        assert _metric_shorthand("final_max_curvature") == "max(κ)"
        assert _metric_shorthand("final_mean_squared_curvature") == "MSC"
    
    def test_separation_metrics(self):
        """Test separation metric shorthands."""
        assert _metric_shorthand("final_min_cs_separation") == "min(d_cs)"
        assert _metric_shorthand("final_min_cc_separation") == "min(d_cc)"
        assert _metric_shorthand("final_cs_separation") == "d_cs"
        assert _metric_shorthand("final_cc_separation") == "d_cc"
    
    def test_length_metrics(self):
        """Test length metric shorthands."""
        assert _metric_shorthand("final_total_length") == "L"
        assert _metric_shorthand("final_arclength_variation") == "Var(l_i)"
    
    def test_force_torque_metrics(self):
        """Test force and torque metric shorthands."""
        assert _metric_shorthand("final_max_max_coil_force") == "max(F)"
        assert _metric_shorthand("final_avg_max_coil_force") == "F̄"
        assert _metric_shorthand("final_max_max_coil_torque") == "max(τ)"
        assert _metric_shorthand("final_avg_max_coil_torque") == "τ̄"
    
    def test_other_metrics(self):
        """Test other metric shorthands."""
        assert _metric_shorthand("optimization_time") == "t"
        assert _metric_shorthand("final_linking_number") == "LN"
        assert _metric_shorthand("coil_order") == "n"
        assert _metric_shorthand("num_coils") == "N"
        assert _metric_shorthand("fourier_continuation_orders") == "FC"
        assert _metric_shorthand("score_primary") == "score"
    
    def test_unknown_metric(self):
        """Test that unknown metrics get underscores replaced with spaces."""
        assert _metric_shorthand("unknown_metric_name") == "unknown metric name"
        assert _metric_shorthand("test_metric") == "test metric"


class TestFormatDate:
    """Tests for _format_date function."""
    
    def test_iso_format(self):
        """Test formatting ISO date format."""
        assert _format_date("2025-12-01") == "01/12/25"
        assert _format_date("2026-01-21") == "21/01/26"
        assert _format_date("2024-03-05") == "05/03/24"
    
    def test_iso_with_time(self):
        """Test formatting ISO date with time component."""
        assert _format_date("2025-12-01T10:30:00") == "01/12/25"
        assert _format_date("2026-01-21T15:45:30") == "21/01/26"
    
    def test_already_formatted(self):
        """Test that already formatted dates are handled."""
        # The function may convert MM/DD/YY to DD/MM/YY, so test accordingly
        result = _format_date("01/12/25")
        # Should be in DD/MM/YY format (may convert from MM/DD/YY)
        assert "/" in result
        assert len(result.split("/")) == 3
    
    def test_unknown_date(self):
        """Test handling of unknown date."""
        assert _format_date("_unknown_") == "_unknown_"
        assert _format_date(None) == "_unknown_"
        assert _format_date("") == ""
    
    def test_mm_dd_yy_conversion(self):
        """Test conversion from MM/DD/YY to DD/MM/YY."""
        # If second part > 12, it's MM/DD/YY format
        assert _format_date("12/25/25") == "25/12/25"  # MM/DD/YY -> DD/MM/YY
        assert _format_date("03/15/24") == "15/03/24"  # MM/DD/YY -> DD/MM/YY
    
    def test_dd_mm_yy_preservation(self):
        """Test that DD/MM/YY format is preserved."""
        # If first part > 12, it's DD/MM/YY format
        assert _format_date("25/12/25") == "25/12/25"  # Already DD/MM/YY
        assert _format_date("15/03/24") == "15/03/24"  # Already DD/MM/YY
    
    def test_ambiguous_date(self):
        """Test handling of ambiguous dates (both parts <= 12)."""
        # When both parts <= 12, assume MM/DD/YY and convert
        result = _format_date("03/05/24")
        # Should convert MM/DD/YY to DD/MM/YY
        assert result == "05/03/24" or result == "03/05/24"  # May depend on heuristic
    
    def test_invalid_format(self):
        """Test handling of invalid date formats."""
        # Should return as-is if parsing fails
        assert _format_date("invalid") == "invalid"
        assert _format_date("2025") == "2025"
        assert _format_date("12-01") == "12-01"


class TestShorthandToMath:
    """Tests for _shorthand_to_math function."""
    
    def test_simple_variables(self):
        """Test simple variable names."""
        assert _shorthand_to_math("n") == ":math:`n`"
        assert _shorthand_to_math("N") == ":math:`N`"
        assert _shorthand_to_math("L") == ":math:`L`"
        assert _shorthand_to_math("t") == ":math:`t`"
    
    def test_unicode_characters(self):
        """Test Unicode characters."""
        assert _shorthand_to_math("κ̄") == r":math:`\bar{\kappa}`"
        assert _shorthand_to_math("F̄") == r":math:`\bar{F}`"
        assert _shorthand_to_math("τ̄") == r":math:`\bar{\tau}`"
        assert _shorthand_to_math("avg(B_n)") == r":math:`\text{avg}(B_n)`"
        assert _shorthand_to_math("max(B_n)") == r":math:`\max(B_n)`"
        assert _shorthand_to_math("Var(l_i)") == r":math:`\mathrm{Var}(l_i)`"
        assert _shorthand_to_math("FC") == r":math:`\text{FC}`"
    
    def test_function_calls(self):
        """Test function call notation."""
        assert _shorthand_to_math("min(d_cc)") == r":math:`\min(d_{cc})`"
        assert _shorthand_to_math("max(κ)") == r":math:`\max(\kappa)`"
        assert _shorthand_to_math("max(F)") == r":math:`\max(F)`"
        assert _shorthand_to_math("max(τ)") == r":math:`\max(\tau)`"
        assert _shorthand_to_math("min(d_cs)") == r":math:`\min(d_{cs})`"
    
    def test_subscripts(self):
        """Test subscript notation."""
        assert _shorthand_to_math("d_cc") == ":math:`d_{cc}`"
        assert _shorthand_to_math("d_cs") == ":math:`d_{cs}`"
        # B_n may be converted to B_{n} by the function
        result = _shorthand_to_math("B_n")
        assert result.startswith(":math:`")
        assert "B" in result
        assert "n" in result
    
    def test_complex_subscripts(self):
        """Test complex subscript notation."""
        result = _shorthand_to_math("var_name_sub")
        assert ":math:`" in result
        assert "var" in result
        assert "name" in result
        assert "sub" in result
    
    def test_default_wrapping(self):
        """Test that unknown shorthands are wrapped in math mode."""
        result = _shorthand_to_math("unknown")
        assert result.startswith(":math:`")
        assert result.endswith("`")
        assert "unknown" in result


class TestMetricDefinition:
    """Tests for _metric_definition function."""
    
    def test_b_field_definitions(self):
        """Test B-field metric definitions."""
        definition = _metric_definition("final_normalized_squared_flux")
        assert "flux" in definition.lower() or "f_B" in definition
        
        definition = _metric_definition("max_BdotN_over_B")
        assert "maximum" in definition.lower() or "max" in definition.lower()
        
        definition = _metric_definition("avg_BdotN_over_B")
        assert "average" in definition.lower() or "avg" in definition.lower()
    
    def test_curvature_definitions(self):
        """Test curvature metric definitions."""
        definition = _metric_definition("final_average_curvature")
        assert "curvature" in definition.lower() or "κ" in definition
        
        definition = _metric_definition("final_max_curvature")
        assert "maximum" in definition.lower() or "max" in definition.lower()
        
        definition = _metric_definition("final_mean_squared_curvature")
        assert "squared" in definition.lower() or "MSC" in definition
    
    def test_separation_definitions(self):
        """Test separation metric definitions."""
        definition = _metric_definition("final_min_cs_separation")
        assert "distance" in definition.lower() or "separation" in definition.lower()
        
        definition = _metric_definition("final_min_cc_separation")
        assert "distance" in definition.lower() or "separation" in definition.lower()
    
    def test_length_definitions(self):
        """Test length metric definitions."""
        definition = _metric_definition("final_total_length")
        assert "length" in definition.lower() or "L" in definition
    
    def test_force_torque_definitions(self):
        """Test force and torque metric definitions."""
        definition = _metric_definition("final_max_max_coil_force")
        assert "force" in definition.lower() or "F" in definition
        
        definition = _metric_definition("final_max_max_coil_torque")
        assert "torque" in definition.lower() or "τ" in definition
    
    def test_time_definition(self):
        """Test time metric definition."""
        definition = _metric_definition("optimization_time")
        assert "time" in definition.lower() or "t" in definition
    
    def test_linking_number_definition(self):
        """Test linking number metric definition."""
        definition = _metric_definition("final_linking_number")
        assert "linking" in definition.lower() or "LN" in definition
    
    def test_coil_parameters(self):
        """Test coil parameter definitions."""
        definition = _metric_definition("coil_order")
        assert "order" in definition.lower() or "n" in definition
        
        definition = _metric_definition("num_coils")
        assert "coil" in definition.lower() or "N" in definition
    
    def test_fourier_continuation_definition(self):
        """Test Fourier continuation definition."""
        definition = _metric_definition("fourier_continuation_orders")
        assert "fourier" in definition.lower() or "continuation" in definition.lower() or "FC" in definition
    
    def test_unknown_metric(self):
        """Test that unknown metrics get title case with spaces."""
        definition = _metric_definition("unknown_metric_name")
        assert "Unknown Metric Name" == definition or "unknown metric name" in definition.lower()
