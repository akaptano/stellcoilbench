"""
Validation functions for case.yaml configuration files.
"""
from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path
import yaml


def validate_case_config(data: Dict[str, Any], file_path: Path | None = None) -> List[str]:
    """
    Validate a case.yaml configuration dictionary.
    
    Returns a list of error messages. Empty list means validation passed.
    """
    errors: List[str] = []
    file_prefix = f"{file_path}: " if file_path else ""
    
    # Required fields
    required_fields = ["description", "surface_params", "coils_params", "optimizer_params"]
    for field in required_fields:
        if field not in data:
            errors.append(f"{file_prefix}Missing required field: {field}")
    
    # Validate surface_params
    if "surface_params" in data:
        surface_params = data["surface_params"]
        if not isinstance(surface_params, dict):
            errors.append(f"{file_prefix}surface_params must be a dictionary")
        else:
            # Valid surface_params keys
            valid_surface_params = {
                "surface",  # Required: surface filename (must match file in plasma_surfaces/)
                "range",  # Optional: surface range ("half period" or "full torus")
            }
            
            # Check for unknown parameters
            for key in surface_params.keys():
                if key not in valid_surface_params:
                    errors.append(
                        f"{file_prefix}Unknown surface_params key: '{key}'. "
                        f"Valid keys: {sorted(valid_surface_params)}"
                    )
            
            if "surface" not in surface_params:
                errors.append(f"{file_prefix}surface_params must contain 'surface' field")
            if "range" in surface_params:
                valid_ranges = ["half period", "full torus"]
                if surface_params["range"] not in valid_ranges:
                    errors.append(f"{file_prefix}surface_params.range must be one of {valid_ranges}")
    
    # Validate coils_params
    if "coils_params" in data:
        coils_params = data["coils_params"]
        if not isinstance(coils_params, dict):
            errors.append(f"{file_prefix}coils_params must be a dictionary")
        else:
            # Valid coils_params keys
            valid_coils_params = {
                "ncoils",  # Required: number of coils (must be int)
                "order",  # Required: Fourier order (must be int)
                "verbose",  # Optional: verbose flag (string "True"/"False")
            }
            
            # Check for unknown parameters
            for key in coils_params.keys():
                if key not in valid_coils_params:
                    errors.append(
                        f"{file_prefix}Unknown coils_params key: '{key}'. "
                        f"Valid keys: {sorted(valid_coils_params)}. "
                        f"Note: 'target_B' is no longer used (determined from surface file)."
                    )
            
            # Validate ncoils (must be integer, not float)
            if "ncoils" in coils_params:
                ncoils = coils_params["ncoils"]
                if isinstance(ncoils, float) and ncoils.is_integer():
                    # Allow float that represents an integer, but warn
                    errors.append(
                        f"{file_prefix}coils_params.ncoils should be an integer, not a float. "
                        f"Got {ncoils}. Use {int(ncoils)} instead."
                    )
                elif not isinstance(ncoils, int) or ncoils < 1:
                    errors.append(
                        f"{file_prefix}coils_params.ncoils must be a positive integer, "
                        f"got {type(ncoils).__name__}: {ncoils}"
                    )
            
            # Validate order (must be integer, not float)
            if "order" in coils_params:
                order = coils_params["order"]
                if isinstance(order, float) and order.is_integer():
                    # Allow float that represents an integer, but warn
                    errors.append(
                        f"{file_prefix}coils_params.order should be an integer, not a float. "
                        f"Got {order}. Use {int(order)} instead."
                    )
                elif not isinstance(order, int) or order < 1:
                    errors.append(
                        f"{file_prefix}coils_params.order must be a positive integer, "
                        f"got {type(order).__name__}: {order}"
                    )
            
    
    # Validate optimizer_params
    if "optimizer_params" in data:
        optimizer_params = data["optimizer_params"]
        if not isinstance(optimizer_params, dict):
            errors.append(f"{file_prefix}optimizer_params must be a dictionary")
        else:
            if "max_iterations" in optimizer_params:
                max_iter = optimizer_params["max_iterations"]
                if not isinstance(max_iter, int) or max_iter < 1:
                    errors.append(f"{file_prefix}optimizer_params.max_iterations must be a positive integer")
            if "max_iter_lag" in optimizer_params:
                max_iter_lag = optimizer_params["max_iter_lag"]
                if not isinstance(max_iter_lag, int) or max_iter_lag < 1:
                    errors.append(f"{file_prefix}optimizer_params.max_iter_lag must be a positive integer")
    
    # Validate coil_objective_terms if present
    if "coil_objective_terms" in data:
        obj_terms = data["coil_objective_terms"]
        if not isinstance(obj_terms, dict):
            errors.append(f"{file_prefix}coil_objective_terms must be a dictionary")
        else:
            # Valid objective term names
            valid_term_names = {
                "total_length",
                "coil_coil_distance",
                "coil_surface_distance",
                "coil_curvature",
                "coil_mean_squared_curvature",
                "linking_number",
                "coil_coil_force",
                "coil_coil_torque",
            }
            
            # Valid options for each term type
            valid_options_l2 = ["l2", "l2_threshold"]
            valid_options_l1 = ["l1", "l1_threshold"]
            # valid_options_lp = ["lp", "lp_threshold"]
            valid_options_curvature = ["lp", "lp_threshold"]
            valid_options_msc = ["l2", "l2_threshold", "l1", "l1_threshold"]
            valid_options_force_torque = ["lp", "lp_threshold"]
            
            for term_name, term_value in obj_terms.items():
                # Check for unknown term names
                if term_name not in valid_term_names and not term_name.endswith("_p"):
                    errors.append(
                        f"{file_prefix}Unknown coil_objective_terms key: '{term_name}'. "
                        f"Valid keys: {sorted(valid_term_names)}"
                    )
                    continue
                
                # Skip _p parameters (handled separately)
                if term_name.endswith("_p"):
                    if not isinstance(term_value, (int, float)) or term_value <= 0:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.{term_name} must be a positive number"
                        )
                    continue
                
                # Validate term values
                if term_name == "total_length":
                    if term_value not in valid_options_l2:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.total_length must be one of {valid_options_l2}, "
                            f"got '{term_value}'"
                        )
                elif term_name == "coil_coil_distance":
                    if term_value not in valid_options_l1:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.coil_coil_distance must be one of {valid_options_l1}, "
                            f"got '{term_value}'"
                        )
                elif term_name == "coil_surface_distance":
                    if term_value not in valid_options_l1:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.coil_surface_distance must be one of {valid_options_l1}, "
                            f"got '{term_value}'"
                        )
                elif term_name == "coil_curvature":
                    if term_value not in valid_options_curvature:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.coil_curvature must be one of {valid_options_curvature}, "
                            f"got '{term_value}'"
                        )
                elif term_name == "coil_mean_squared_curvature":
                    if term_value not in valid_options_msc:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.coil_mean_squared_curvature must be one of {valid_options_msc}, "
                            f"got '{term_value}'"
                        )
                elif term_name == "linking_number":
                    # Empty string means include it (defaults to l2), otherwise should be a valid option
                    if term_value != "" and term_value not in ["l2", "l2_threshold"]:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.linking_number must be empty string or one of ['l2', 'l2_threshold'], "
                            f"got '{term_value}'"
                        )
                elif term_name == "coil_coil_force":
                    if term_value not in valid_options_force_torque:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.coil_coil_force must be one of {valid_options_force_torque}, "
                            f"got '{term_value}'"
                        )
                elif term_name == "coil_coil_torque":
                    if term_value not in valid_options_force_torque:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.coil_coil_torque must be one of {valid_options_force_torque}, "
                            f"got '{term_value}'"
                        )
    
    return errors


def validate_case_yaml_file(file_path: Path) -> List[str]:
    """
    Validate a case.yaml file.
    
    Returns a list of error messages. Empty list means validation passed.
    """
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return [f"{file_path}: File is empty or contains no valid YAML"]
        
        if not isinstance(data, dict):
            return [f"{file_path}: Root element must be a dictionary"]
        
        return validate_case_config(data, file_path)
    except yaml.YAMLError as e:
        return [f"{file_path}: YAML parsing error: {e}"]
    except Exception as e:
        return [f"{file_path}: Error reading file: {e}"]

