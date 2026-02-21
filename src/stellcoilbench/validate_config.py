"""
Validation functions for case.yaml configuration files and CI autopilot case JSON.
"""
from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path
import json
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
                "virtual_casing",  # Optional: enable virtual casing (boolean, default: false)
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
            if "virtual_casing" in surface_params:
                if not isinstance(surface_params["virtual_casing"], bool):
                    errors.append(f"{file_prefix}surface_params.virtual_casing must be a boolean (true/false)")
    
    # Validate coils_params
    if "coils_params" in data:
        coils_params = data["coils_params"]
        if not isinstance(coils_params, dict):
            errors.append(f"{file_prefix}coils_params must be a dictionary")
        else:
            # Valid coils_params keys (modular + dipole)
            valid_coils_params = {
                "ncoils",
                "order",
                "coil_type",
                "tf_configuration",
                "Nx", "Ny", "Nz",
                "dipole_order",
                "poff", "coff",
                "dipole_coil_size", "tf_coil_size",
                "remove_inboard_eps",
            }
            
            # Check for unknown parameters
            for key in coils_params.keys():
                if key not in valid_coils_params:
                    errors.append(
                        f"{file_prefix}Unknown coils_params key: '{key}'. "
                        f"Valid keys: {sorted(valid_coils_params)}. "
                        f"Note: 'target_B' is no longer used (determined from surface file)."
                    )
            
            coil_type = coils_params.get("coil_type", "modular")
            if coil_type == "dipole":
                pass  # dipole uses Nx, dipole_order, tf_configuration instead
            elif "ncoils" not in coils_params:
                errors.append(
                    f"{file_prefix}coils_params must include 'ncoils' for modular coils. "
                    f"For dipole coils, set coil_type: 'dipole'."
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
                "coil_arclength_variation",
                "coil_mean_squared_curvature",
                "linking_number",
                "coil_coil_force",
                "coil_coil_torque",
            }
            
            # Valid threshold parameter names (these are extracted and passed as kwargs)
            valid_threshold_names = {
                "length_threshold",
                "cc_threshold",
                "cs_threshold",
                "curvature_threshold",
                "arclength_variation_threshold",
                "msc_threshold",
                "force_threshold",
                "torque_threshold",
                "flux_threshold",
            }
            
            # Valid weight parameter names (allow specifying weights for each term)
            valid_weight_names = {
                "length_weight",
                "cc_weight",
                "cs_weight",
                "curvature_weight",
                "arclength_variation_weight",
                "msc_weight",
                "force_weight",
                "torque_weight",
                "flux_weight",
                "linking_weight",
            }
            
            # Valid options for each term type
            valid_options_l2 = ["l2", "l2_threshold"]
            # valid_options_l1 = ["l1", "l1_threshold"]
            # valid_options_l1_l2 = ["l1", "l1_threshold", "l2", "l2_threshold"]  # For distance terms that support both
            # valid_options_lp = ["lp", "lp_threshold"]
            valid_options_curvature = ["lp", "lp_threshold"]
            valid_options_msc = ["l2", "l2_threshold", "l1", "l1_threshold"]
            valid_options_arclength = ["l2", "l2_threshold", "l1", "l1_threshold"]
            valid_options_force_torque = ["lp", "lp_threshold"]
            
            def is_valid_non_negative_number(value):
                """Check if value is a valid non-negative number (int, float, or parseable string)."""
                if isinstance(value, bool):  # bool is subclass of int, reject it
                    return False
                if isinstance(value, (int, float)):
                    return value >= 0
                if isinstance(value, str):
                    try:
                        return float(value) >= 0
                    except ValueError:
                        return False
                return False
            
            for term_name, term_value in obj_terms.items():
                # Skip threshold parameters (they are validated separately)
                if term_name in valid_threshold_names:
                    if not is_valid_non_negative_number(term_value):
                        errors.append(
                            f"{file_prefix}coil_objective_terms.{term_name} must be a non-negative number"
                        )
                    continue
                
                # Skip weight parameters (they are validated separately)
                if term_name in valid_weight_names:
                    if not is_valid_non_negative_number(term_value):
                        errors.append(
                            f"{file_prefix}coil_objective_terms.{term_name} must be a non-negative number"
                        )
                    continue
                
                # Check for unknown term names
                if term_name not in valid_term_names and not term_name.endswith("_p"):
                    errors.append(
                        f"{file_prefix}Unknown coil_objective_terms key: '{term_name}'. "
                        f"Valid keys: {sorted(valid_term_names | valid_threshold_names | valid_weight_names)}"
                    )
                    continue
                
                # Skip _p parameters (handled separately)
                if term_name.endswith("_p"):
                    # _p parameters must be positive (> 0), not just non-negative
                    valid = False
                    if isinstance(term_value, bool):
                        valid = False
                    elif isinstance(term_value, (int, float)):
                        valid = term_value > 0
                    elif isinstance(term_value, str):
                        try:
                            valid = float(term_value) > 0
                        except ValueError:
                            valid = False
                    if not valid:
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
                    # coil_coil_distance is always included automatically
                    # If specified, it should be empty string (no options needed)
                    if term_value != "":
                        errors.append(
                            f"{file_prefix}coil_objective_terms.coil_coil_distance must be empty string (\"\"), "
                            f"got '{term_value}'. It is always included automatically - use cc_threshold to set threshold."
                        )
                elif term_name == "coil_surface_distance":
                    # coil_surface_distance is always included automatically
                    # If specified, it should be empty string (no options needed)
                    if term_value != "":
                        errors.append(
                            f"{file_prefix}coil_objective_terms.coil_surface_distance must be empty string (\"\"), "
                            f"got '{term_value}'. It is always included automatically - use cs_threshold to set threshold."
                        )
                elif term_name == "coil_curvature":
                    if term_value not in valid_options_curvature:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.coil_curvature must be one of {valid_options_curvature}, "
                            f"got '{term_value}'"
                        )
                elif term_name == "coil_arclength_variation":
                    if term_value not in valid_options_arclength:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.{term_name} must be one of "
                            f"{valid_options_arclength}, got '{term_value}'"
                        )
                elif term_name == "coil_mean_squared_curvature":
                    if term_value not in valid_options_msc:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.coil_mean_squared_curvature must be one of {valid_options_msc}, "
                            f"got '{term_value}'"
                        )
                elif term_name == "linking_number":
                    valid_linking_options = [""]
                    if term_value not in valid_linking_options:
                        errors.append(
                            f"{file_prefix}coil_objective_terms.linking_number must be one of {valid_linking_options}, "
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
    
    # Validate fourier_continuation if present
    if "fourier_continuation" in data:
        fc = data["fourier_continuation"]
        if not isinstance(fc, dict):
            errors.append(f"{file_prefix}fourier_continuation must be a dictionary")
        else:
            if "enabled" in fc:
                if not isinstance(fc["enabled"], bool):
                    errors.append(f"{file_prefix}fourier_continuation.enabled must be a boolean")
            
            if "orders" in fc:
                orders = fc["orders"]
                if not isinstance(orders, list):
                    errors.append(f"{file_prefix}fourier_continuation.orders must be a list")
                elif not orders:
                    errors.append(f"{file_prefix}fourier_continuation.orders must be non-empty")
                elif not all(isinstance(o, int) and o > 0 for o in orders):
                    errors.append(f"{file_prefix}fourier_continuation.orders must contain only positive integers")
                elif orders != sorted(orders):
                    errors.append(f"{file_prefix}fourier_continuation.orders must be in ascending order")
    
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


# ---------------------------------------------------------------------------
# CI autopilot case JSON validation
# ---------------------------------------------------------------------------

# Default resource caps (can be overridden by policy)
_DEFAULT_MAX_TOTAL_ITERATIONS = 10000
_DEFAULT_TIMEOUT_MINUTES_MIN = 5
_DEFAULT_TIMEOUT_MINUTES_MAX = 180


def validate_ci_case(
    data: Dict[str, Any],
    policy: Dict[str, Any] | None = None,
    file_path: Path | None = None,
) -> List[str]:
    """
    Validate a CI autopilot case JSON dictionary.

    The CI case wraps a standard case config inside a ``case_config`` key and
    adds ``case_id``, ``resource``, and optional ``parent_ids`` / ``tags`` /
    ``random_seed`` fields.

    Parameters
    ----------
    data : dict
        Parsed JSON for the CI case.
    policy : dict, optional
        Proposer policy (from ``policy/proposer_policy.yaml``).  If provided,
        resource caps are taken from ``policy["resource_caps"]``.
    file_path : Path, optional
        Used for error-message prefixes.

    Returns
    -------
    list[str]
        Error messages.  Empty list means validation passed.
    """
    errors: List[str] = []
    pfx = f"{file_path}: " if file_path else ""

    caps = (policy or {}).get("resource_caps", {})
    max_iter_cap = caps.get("max_total_iterations", _DEFAULT_MAX_TOTAL_ITERATIONS)
    timeout_min = caps.get("timeout_minutes_min", _DEFAULT_TIMEOUT_MINUTES_MIN)
    timeout_max = caps.get("timeout_minutes_max", _DEFAULT_TIMEOUT_MINUTES_MAX)

    # ---- required top-level keys ----
    if "case_id" not in data:
        errors.append(f"{pfx}Missing required field: case_id")
    elif not isinstance(data["case_id"], str) or not data["case_id"]:
        errors.append(f"{pfx}case_id must be a non-empty string")

    # ---- resource block ----
    resource = data.get("resource", {})
    if not isinstance(resource, dict):
        errors.append(f"{pfx}resource must be a dictionary")
    else:
        mti = resource.get("max_total_iterations")
        if mti is not None:
            if not isinstance(mti, int) or mti < 1:
                errors.append(f"{pfx}resource.max_total_iterations must be a positive integer")
            elif mti > max_iter_cap:
                errors.append(
                    f"{pfx}resource.max_total_iterations ({mti}) exceeds cap ({max_iter_cap})"
                )

        tm = resource.get("timeout_minutes")
        if tm is not None:
            if not isinstance(tm, (int, float)) or tm <= 0:
                errors.append(f"{pfx}resource.timeout_minutes must be a positive number")
            elif tm < timeout_min or tm > timeout_max:
                errors.append(
                    f"{pfx}resource.timeout_minutes ({tm}) outside allowed range "
                    f"[{timeout_min}, {timeout_max}]"
                )

    # ---- optional typed fields ----
    if "parent_ids" in data:
        if not isinstance(data["parent_ids"], list):
            errors.append(f"{pfx}parent_ids must be a list")
    if "tags" in data:
        if not isinstance(data["tags"], list):
            errors.append(f"{pfx}tags must be a list")
    if "random_seed" in data:
        if not isinstance(data["random_seed"], int):
            errors.append(f"{pfx}random_seed must be an integer")

    # ---- case_config (the actual optimisation specification) ----
    cc = data.get("case_config")
    if cc is None:
        errors.append(f"{pfx}Missing required field: case_config")
    elif not isinstance(cc, dict):
        errors.append(f"{pfx}case_config must be a dictionary")
    else:
        inner = validate_case_config(cc, file_path)
        errors.extend(inner)

        # ---- cross-check: maxiter vs resource cap ----
        opt = cc.get("optimizer_params", {})
        maxiter = opt.get("max_iterations")
        if isinstance(maxiter, int) and maxiter > max_iter_cap:
            errors.append(
                f"{pfx}case_config.optimizer_params.max_iterations ({maxiter}) "
                f"exceeds cap ({max_iter_cap})"
            )

    return errors


def validate_ci_case_file(file_path: Path, policy: Dict[str, Any] | None = None) -> List[str]:
    """
    Validate a CI autopilot case JSON file on disk.

    Returns a list of error messages.  Empty list means validation passed.
    """
    try:
        with open(file_path, "r") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        return [f"{file_path}: JSON parse error: {exc}"]
    except Exception as exc:
        return [f"{file_path}: Error reading file: {exc}"]

    if not isinstance(data, dict):
        return [f"{file_path}: Root element must be a JSON object"]

    return validate_ci_case(data, policy=policy, file_path=file_path)

