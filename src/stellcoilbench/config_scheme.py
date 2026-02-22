"""
Configuration dataclasses for StellCoilBench case definitions.

This module defines the schema for case YAML files and submission metadata,
providing typed structures for validation and runtime use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class CaseConfig:
    """
    Configuration for a single benchmark case, usually parsed from case.yaml.

    Attributes
    ----------
    description : str
        Human-readable description of the benchmark case.
    surface_params : dict[str, Any]
        Plasma surface parameters (e.g., surface filename, range, virtual_casing).
    coils_params : dict[str, Any]
        Coil configuration (ncoils, order, coil_type, dipole params, etc.).
    optimizer_params : dict[str, Any]
        Optimizer settings (max_iterations, algorithm, tolerances).
    coil_objective_terms : dict[str, Any] | None
        Optional coil regularization terms (length, curvature, distances).
    fourier_continuation : dict[str, Any] | None
        Optional Fourier continuation orders for progressive refinement.
    post_processing_params : dict[str, Any] | None
        Optional post-processing options (VMEC, Poincaré, etc.).
    """

    description: str
    surface_params: Dict[str, Any]
    coils_params: Dict[str, Any]
    optimizer_params: Dict[str, Any]
    scoring: Dict[str, Any] | None = None
    coil_objective_terms: Dict[str, Any] | None = None
    fourier_continuation: Dict[str, Any] | None = None
    post_processing_params: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CaseConfig:
        """
        Construct a CaseConfig from a parsed YAML/JSON dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Raw configuration dict (e.g., from yaml.safe_load).

        Returns
        -------
        CaseConfig
            Validated configuration instance.
        """
        return cls(
            description=data.get("description", ""),
            surface_params=data.get("surface_params", {}),
            coils_params=data.get("coils_params", {}),
            optimizer_params=data.get("optimizer_params", {}),
            coil_objective_terms=data.get("coil_objective_terms"),
            fourier_continuation=data.get("fourier_continuation"),
            post_processing_params=data.get("post_processing_params"),
        )


@dataclass
class SubmissionMetadata:
    """
    Descriptive information about a submission or method implementation.

    Attributes
    ----------
    method_version : str
        Version string for reproducibility.
    contact : str
        Contact identifier (e.g., GitHub username, email).
    hardware : str
        Hardware description (e.g., CPU model, GPU type).
    """

    method_version: str
    contact: str
    hardware: str

