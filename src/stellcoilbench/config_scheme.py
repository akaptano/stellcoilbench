from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class CaseConfig:
    """
    Configuration for a single benchmark case, usually parsed from case.yaml.
    """

    description: str
    surface_params: Dict[str, Any]
    coils_params: Dict[str, Any]
    optimizer_params: Dict[str, Any]
    scoring: Dict[str, Any] | None = None
    coil_objective_terms: Dict[str, Any] | None = None
    fourier_continuation: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseConfig":
        return cls(
            description=data.get("description", ""),
            surface_params=data.get("surface_params", {}),
            coils_params=data.get("coils_params", {}),
            optimizer_params=data.get("optimizer_params", {}),
            scoring=data.get("scoring"),
            coil_objective_terms=data.get("coil_objective_terms"),
            fourier_continuation=data.get("fourier_continuation"),
        )


@dataclass
class SubmissionMetadata:
    """
    Descriptive information about a submission / method implementation.
    """

    method_name: str
    method_version: str
    contact: str
    hardware: str
    notes: str = ""

