from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CaseConfig:
    """
    Configuration for a single benchmark case, usually parsed from case.yaml.
    """

    case_id: str
    nfp: int
    description: str
    coil_type: str
    required_coils: int
    normalization_B0: float
    normalization_length: float
    scoring: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseConfig":
        return cls(
            case_id=data["case_id"],
            nfp=int(data["nfp"]),
            description=data.get("description", ""),
            coil_type=data.get("coil_type", "winding_surface"),
            required_coils=int(data.get("required_coils", 0)),
            normalization_B0=float(data.get("normalization_B0", 1.0)),
            normalization_length=float(data.get("normalization_length", 1.0)),
            scoring=data.get("scoring"),
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

