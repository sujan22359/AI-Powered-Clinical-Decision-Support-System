"""
Clinical Parameter Data Models

This module defines data models for clinical parameters, reference ranges,
and evaluated parameters used in threshold-based risk assessment.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    """Risk level enumeration for clinical parameter evaluation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass
class ReferenceRange:
    """
    Reference range for a clinical parameter.
    
    Attributes:
        min_value: Minimum acceptable value (optional)
        max_value: Maximum acceptable value (optional)
        unit: Unit of measurement
    
    At least one of min_value or max_value must be provided.
    """
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = ""
    
    def __post_init__(self):
        """Validate that at least one threshold is provided"""
        if self.min_value is None and self.max_value is None:
            raise ValueError("At least one of min_value or max_value must be provided")
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ClinicalParameter:
    """
    Represents a clinical parameter extracted from text.
    
    Attributes:
        name: Parameter name (e.g., "Blood Glucose")
        value: Numeric value of the parameter
        unit: Unit of measurement
        reference_range: Optional reference range for evaluation
    """
    name: str
    value: float
    unit: str
    reference_range: Optional[ReferenceRange] = None
    
    def __post_init__(self):
        """Validate parameter data"""
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Parameter value must be numeric, got {type(self.value).__name__}")
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        result = {
            "name": self.name,
            "value": self.value,
            "unit": self.unit
        }
        if self.reference_range:
            result["reference_range"] = self.reference_range.to_dict()
        return result


@dataclass
class EvaluatedParameter:
    """
    Result of evaluating a clinical parameter against its reference range.
    
    Attributes:
        parameter: The clinical parameter that was evaluated
        risk_level: Assigned risk level (LOW, MEDIUM, HIGH, UNKNOWN)
        deviation_percent: Percentage deviation from reference range
        explanation: Human-readable explanation of the risk assessment
    """
    parameter: ClinicalParameter
    risk_level: RiskLevel
    deviation_percent: Optional[float]
    explanation: str
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "parameter": self.parameter.to_dict(),
            "risk_level": self.risk_level.value,
            "deviation_percent": self.deviation_percent,
            "explanation": self.explanation
        }
