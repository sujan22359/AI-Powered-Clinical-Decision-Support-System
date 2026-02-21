"""
Threshold Evaluator

This module evaluates clinical parameters against reference ranges and assigns risk levels.
"""

from typing import List, Optional
import logging
from backend.models.clinical_parameters import (
    ClinicalParameter,
    EvaluatedParameter,
    ReferenceRange,
    RiskLevel
)


class ThresholdEvaluator:
    """
    Evaluates clinical parameters against reference ranges and assigns risk levels.
    
    Risk level assignment rules:
    - Within range (0% deviation): LOW
    - 1-20% deviation: MEDIUM
    - >20% deviation: HIGH
    - No reference range: UNKNOWN
    """
    
    def __init__(self):
        """Initialize the threshold evaluator"""
        self.logger = logging.getLogger(__name__)
    
    def evaluate_parameters(self, params: List[ClinicalParameter]) -> List[EvaluatedParameter]:
        """
        Evaluate multiple clinical parameters against their reference ranges.
        
        Args:
            params: List of ClinicalParameter objects to evaluate
        
        Returns:
            List of EvaluatedParameter objects with risk assessments
        """
        evaluated_params = []
        
        for param in params:
            try:
                evaluated = self.evaluate_parameter(param)
                evaluated_params.append(evaluated)
            except Exception as e:
                self.logger.error(f"Failed to evaluate parameter {param.name}: {e}")
                # Continue processing other parameters
                continue
        
        return evaluated_params
    
    def evaluate_parameter(self, param: ClinicalParameter) -> EvaluatedParameter:
        """
        Evaluate a single clinical parameter against its reference range.
        
        Args:
            param: ClinicalParameter object to evaluate
        
        Returns:
            EvaluatedParameter object with risk assessment
        """
        # Check if reference range is available
        if param.reference_range is None:
            return EvaluatedParameter(
                parameter=param,
                risk_level=RiskLevel.UNKNOWN,
                deviation_percent=None,
                explanation=f"No reference range available for {param.name}."
            )
        
        # Calculate deviation from reference range
        deviation = self._calculate_deviation(param.value, param.reference_range)
        
        # Assign risk level based on deviation
        risk_level = self._assign_risk_level(deviation)
        
        # Generate explanation (will be done by RiskExplainer in full implementation)
        # For now, create a basic explanation
        explanation = self._generate_basic_explanation(param, deviation, risk_level)
        
        return EvaluatedParameter(
            parameter=param,
            risk_level=risk_level,
            deviation_percent=deviation,
            explanation=explanation
        )
    
    def _calculate_deviation(self, value: float, ref_range: ReferenceRange) -> float:
        """
        Calculate percentage deviation from reference range.
        
        Deviation calculation logic:
        - If value is within range: 0%
        - If value is above max: ((value - max) / max) * 100
        - If value is below min: ((value - min) / min) * 100 (negative)
        
        Examples:
        - Value 150, Range 70-100: ((150-100)/100)*100 = 50% (HIGH risk)
        - Value 110, Range 70-100: ((110-100)/100)*100 = 10% (MEDIUM risk)
        - Value 85, Range 70-100: 0% (LOW risk)
        - Value 60, Range 70-100: ((60-70)/70)*100 = -14.3% (MEDIUM risk)
        
        Args:
            value: Parameter value to evaluate
            ref_range: Reference range to compare against
        
        Returns:
            Deviation percentage:
            - 0 if within range
            - Positive % if above max
            - Negative % if below min
        """
        # Check if value is below minimum
        if ref_range.min_value is not None and value < ref_range.min_value:
            # Below minimum - calculate negative deviation
            # Example: value=60, min=70 -> ((60-70)/70)*100 = -14.3%
            if ref_range.min_value == 0:
                # Handle division by zero - if min is 0 and value is negative,
                # return a large negative deviation
                return -100.0
            return ((value - ref_range.min_value) / ref_range.min_value) * 100
        
        # Check if value is above maximum
        elif ref_range.max_value is not None and value > ref_range.max_value:
            # Above maximum - calculate positive deviation
            # Example: value=150, max=100 -> ((150-100)/100)*100 = 50%
            if ref_range.max_value == 0:
                # Handle division by zero - if max is 0 and value is positive,
                # return a large positive deviation
                return 100.0
            return ((value - ref_range.max_value) / ref_range.max_value) * 100
        
        # Within range - no deviation
        else:
            return 0.0
    
    def _assign_risk_level(self, deviation_percent: float) -> RiskLevel:
        """
        Assign risk level based on deviation percentage.
        
        Risk level thresholds:
        - 0% deviation: LOW (within normal range)
        - 1-20% deviation: MEDIUM (slightly outside range, monitor)
        - >20% deviation: HIGH (significantly outside range, requires attention)
        
        Examples:
        - 0% -> LOW
        - 10% -> MEDIUM
        - -15% -> MEDIUM (absolute value used)
        - 50% -> HIGH
        - -30% -> HIGH (absolute value used)
        
        Args:
            deviation_percent: Percentage deviation from reference range
        
        Returns:
            RiskLevel (LOW, MEDIUM, or HIGH)
        """
        # Use absolute value since direction doesn't affect risk level
        # (both high and low values can be concerning)
        abs_deviation = abs(deviation_percent)
        
        if abs_deviation == 0:
            # Within normal range
            return RiskLevel.LOW
        elif abs_deviation <= 20:
            # Slightly outside range - monitor
            return RiskLevel.MEDIUM
        else:
            # Significantly outside range - requires attention
            return RiskLevel.HIGH
    
    def _generate_basic_explanation(
        self,
        param: ClinicalParameter,
        deviation: float,
        risk_level: RiskLevel
    ) -> str:
        """
        Generate a basic explanation for the risk assessment.
        
        This is a simplified version. The full implementation will use RiskExplainer.
        
        Args:
            param: Clinical parameter
            deviation: Deviation percentage
            risk_level: Assigned risk level
        
        Returns:
            Explanation string
        """
        ref_range = param.reference_range
        
        # Format reference range
        if ref_range.min_value is not None and ref_range.max_value is not None:
            range_str = f"{ref_range.min_value}-{ref_range.max_value} {ref_range.unit}"
        elif ref_range.min_value is not None:
            range_str = f"≥{ref_range.min_value} {ref_range.unit}"
        elif ref_range.max_value is not None:
            range_str = f"≤{ref_range.max_value} {ref_range.unit}"
        else:
            range_str = "unknown"
        
        # Generate explanation based on risk level
        if risk_level == RiskLevel.LOW:
            return (
                f"{param.name} is within normal range. "
                f"Value: {param.value} {param.unit} (Normal: {range_str})."
            )
        elif risk_level == RiskLevel.MEDIUM:
            direction = "above" if deviation > 0 else "below"
            return (
                f"{param.name} is {direction} normal range. "
                f"Value: {param.value} {param.unit} (Normal: {range_str}). "
                f"Deviation: {abs(deviation):.1f}%. Consider follow-up with healthcare provider."
            )
        elif risk_level == RiskLevel.HIGH:
            direction = "above" if deviation > 0 else "below"
            return (
                f"{param.name} is significantly {direction} normal range. "
                f"Value: {param.value} {param.unit} (Normal: {range_str}). "
                f"Deviation: {abs(deviation):.1f}%. This requires medical attention."
            )
        else:
            return f"Unable to assess {param.name}."
