"""
Risk Explainer

This module generates human-readable explanations for risk assessments.
"""

from backend.models.clinical_parameters import EvaluatedParameter, RiskLevel, ReferenceRange


class RiskExplainer:
    """
    Generates human-readable explanations for clinical parameter risk assessments.
    
    Provides context-appropriate explanations based on risk level, including
    actual values, reference ranges, and deviation percentages.
    """
    
    # Explanation templates for different risk levels
    HIGH_RISK_TEMPLATE = (
        "{param_name} is significantly {direction} normal range. "
        "Value: {value} {unit} (Normal: {ref_range}). "
        "Deviation: {deviation:.1f}%. This requires medical attention."
    )
    
    MEDIUM_RISK_TEMPLATE = (
        "{param_name} is {direction} normal range. "
        "Value: {value} {unit} (Normal: {ref_range}). "
        "Deviation: {deviation:.1f}%. Consider follow-up with healthcare provider."
    )
    
    LOW_RISK_TEMPLATE = (
        "{param_name} is within normal range. "
        "Value: {value} {unit} (Normal: {ref_range})."
    )
    
    UNKNOWN_RISK_TEMPLATE = (
        "No reference range available for {param_name}. "
        "Value: {value} {unit}."
    )
    
    def generate_explanation(self, evaluated_param: EvaluatedParameter) -> str:
        """
        Generate a human-readable explanation for a risk assessment.
        
        Args:
            evaluated_param: EvaluatedParameter object with risk assessment
        
        Returns:
            Explanation string appropriate for the risk level
        """
        param = evaluated_param.parameter
        risk_level = evaluated_param.risk_level
        deviation = evaluated_param.deviation_percent
        
        # Format reference range
        ref_range_str = self._format_reference_range(param.reference_range)
        
        # Determine direction (above or below)
        direction = "above" if deviation and deviation > 0 else "below"
        
        # Generate explanation based on risk level
        if risk_level == RiskLevel.HIGH:
            return self.HIGH_RISK_TEMPLATE.format(
                param_name=param.name,
                direction=direction,
                value=param.value,
                unit=param.unit,
                ref_range=ref_range_str,
                deviation=abs(deviation) if deviation else 0
            )
        elif risk_level == RiskLevel.MEDIUM:
            return self.MEDIUM_RISK_TEMPLATE.format(
                param_name=param.name,
                direction=direction,
                value=param.value,
                unit=param.unit,
                ref_range=ref_range_str,
                deviation=abs(deviation) if deviation else 0
            )
        elif risk_level == RiskLevel.LOW:
            return self.LOW_RISK_TEMPLATE.format(
                param_name=param.name,
                value=param.value,
                unit=param.unit,
                ref_range=ref_range_str
            )
        else:  # UNKNOWN
            return self.UNKNOWN_RISK_TEMPLATE.format(
                param_name=param.name,
                value=param.value,
                unit=param.unit
            )
    
    def _format_reference_range(self, ref_range: ReferenceRange) -> str:
        """
        Format reference range for display in explanations.
        
        Args:
            ref_range: ReferenceRange object to format
        
        Returns:
            Formatted string representation of the range
        """
        if ref_range is None:
            return "unknown"
        
        if ref_range.min_value is not None and ref_range.max_value is not None:
            # Both min and max
            return f"{ref_range.min_value}-{ref_range.max_value} {ref_range.unit}"
        elif ref_range.min_value is not None:
            # Only min (directional threshold)
            return f"≥{ref_range.min_value} {ref_range.unit}"
        elif ref_range.max_value is not None:
            # Only max (directional threshold)
            return f"≤{ref_range.max_value} {ref_range.unit}"
        else:
            return "unknown"
