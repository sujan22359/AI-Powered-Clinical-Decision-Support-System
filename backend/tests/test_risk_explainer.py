"""
Unit tests for risk explainer
"""

import pytest
from backend.services.risk_explainer import RiskExplainer
from backend.models.clinical_parameters import (
    ClinicalParameter,
    ReferenceRange,
    EvaluatedParameter,
    RiskLevel
)


class TestRiskExplainer:
    """Tests for RiskExplainer class"""
    
    def test_high_risk_explanation_contains_severity_language(self):
        """Test that HIGH risk explanations contain severity language"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=200.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.HIGH,
            deviation_percent=100.0,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        # Should contain severity language
        assert "significantly" in explanation.lower()
        assert "requires medical attention" in explanation.lower()
    
    def test_high_risk_explanation_includes_value_range_deviation(self):
        """Test that HIGH risk explanations include value, range, and deviation"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=150.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.HIGH,
            deviation_percent=50.0,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        # Should include actual value
        assert "150" in explanation or "150.0" in explanation
        # Should include reference range
        assert "70" in explanation or "100" in explanation
        # Should include deviation
        assert "50" in explanation
    
    def test_medium_risk_explanation_contains_monitoring_language(self):
        """Test that MEDIUM risk explanations contain monitoring language"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=110.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.MEDIUM,
            deviation_percent=10.0,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        # Should contain monitoring language
        assert "consider" in explanation.lower()
        assert "follow-up" in explanation.lower()
        assert "healthcare provider" in explanation.lower()
    
    def test_medium_risk_explanation_includes_value_range_deviation(self):
        """Test that MEDIUM risk explanations include value, range, and deviation"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=115.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.MEDIUM,
            deviation_percent=15.0,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        # Should include actual value
        assert "115" in explanation or "115.0" in explanation
        # Should include reference range
        assert "70" in explanation or "100" in explanation
        # Should include deviation
        assert "15" in explanation
    
    def test_low_risk_explanation_confirms_normal_status(self):
        """Test that LOW risk explanations confirm normal status"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=85.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.LOW,
            deviation_percent=0.0,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        # Should confirm normal status
        assert "within normal range" in explanation.lower()
    
    def test_low_risk_explanation_includes_value_and_range(self):
        """Test that LOW risk explanations include value and range"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=85.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.LOW,
            deviation_percent=0.0,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        # Should include actual value
        assert "85" in explanation or "85.0" in explanation
        # Should include reference range
        assert "70" in explanation or "100" in explanation
    
    def test_distinct_explanations_for_multiple_parameters(self):
        """Test that different parameters get distinct explanations"""
        explainer = RiskExplainer()
        
        ref_range1 = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param1 = ClinicalParameter(
            name="Blood Glucose",
            value=150.0,
            unit="mg/dL",
            reference_range=ref_range1
        )
        evaluated1 = EvaluatedParameter(
            parameter=param1,
            risk_level=RiskLevel.HIGH,
            deviation_percent=50.0,
            explanation=""
        )
        
        ref_range2 = ReferenceRange(max_value=200, unit="mg/dL")
        param2 = ClinicalParameter(
            name="Total Cholesterol",
            value=250.0,
            unit="mg/dL",
            reference_range=ref_range2
        )
        evaluated2 = EvaluatedParameter(
            parameter=param2,
            risk_level=RiskLevel.MEDIUM,
            deviation_percent=25.0,
            explanation=""
        )
        
        explanation1 = explainer.generate_explanation(evaluated1)
        explanation2 = explainer.generate_explanation(evaluated2)
        
        # Explanations should be different
        assert explanation1 != explanation2
        # Each should contain its parameter name
        assert "Blood Glucose" in explanation1
        assert "Total Cholesterol" in explanation2
        # Each should contain its value
        assert "150" in explanation1
        assert "250" in explanation2
    
    def test_format_reference_range_with_both_values(self):
        """Test formatting reference range with both min and max"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        formatted = explainer._format_reference_range(ref_range)
        
        assert "70" in formatted
        assert "100" in formatted
        assert "mg/dL" in formatted
    
    def test_format_reference_range_with_min_only(self):
        """Test formatting reference range with only min value"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=40, unit="mg/dL")
        formatted = explainer._format_reference_range(ref_range)
        
        assert "40" in formatted
        assert "mg/dL" in formatted
        assert "≥" in formatted
    
    def test_format_reference_range_with_max_only(self):
        """Test formatting reference range with only max value"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(max_value=200, unit="mg/dL")
        formatted = explainer._format_reference_range(ref_range)
        
        assert "200" in formatted
        assert "mg/dL" in formatted
        assert "≤" in formatted
    
    def test_format_reference_range_with_none(self):
        """Test formatting when reference range is None"""
        explainer = RiskExplainer()
        
        formatted = explainer._format_reference_range(None)
        
        assert formatted == "unknown"
    
    def test_explanation_direction_above_for_positive_deviation(self):
        """Test that explanations use 'above' for positive deviation"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=150.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.HIGH,
            deviation_percent=50.0,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        assert "above" in explanation.lower()
    
    def test_explanation_direction_below_for_negative_deviation(self):
        """Test that explanations use 'below' for negative deviation"""
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=50.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.HIGH,
            deviation_percent=-28.6,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        assert "below" in explanation.lower()
    
    def test_unknown_risk_explanation(self):
        """Test explanation for UNKNOWN risk level"""
        explainer = RiskExplainer()
        
        param = ClinicalParameter(
            name="Unknown Parameter",
            value=100.0,
            unit="units",
            reference_range=None
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.UNKNOWN,
            deviation_percent=None,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        assert "no reference range" in explanation.lower()
        assert "Unknown Parameter" in explanation
        assert "100" in explanation


# Property-Based Tests
from hypothesis import given, strategies as st, settings


class TestRiskExplainerProperties:
    """Property-based tests for RiskExplainer"""
    
    @given(
        value=st.floats(min_value=150, max_value=300, allow_nan=False, allow_infinity=False),
        deviation=st.floats(min_value=21, max_value=200, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_high_risk_explanation_content(self, value, deviation):
        """
        Feature: threshold-based-risk-assessment, Property 14:
        For any parameter with HIGH risk level, the generated explanation should contain
        severity language and include the actual value, reference range, and deviation percentage.
        
        Validates: Requirements 4.1, 4.4
        """
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Test Parameter",
            value=value,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.HIGH,
            deviation_percent=deviation,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        # Should contain severity language
        assert any(word in explanation.lower() for word in ["significantly", "requires", "medical attention"])
        # Should include value
        assert str(int(value)) in explanation or f"{value:.1f}" in explanation
        # Should include reference range
        assert "70" in explanation or "100" in explanation
        # Should include deviation
        assert str(int(deviation)) in explanation or f"{deviation:.1f}" in explanation
    
    @given(
        value=st.floats(min_value=101, max_value=120, allow_nan=False, allow_infinity=False),
        deviation=st.floats(min_value=1, max_value=20, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_medium_risk_explanation_content(self, value, deviation):
        """
        Feature: threshold-based-risk-assessment, Property 15:
        For any parameter with MEDIUM risk level, the generated explanation should contain
        monitoring language and include the actual value, reference range, and deviation percentage.
        
        Validates: Requirements 4.2, 4.4
        """
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Test Parameter",
            value=value,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.MEDIUM,
            deviation_percent=deviation,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        # Should contain monitoring language
        assert any(word in explanation.lower() for word in ["consider", "follow-up", "healthcare provider"])
        # Should include value
        assert str(int(value)) in explanation or f"{value:.1f}" in explanation
        # Should include reference range
        assert "70" in explanation or "100" in explanation
        # Should include deviation
        assert str(int(deviation)) in explanation or f"{deviation:.1f}" in explanation
    
    @given(
        value=st.floats(min_value=70, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_low_risk_explanation_content(self, value):
        """
        Feature: threshold-based-risk-assessment, Property 16:
        For any parameter with LOW risk level, the generated explanation should confirm
        normal status and include the actual value and reference range.
        
        Validates: Requirements 4.3, 4.4
        """
        explainer = RiskExplainer()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Test Parameter",
            value=value,
            unit="mg/dL",
            reference_range=ref_range
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.LOW,
            deviation_percent=0.0,
            explanation=""
        )
        
        explanation = explainer.generate_explanation(evaluated)
        
        # Should confirm normal status
        assert "within normal range" in explanation.lower()
        # Should include value
        assert str(int(value)) in explanation or f"{value:.1f}" in explanation
        # Should include reference range
        assert "70" in explanation or "100" in explanation
    
    @given(
        param_names=st.lists(
            st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
            min_size=2,
            max_size=5,
            unique=True
        ),
        values=st.lists(
            st.floats(min_value=50, max_value=200, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=100)
    def test_property_distinct_explanations(self, param_names, values):
        """
        Feature: threshold-based-risk-assessment, Property 17:
        For any set of evaluated parameters, each should have a distinct explanation
        that includes the specific parameter name and values.
        
        Validates: Requirements 4.6
        """
        explainer = RiskExplainer()
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        
        # Ensure we have matching lengths
        num_params = min(len(param_names), len(values))
        param_names = param_names[:num_params]
        values = values[:num_params]
        
        explanations = []
        for param_name, value in zip(param_names, values):
            param = ClinicalParameter(
                name=param_name,
                value=value,
                unit="mg/dL",
                reference_range=ref_range
            )
            evaluated = EvaluatedParameter(
                parameter=param,
                risk_level=RiskLevel.HIGH,
                deviation_percent=50.0,
                explanation=""
            )
            explanation = explainer.generate_explanation(evaluated)
            explanations.append(explanation)
            
            # Each explanation should contain its parameter name
            assert param_name in explanation
        
        # All explanations should be distinct (if parameter names are distinct)
        if len(set(param_names)) == len(param_names):
            assert len(set(explanations)) == len(explanations)
