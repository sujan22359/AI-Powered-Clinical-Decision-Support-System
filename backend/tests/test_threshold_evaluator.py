"""
Unit tests for threshold evaluator
"""

import pytest
from backend.services.threshold_evaluator import ThresholdEvaluator
from backend.models.clinical_parameters import (
    ClinicalParameter,
    ReferenceRange,
    RiskLevel
)


class TestThresholdEvaluator:
    """Tests for ThresholdEvaluator class"""
    
    def test_within_range_assigns_low_risk(self):
        """Test that values within reference range are assigned LOW risk"""
        evaluator = ThresholdEvaluator()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=85.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        assert result.risk_level == RiskLevel.LOW
        assert result.deviation_percent == 0.0
        assert "within normal range" in result.explanation.lower()
    
    def test_1_to_20_percent_deviation_assigns_medium_risk(self):
        """Test that 1-20% deviation assigns MEDIUM risk"""
        evaluator = ThresholdEvaluator()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        
        # Test 10% above max (110)
        param1 = ClinicalParameter(
            name="Blood Glucose",
            value=110.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        result1 = evaluator.evaluate_parameter(param1)
        assert result1.risk_level == RiskLevel.MEDIUM
        assert result1.deviation_percent == 10.0
        
        # Test 15% above max (115)
        param2 = ClinicalParameter(
            name="Blood Glucose",
            value=115.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        result2 = evaluator.evaluate_parameter(param2)
        assert result2.risk_level == RiskLevel.MEDIUM
        assert 14.0 < result2.deviation_percent < 16.0
    
    def test_over_20_percent_deviation_assigns_high_risk(self):
        """Test that >20% deviation assigns HIGH risk"""
        evaluator = ThresholdEvaluator()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        
        # Test 50% above max (150)
        param = ClinicalParameter(
            name="Blood Glucose",
            value=150.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        assert result.risk_level == RiskLevel.HIGH
        assert result.deviation_percent == 50.0
        assert "significantly" in result.explanation.lower()
    
    def test_below_range_calculates_negative_deviation(self):
        """Test that values below minimum have negative deviation"""
        evaluator = ThresholdEvaluator()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=60.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        # Deviation should be negative
        assert result.deviation_percent < 0
        # Should be approximately -14.3% ((60-70)/70 * 100)
        assert -15.0 < result.deviation_percent < -14.0
    
    def test_missing_reference_range_assigns_unknown_risk(self):
        """Test that missing reference range assigns UNKNOWN risk"""
        evaluator = ThresholdEvaluator()
        
        param = ClinicalParameter(
            name="Unknown Parameter",
            value=100.0,
            unit="units",
            reference_range=None
        )
        
        result = evaluator.evaluate_parameter(param)
        
        assert result.risk_level == RiskLevel.UNKNOWN
        assert result.deviation_percent is None
        assert "no reference range" in result.explanation.lower()
    
    def test_deviation_calculation_for_various_scenarios(self):
        """Test deviation calculation accuracy for various scenarios"""
        evaluator = ThresholdEvaluator()
        
        # Scenario 1: Exactly at max
        ref_range1 = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param1 = ClinicalParameter(
            name="Test", value=100.0, unit="mg/dL", reference_range=ref_range1
        )
        result1 = evaluator.evaluate_parameter(param1)
        assert result1.deviation_percent == 0.0
        
        # Scenario 2: Exactly at min
        param2 = ClinicalParameter(
            name="Test", value=70.0, unit="mg/dL", reference_range=ref_range1
        )
        result2 = evaluator.evaluate_parameter(param2)
        assert result2.deviation_percent == 0.0
        
        # Scenario 3: Only max threshold (directional)
        ref_range3 = ReferenceRange(max_value=200, unit="mg/dL")
        param3 = ClinicalParameter(
            name="Test", value=150.0, unit="mg/dL", reference_range=ref_range3
        )
        result3 = evaluator.evaluate_parameter(param3)
        assert result3.deviation_percent == 0.0
        
        # Scenario 4: Only min threshold (directional)
        ref_range4 = ReferenceRange(min_value=40, unit="mg/dL")
        param4 = ClinicalParameter(
            name="Test", value=50.0, unit="mg/dL", reference_range=ref_range4
        )
        result4 = evaluator.evaluate_parameter(param4)
        assert result4.deviation_percent == 0.0
    
    def test_batch_evaluation_of_multiple_parameters(self):
        """Test evaluating multiple parameters at once"""
        evaluator = ThresholdEvaluator()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        
        params = [
            ClinicalParameter(name="Param1", value=85.0, unit="mg/dL", reference_range=ref_range),
            ClinicalParameter(name="Param2", value=110.0, unit="mg/dL", reference_range=ref_range),
            ClinicalParameter(name="Param3", value=150.0, unit="mg/dL", reference_range=ref_range),
        ]
        
        results = evaluator.evaluate_parameters(params)
        
        # Should return same number of results
        assert len(results) == 3
        
        # Check risk levels
        assert results[0].risk_level == RiskLevel.LOW
        assert results[1].risk_level == RiskLevel.MEDIUM
        assert results[2].risk_level == RiskLevel.HIGH
    
    def test_division_by_zero_handling(self):
        """Test graceful handling of division by zero in deviation calculation"""
        evaluator = ThresholdEvaluator()
        
        # Test with min_value = 0
        ref_range1 = ReferenceRange(min_value=0, max_value=100, unit="units")
        param1 = ClinicalParameter(
            name="Test", value=-10.0, unit="units", reference_range=ref_range1
        )
        result1 = evaluator.evaluate_parameter(param1)
        # Should handle gracefully and return a large negative deviation
        assert result1.deviation_percent < 0
        assert result1.risk_level == RiskLevel.HIGH
        
        # Test with max_value = 0
        ref_range2 = ReferenceRange(min_value=-100, max_value=0, unit="units")
        param2 = ClinicalParameter(
            name="Test", value=10.0, unit="units", reference_range=ref_range2
        )
        result2 = evaluator.evaluate_parameter(param2)
        # Should handle gracefully and return a large positive deviation
        assert result2.deviation_percent > 0
        assert result2.risk_level == RiskLevel.HIGH
    
    def test_explanation_includes_value_range_and_deviation(self):
        """Test that explanations include value, range, and deviation"""
        evaluator = ThresholdEvaluator()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=150.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        # Check that explanation contains key information
        assert "150" in result.explanation or "150.0" in result.explanation
        assert "70" in result.explanation or "100" in result.explanation
        assert "50" in result.explanation  # Deviation percentage
    
    def test_high_risk_explanation_contains_severity_language(self):
        """Test that HIGH risk explanations contain severity language"""
        evaluator = ThresholdEvaluator()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=200.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        assert result.risk_level == RiskLevel.HIGH
        # Should contain severity language
        assert any(word in result.explanation.lower() for word in ["significantly", "requires", "medical attention"])
    
    def test_medium_risk_explanation_contains_monitoring_language(self):
        """Test that MEDIUM risk explanations contain monitoring language"""
        evaluator = ThresholdEvaluator()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=110.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        assert result.risk_level == RiskLevel.MEDIUM
        # Should contain monitoring language
        assert any(word in result.explanation.lower() for word in ["consider", "follow-up", "healthcare provider"])
    
    def test_low_risk_explanation_confirms_normal_status(self):
        """Test that LOW risk explanations confirm normal status"""
        evaluator = ThresholdEvaluator()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=85.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        assert result.risk_level == RiskLevel.LOW
        # Should confirm normal status
        assert "within normal range" in result.explanation.lower()
    
    def test_edge_case_exactly_20_percent_deviation(self):
        """Test edge case where deviation is exactly 20%"""
        evaluator = ThresholdEvaluator()
        
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=120.0,  # Exactly 20% above max
            unit="mg/dL",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        # 20% should be MEDIUM (not HIGH)
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.deviation_percent == 20.0



# Property-Based Tests
from hypothesis import given, strategies as st, settings, assume
from backend.models.clinical_parameters import EvaluatedParameter


class TestThresholdEvaluatorProperties:
    """Property-based tests for ThresholdEvaluator"""
    
    @given(
        min_val=st.floats(min_value=50, max_value=100, allow_nan=False, allow_infinity=False),
        max_val=st.floats(min_value=100, max_value=200, allow_nan=False, allow_infinity=False),
        value=st.floats(min_value=100, max_value=200, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_within_range_risk_assignment(self, min_val, max_val, value):
        """
        Feature: threshold-based-risk-assessment, Property 8:
        For any parameter value that falls within its reference range (min ≤ value ≤ max),
        the Threshold_Evaluator should assign a risk level of LOW.
        
        Validates: Requirements 3.1
        """
        assume(min_val < max_val)
        assume(min_val <= value <= max_val)
        
        evaluator = ThresholdEvaluator()
        ref_range = ReferenceRange(min_value=min_val, max_value=max_val, unit="units")
        param = ClinicalParameter(
            name="Test Parameter",
            value=value,
            unit="units",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        assert result.risk_level == RiskLevel.LOW
        assert result.deviation_percent == 0.0
    
    @given(
        max_val=st.floats(min_value=100, max_value=200, allow_nan=False, allow_infinity=False),
        deviation_factor=st.floats(min_value=1.01, max_value=1.20)
    )
    @settings(max_examples=100)
    def test_property_medium_risk_assignment(self, max_val, deviation_factor):
        """
        Feature: threshold-based-risk-assessment, Property 9:
        For any parameter value that deviates from its reference range by 1-20%,
        the Threshold_Evaluator should assign a risk level of MEDIUM.
        
        Validates: Requirements 3.2
        """
        evaluator = ThresholdEvaluator()
        
        # Calculate value that is 1-20% above max
        value = max_val * deviation_factor
        
        ref_range = ReferenceRange(max_value=max_val, unit="units")
        param = ClinicalParameter(
            name="Test Parameter",
            value=value,
            unit="units",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        assert result.risk_level == RiskLevel.MEDIUM
        assert 0 < result.deviation_percent <= 20
    
    @given(
        max_val=st.floats(min_value=100, max_value=200, allow_nan=False, allow_infinity=False),
        deviation_factor=st.floats(min_value=1.21, max_value=2.0)
    )
    @settings(max_examples=100)
    def test_property_high_risk_assignment(self, max_val, deviation_factor):
        """
        Feature: threshold-based-risk-assessment, Property 10:
        For any parameter value that deviates from its reference range by more than 20%,
        the Threshold_Evaluator should assign a risk level of HIGH.
        
        Validates: Requirements 3.3
        """
        evaluator = ThresholdEvaluator()
        
        # Calculate value that is >20% above max
        value = max_val * deviation_factor
        
        ref_range = ReferenceRange(max_value=max_val, unit="units")
        param = ClinicalParameter(
            name="Test Parameter",
            value=value,
            unit="units",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        assert result.risk_level == RiskLevel.HIGH
        assert result.deviation_percent > 20
    
    @given(
        min_val=st.floats(min_value=50, max_value=100, allow_nan=False, allow_infinity=False),
        deviation_factor=st.floats(min_value=0.5, max_value=0.99)
    )
    @settings(max_examples=100)
    def test_property_negative_deviation_for_below_range(self, min_val, deviation_factor):
        """
        Feature: threshold-based-risk-assessment, Property 11:
        For any parameter value below its minimum reference value,
        the Threshold_Evaluator should calculate a negative deviation percentage.
        
        Validates: Requirements 3.4
        """
        assume(min_val > 0)  # Avoid division by zero
        
        evaluator = ThresholdEvaluator()
        
        # Calculate value that is below min
        value = min_val * deviation_factor
        
        ref_range = ReferenceRange(min_value=min_val, unit="units")
        param = ClinicalParameter(
            name="Test Parameter",
            value=value,
            unit="units",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        assert result.deviation_percent < 0
    
    @given(
        min_val=st.floats(min_value=50, max_value=100, allow_nan=False, allow_infinity=False),
        max_val=st.floats(min_value=100, max_value=200, allow_nan=False, allow_infinity=False),
        value=st.floats(min_value=0, max_value=300, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_deviation_calculation_completeness(self, min_val, max_val, value):
        """
        Feature: threshold-based-risk-assessment, Property 12:
        For any evaluated parameter, the result should include a deviation percentage
        (0 for within-range, positive for above-range, negative for below-range).
        
        Validates: Requirements 3.6
        """
        assume(min_val < max_val)
        
        evaluator = ThresholdEvaluator()
        ref_range = ReferenceRange(min_value=min_val, max_value=max_val, unit="units")
        param = ClinicalParameter(
            name="Test Parameter",
            value=value,
            unit="units",
            reference_range=ref_range
        )
        
        result = evaluator.evaluate_parameter(param)
        
        # Deviation should always be present
        assert result.deviation_percent is not None
        
        # Check deviation sign matches position relative to range
        if value < min_val:
            assert result.deviation_percent < 0
        elif value > max_val:
            assert result.deviation_percent > 0
        else:
            assert result.deviation_percent == 0.0
    
    @given(
        num_params=st.integers(min_value=1, max_value=10),
        base_value=st.floats(min_value=50, max_value=150, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_batch_parameter_evaluation(self, num_params, base_value):
        """
        Feature: threshold-based-risk-assessment, Property 13:
        For any list of clinical parameters, the Threshold_Evaluator should evaluate
        all parameters and return a list of evaluated results with the same length.
        
        Validates: Requirements 3.7
        """
        evaluator = ThresholdEvaluator()
        ref_range = ReferenceRange(min_value=70, max_value=100, unit="units")
        
        # Create list of parameters
        params = []
        for i in range(num_params):
            param = ClinicalParameter(
                name=f"Parameter {i}",
                value=base_value + i,
                unit="units",
                reference_range=ref_range
            )
            params.append(param)
        
        results = evaluator.evaluate_parameters(params)
        
        # Should return same number of results
        assert len(results) == num_params
        
        # All results should be EvaluatedParameter objects
        for result in results:
            assert isinstance(result, EvaluatedParameter)
            assert result.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
