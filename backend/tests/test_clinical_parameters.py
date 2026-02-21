"""
Unit tests for clinical parameter data models
"""

import pytest
from backend.models.clinical_parameters import (
    RiskLevel,
    ReferenceRange,
    ClinicalParameter,
    EvaluatedParameter
)


class TestRiskLevel:
    """Tests for RiskLevel enum"""
    
    def test_risk_level_values(self):
        """Test that RiskLevel enum has all required values"""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.UNKNOWN.value == "unknown"
    
    def test_risk_level_members(self):
        """Test that all expected risk levels are present"""
        risk_levels = [level.name for level in RiskLevel]
        assert "LOW" in risk_levels
        assert "MEDIUM" in risk_levels
        assert "HIGH" in risk_levels
        assert "UNKNOWN" in risk_levels


class TestReferenceRange:
    """Tests for ReferenceRange dataclass"""
    
    def test_reference_range_with_both_values(self):
        """Test creating reference range with both min and max values"""
        ref_range = ReferenceRange(min_value=70.0, max_value=100.0, unit="mg/dL")
        assert ref_range.min_value == 70.0
        assert ref_range.max_value == 100.0
        assert ref_range.unit == "mg/dL"
    
    def test_reference_range_with_min_only(self):
        """Test creating reference range with only min value (directional threshold)"""
        ref_range = ReferenceRange(min_value=40.0, unit="mg/dL")
        assert ref_range.min_value == 40.0
        assert ref_range.max_value is None
        assert ref_range.unit == "mg/dL"
    
    def test_reference_range_with_max_only(self):
        """Test creating reference range with only max value (directional threshold)"""
        ref_range = ReferenceRange(max_value=200.0, unit="mg/dL")
        assert ref_range.min_value is None
        assert ref_range.max_value == 200.0
        assert ref_range.unit == "mg/dL"
    
    def test_reference_range_validation_fails_without_thresholds(self):
        """Test that validation fails when neither min nor max is provided"""
        with pytest.raises(ValueError, match="At least one of min_value or max_value must be provided"):
            ReferenceRange(unit="mg/dL")
    
    def test_reference_range_to_dict(self):
        """Test serialization to dictionary"""
        ref_range = ReferenceRange(min_value=70.0, max_value=100.0, unit="mg/dL")
        result = ref_range.to_dict()
        assert isinstance(result, dict)
        assert result["min_value"] == 70.0
        assert result["max_value"] == 100.0
        assert result["unit"] == "mg/dL"


class TestClinicalParameter:
    """Tests for ClinicalParameter dataclass"""
    
    def test_clinical_parameter_creation(self):
        """Test creating a clinical parameter with all fields"""
        ref_range = ReferenceRange(min_value=70.0, max_value=100.0, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=85.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        assert param.name == "Blood Glucose"
        assert param.value == 85.0
        assert param.unit == "mg/dL"
        assert param.reference_range == ref_range
    
    def test_clinical_parameter_without_reference_range(self):
        """Test creating a clinical parameter without reference range"""
        param = ClinicalParameter(
            name="Blood Glucose",
            value=85.0,
            unit="mg/dL"
        )
        assert param.name == "Blood Glucose"
        assert param.value == 85.0
        assert param.unit == "mg/dL"
        assert param.reference_range is None
    
    def test_clinical_parameter_with_integer_value(self):
        """Test that integer values are accepted"""
        param = ClinicalParameter(
            name="Blood Glucose",
            value=85,
            unit="mg/dL"
        )
        assert param.value == 85
        assert isinstance(param.value, int)
    
    def test_clinical_parameter_validation_fails_with_string_value(self):
        """Test that validation fails when value is not numeric"""
        with pytest.raises(ValueError, match="Parameter value must be numeric"):
            ClinicalParameter(
                name="Blood Glucose",
                value="85.0",  # String instead of number
                unit="mg/dL"
            )
    
    def test_clinical_parameter_validation_fails_with_none_value(self):
        """Test that validation fails when value is None"""
        with pytest.raises(ValueError, match="Parameter value must be numeric"):
            ClinicalParameter(
                name="Blood Glucose",
                value=None,
                unit="mg/dL"
            )
    
    def test_clinical_parameter_to_dict_with_reference_range(self):
        """Test serialization to dictionary with reference range"""
        ref_range = ReferenceRange(min_value=70.0, max_value=100.0, unit="mg/dL")
        param = ClinicalParameter(
            name="Blood Glucose",
            value=85.0,
            unit="mg/dL",
            reference_range=ref_range
        )
        result = param.to_dict()
        assert isinstance(result, dict)
        assert result["name"] == "Blood Glucose"
        assert result["value"] == 85.0
        assert result["unit"] == "mg/dL"
        assert "reference_range" in result
        assert result["reference_range"]["min_value"] == 70.0
    
    def test_clinical_parameter_to_dict_without_reference_range(self):
        """Test serialization to dictionary without reference range"""
        param = ClinicalParameter(
            name="Blood Glucose",
            value=85.0,
            unit="mg/dL"
        )
        result = param.to_dict()
        assert isinstance(result, dict)
        assert result["name"] == "Blood Glucose"
        assert result["value"] == 85.0
        assert result["unit"] == "mg/dL"
        assert "reference_range" not in result


class TestEvaluatedParameter:
    """Tests for EvaluatedParameter dataclass"""
    
    def test_evaluated_parameter_creation(self):
        """Test creating an evaluated parameter"""
        ref_range = ReferenceRange(min_value=70.0, max_value=100.0, unit="mg/dL")
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
            explanation="Blood Glucose is significantly above normal range."
        )
        assert evaluated.parameter == param
        assert evaluated.risk_level == RiskLevel.HIGH
        assert evaluated.deviation_percent == 50.0
        assert "significantly above" in evaluated.explanation
    
    def test_evaluated_parameter_with_none_deviation(self):
        """Test evaluated parameter with None deviation (for UNKNOWN risk)"""
        param = ClinicalParameter(
            name="Unknown Parameter",
            value=100.0,
            unit="units"
        )
        evaluated = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.UNKNOWN,
            deviation_percent=None,
            explanation="No reference range available."
        )
        assert evaluated.risk_level == RiskLevel.UNKNOWN
        assert evaluated.deviation_percent is None
    
    def test_evaluated_parameter_to_dict(self):
        """Test serialization to dictionary"""
        ref_range = ReferenceRange(min_value=70.0, max_value=100.0, unit="mg/dL")
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
            explanation="Blood Glucose is significantly above normal range."
        )
        result = evaluated.to_dict()
        assert isinstance(result, dict)
        assert "parameter" in result
        assert result["parameter"]["name"] == "Blood Glucose"
        assert result["risk_level"] == "high"
        assert result["deviation_percent"] == 50.0
        assert result["explanation"] == "Blood Glucose is significantly above normal range."
    
    def test_evaluated_parameter_all_risk_levels(self):
        """Test evaluated parameters with all risk levels"""
        param = ClinicalParameter(name="Test", value=100.0, unit="units")
        
        # Test LOW risk
        low_eval = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.LOW,
            deviation_percent=0.0,
            explanation="Within normal range"
        )
        assert low_eval.risk_level == RiskLevel.LOW
        assert low_eval.to_dict()["risk_level"] == "low"
        
        # Test MEDIUM risk
        medium_eval = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.MEDIUM,
            deviation_percent=15.0,
            explanation="Slightly elevated"
        )
        assert medium_eval.risk_level == RiskLevel.MEDIUM
        assert medium_eval.to_dict()["risk_level"] == "medium"
        
        # Test HIGH risk
        high_eval = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.HIGH,
            deviation_percent=50.0,
            explanation="Significantly elevated"
        )
        assert high_eval.risk_level == RiskLevel.HIGH
        assert high_eval.to_dict()["risk_level"] == "high"
        
        # Test UNKNOWN risk
        unknown_eval = EvaluatedParameter(
            parameter=param,
            risk_level=RiskLevel.UNKNOWN,
            deviation_percent=None,
            explanation="No reference range"
        )
        assert unknown_eval.risk_level == RiskLevel.UNKNOWN
        assert unknown_eval.to_dict()["risk_level"] == "unknown"
