"""
Unit tests for parameter extractor
"""

import pytest
from backend.services.parameter_extractor import ParameterExtractor


class TestParameterExtractor:
    """Tests for ParameterExtractor class"""
    
    def test_extract_single_parameter_with_unit(self):
        """Test extraction of single parameter with unit"""
        extractor = ParameterExtractor()
        text = "Blood Glucose: 85.0 mg/dL"
        
        params = extractor.extract_parameters(text)
        assert len(params) == 1
        assert params[0]["name"] == "Blood Glucose"
        assert params[0]["value"] == 85.0
        assert params[0]["unit"] == "mg/dL"
    
    def test_extract_multiple_parameters(self):
        """Test extraction of multiple parameters from text"""
        extractor = ParameterExtractor()
        text = """
        Blood Glucose: 120 mg/dL
        Total Cholesterol: 220 mg/dL
        Hemoglobin: 14.5 g/dL
        """
        
        params = extractor.extract_parameters(text)
        assert len(params) == 3
        
        # Check each parameter
        param_names = [p["name"] for p in params]
        assert "Blood Glucose" in param_names
        assert "Total Cholesterol" in param_names
        assert "Hemoglobin" in param_names
    
    def test_extract_blood_pressure_systolic_diastolic(self):
        """Test extraction of blood pressure with systolic/diastolic components"""
        extractor = ParameterExtractor()
        text = "Blood Pressure: 120/80 mmHg"
        
        params = extractor.extract_parameters(text)
        assert len(params) == 2
        
        # Check systolic
        systolic = next(p for p in params if p["name"] == "Systolic Blood Pressure")
        assert systolic["value"] == 120
        assert systolic["unit"] == "mmHg"
        
        # Check diastolic
        diastolic = next(p for p in params if p["name"] == "Diastolic Blood Pressure")
        assert diastolic["value"] == 80
        assert diastolic["unit"] == "mmHg"
    
    def test_extract_blood_pressure_variations(self):
        """Test extraction of blood pressure with various naming conventions"""
        extractor = ParameterExtractor()
        
        # Test "BP" abbreviation
        text1 = "BP: 130/85 mmHg"
        params1 = extractor.extract_parameters(text1)
        assert len(params1) == 2
        assert any(p["name"] == "Systolic Blood Pressure" and p["value"] == 130 for p in params1)
        
        # Test "B.P." abbreviation
        text2 = "B.P.: 140/90"
        params2 = extractor.extract_parameters(text2)
        assert len(params2) == 2
        assert any(p["name"] == "Diastolic Blood Pressure" and p["value"] == 90 for p in params2)
    
    def test_unit_inference_for_missing_units(self):
        """Test that missing units are inferred based on parameter type"""
        extractor = ParameterExtractor()
        text = """
        Blood Glucose: 95
        Total Cholesterol: 180
        Hemoglobin: 15.2
        """
        
        params = extractor.extract_parameters(text)
        
        # Check inferred units
        glucose = next(p for p in params if p["name"] == "Blood Glucose")
        assert glucose["unit"] == "mg/dL"
        
        cholesterol = next(p for p in params if p["name"] == "Total Cholesterol")
        assert cholesterol["unit"] == "mg/dL"
        
        hemoglobin = next(p for p in params if p["name"] == "Hemoglobin")
        assert hemoglobin["unit"] == "g/dL"
    
    def test_parameter_name_normalization(self):
        """Test that parameter name variations are normalized"""
        extractor = ParameterExtractor()
        
        # Test glucose variations
        text1 = "Blood Sugar: 100 mg/dL"
        params1 = extractor.extract_parameters(text1)
        assert params1[0]["name"] == "Blood Glucose"
        
        text2 = "FBS: 95 mg/dL"
        params2 = extractor.extract_parameters(text2)
        assert params2[0]["name"] == "Blood Glucose"
        
        # Test hemoglobin variations
        text3 = "Hb: 14.0 g/dL"
        params3 = extractor.extract_parameters(text3)
        assert params3[0]["name"] == "Hemoglobin"
    
    def test_handling_invalid_numeric_values(self):
        """Test that invalid numeric values are skipped"""
        extractor = ParameterExtractor()
        text = """
        Blood Glucose: invalid mg/dL
        Total Cholesterol: 200 mg/dL
        """
        
        params = extractor.extract_parameters(text)
        # Should only extract the valid parameter
        assert len(params) == 1
        assert params[0]["name"] == "Total Cholesterol"
    
    def test_empty_text_returns_empty_list(self):
        """Test that empty text returns empty list"""
        extractor = ParameterExtractor()
        
        assert extractor.extract_parameters("") == []
        assert extractor.extract_parameters("   ") == []
        assert extractor.extract_parameters(None) == []
    
    def test_extract_all_supported_parameters(self):
        """Test extraction of all 10+ supported parameters"""
        extractor = ParameterExtractor()
        text = """
        Blood Pressure: 120/80 mmHg
        Blood Glucose: 95 mg/dL
        Total Cholesterol: 190 mg/dL
        LDL Cholesterol: 110 mg/dL
        HDL Cholesterol: 55 mg/dL
        Triglycerides: 140 mg/dL
        Hemoglobin: 15.0 g/dL
        WBC: 7.5 ×10³/μL
        Platelet: 250 ×10³/μL
        Creatinine: 1.0 mg/dL
        """
        
        params = extractor.extract_parameters(text)
        # Blood pressure counts as 2 (systolic + diastolic)
        assert len(params) >= 11
        
        param_names = [p["name"] for p in params]
        assert "Systolic Blood Pressure" in param_names
        assert "Diastolic Blood Pressure" in param_names
        assert "Blood Glucose" in param_names
        assert "Total Cholesterol" in param_names
        assert "LDL Cholesterol" in param_names
        assert "HDL Cholesterol" in param_names
        assert "Triglycerides" in param_names
        assert "Hemoglobin" in param_names
        assert "WBC" in param_names
        assert "Platelet" in param_names
        assert "Creatinine" in param_names
    
    def test_extract_parameters_with_decimal_values(self):
        """Test extraction of parameters with decimal values"""
        extractor = ParameterExtractor()
        text = """
        Blood Glucose: 95.5 mg/dL
        Hemoglobin: 14.2 g/dL
        Creatinine: 0.9 mg/dL
        """
        
        params = extractor.extract_parameters(text)
        assert len(params) == 3
        
        glucose = next(p for p in params if p["name"] == "Blood Glucose")
        assert glucose["value"] == 95.5
        
        hemoglobin = next(p for p in params if p["name"] == "Hemoglobin")
        assert hemoglobin["value"] == 14.2
        
        creatinine = next(p for p in params if p["name"] == "Creatinine")
        assert creatinine["value"] == 0.9
    
    def test_extract_parameters_case_insensitive(self):
        """Test that extraction is case-insensitive"""
        extractor = ParameterExtractor()
        
        text1 = "blood glucose: 100 mg/dL"
        params1 = extractor.extract_parameters(text1)
        assert len(params1) == 1
        assert params1[0]["name"] == "Blood Glucose"
        
        text2 = "HEMOGLOBIN: 15.0 g/dL"
        params2 = extractor.extract_parameters(text2)
        assert len(params2) == 1
        assert params2[0]["name"] == "Hemoglobin"
    
    def test_extract_wbc_variations(self):
        """Test extraction of WBC with various naming conventions"""
        extractor = ParameterExtractor()
        
        text1 = "WBC: 8.0 ×10³/μL"
        params1 = extractor.extract_parameters(text1)
        assert params1[0]["name"] == "WBC"
        
        text2 = "White Blood Cell Count: 7.5"
        params2 = extractor.extract_parameters(text2)
        assert params2[0]["name"] == "WBC"
        
        text3 = "Leukocytes: 9.0"
        params3 = extractor.extract_parameters(text3)
        assert params3[0]["name"] == "WBC"
    
    def test_extract_platelet_variations(self):
        """Test extraction of platelet with various naming conventions"""
        extractor = ParameterExtractor()
        
        text1 = "Platelet: 250 ×10³/μL"
        params1 = extractor.extract_parameters(text1)
        assert params1[0]["name"] == "Platelet"
        
        text2 = "Platelet Count: 300"
        params2 = extractor.extract_parameters(text2)
        assert params2[0]["name"] == "Platelet"
        
        text3 = "PLT: 280"
        params3 = extractor.extract_parameters(text3)
        assert params3[0]["name"] == "Platelet"
    
    def test_extract_parameters_with_spaces_in_format(self):
        """Test extraction with various spacing formats"""
        extractor = ParameterExtractor()
        
        # Test with extra spaces
        text1 = "Blood Glucose:    120   mg/dL"
        params1 = extractor.extract_parameters(text1)
        assert params1[0]["value"] == 120
        
        # Test with no space after colon
        text2 = "Hemoglobin:15.0 g/dL"
        params2 = extractor.extract_parameters(text2)
        assert params2[0]["value"] == 15.0
        
        # Test blood pressure with spaces
        text3 = "BP: 130 / 85 mmHg"
        params3 = extractor.extract_parameters(text3)
        assert len(params3) == 2
        assert any(p["value"] == 130 for p in params3)
        assert any(p["value"] == 85 for p in params3)



# Property-Based Tests
from hypothesis import given, strategies as st, settings


class TestParameterExtractorProperties:
    """Property-based tests for ParameterExtractor"""
    
    @given(
        glucose_value=st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False),
        cholesterol_value=st.floats(min_value=0, max_value=400, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_complete_parameter_extraction(self, glucose_value, cholesterol_value):
        """
        Feature: threshold-based-risk-assessment, Property 1:
        For any clinical text containing valid parameter patterns (name, numeric value, optional unit),
        the Parameter_Extractor should extract all identifiable parameters with their complete information.
        
        Validates: Requirements 1.1, 1.2
        """
        extractor = ParameterExtractor()
        
        # Create text with two parameters
        text = f"Blood Glucose: {glucose_value:.1f} mg/dL\nTotal Cholesterol: {cholesterol_value:.1f} mg/dL"
        
        params = extractor.extract_parameters(text)
        
        # Should extract both parameters
        assert len(params) == 2
        
        # Check that all parameters have complete information
        for param in params:
            assert "name" in param
            assert "value" in param
            assert "unit" in param
            assert param["name"] in ["Blood Glucose", "Total Cholesterol"]
            assert isinstance(param["value"], (int, float))
            assert param["unit"] != ""
    
    @given(
        valid_glucose=st.floats(min_value=50, max_value=300, allow_nan=False, allow_infinity=False),
        systolic=st.integers(min_value=80, max_value=200),
        diastolic=st.integers(min_value=50, max_value=120)
    )
    @settings(max_examples=100)
    def test_property_robust_parameter_extraction(self, valid_glucose, systolic, diastolic):
        """
        Feature: threshold-based-risk-assessment, Property 2:
        For any clinical text containing a mix of valid and invalid parameters,
        the Parameter_Extractor should extract all valid parameters and skip invalid ones without failing.
        
        Validates: Requirements 1.4, 12.1
        """
        extractor = ParameterExtractor()
        
        # Create text with valid and invalid parameters
        text = f"""
        Blood Glucose: {valid_glucose:.1f} mg/dL
        Invalid Parameter: not_a_number mg/dL
        Blood Pressure: {systolic}/{diastolic} mmHg
        Another Invalid: xyz units
        """
        
        # Should not raise an exception
        params = extractor.extract_parameters(text)
        
        # Should extract only valid parameters (glucose + BP systolic + BP diastolic = 3)
        assert len(params) == 3
        
        # All extracted parameters should have valid numeric values
        for param in params:
            assert isinstance(param["value"], (int, float))
            assert not (param["value"] != param["value"])  # Check for NaN
    
    @given(
        systolic=st.integers(min_value=80, max_value=200),
        diastolic=st.integers(min_value=50, max_value=120)
    )
    @settings(max_examples=100)
    def test_property_blood_pressure_component_extraction(self, systolic, diastolic):
        """
        Feature: threshold-based-risk-assessment, Property 3:
        For any blood pressure value in the format "X/Y mmHg",
        the Parameter_Extractor should extract both systolic and diastolic components as separate parameters.
        
        Validates: Requirements 1.3
        """
        extractor = ParameterExtractor()
        
        text = f"Blood Pressure: {systolic}/{diastolic} mmHg"
        
        params = extractor.extract_parameters(text)
        
        # Should extract exactly 2 parameters (systolic and diastolic)
        assert len(params) == 2
        
        # Check systolic
        systolic_param = next((p for p in params if p["name"] == "Systolic Blood Pressure"), None)
        assert systolic_param is not None
        assert systolic_param["value"] == systolic
        assert systolic_param["unit"] == "mmHg"
        
        # Check diastolic
        diastolic_param = next((p for p in params if p["name"] == "Diastolic Blood Pressure"), None)
        assert diastolic_param is not None
        assert diastolic_param["value"] == diastolic
        assert diastolic_param["unit"] == "mmHg"
    
    @given(
        glucose_value=st.floats(min_value=50, max_value=300, allow_nan=False, allow_infinity=False),
        cholesterol_value=st.floats(min_value=100, max_value=400, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_unit_inference(self, glucose_value, cholesterol_value):
        """
        Feature: threshold-based-risk-assessment, Property 4:
        For any extracted parameter missing a unit,
        the Parameter_Extractor should assign the standard unit for that parameter type.
        
        Validates: Requirements 1.6
        """
        extractor = ParameterExtractor()
        
        # Create text without units
        text = f"Blood Glucose: {glucose_value:.1f}\nTotal Cholesterol: {cholesterol_value:.1f}"
        
        params = extractor.extract_parameters(text)
        
        # Should extract both parameters
        assert len(params) == 2
        
        # Check that units were inferred
        glucose_param = next((p for p in params if p["name"] == "Blood Glucose"), None)
        assert glucose_param is not None
        assert glucose_param["unit"] == "mg/dL"  # Standard unit for glucose
        
        cholesterol_param = next((p for p in params if p["name"] == "Total Cholesterol"), None)
        assert cholesterol_param is not None
        assert cholesterol_param["unit"] == "mg/dL"  # Standard unit for cholesterol
    
    @given(
        glucose_value=st.floats(min_value=50, max_value=300, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_parameter_name_normalization(self, glucose_value):
        """
        Feature: threshold-based-risk-assessment, Property 5:
        For any parameter name variation (e.g., "BP", "Blood Pressure", "Systolic BP"),
        the Parameter_Extractor should normalize it to the standard parameter name.
        
        Validates: Requirements 1.7
        """
        extractor = ParameterExtractor()
        
        # Test with different name variations
        variations = [
            f"Blood Glucose: {glucose_value:.1f} mg/dL",
            f"Blood Sugar: {glucose_value:.1f} mg/dL",
            f"FBS: {glucose_value:.1f} mg/dL"
        ]
        
        for text in variations:
            params = extractor.extract_parameters(text)
            
            # Should extract one parameter
            assert len(params) == 1
            
            # Should normalize to standard name
            assert params[0]["name"] == "Blood Glucose"
            assert abs(params[0]["value"] - glucose_value) < 0.1
