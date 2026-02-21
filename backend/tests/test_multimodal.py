"""
Test suite for multi-modal analysis functionality.

This module tests the integration of lab report and medical image analysis,
including correlation logic and integrated diagnosis generation.

Requirements tested:
- 8.1: Multi-modal analysis with both report and image
- 8.2: Multi-modal analysis with only report
- 8.3: Multi-modal analysis with only image
- 8.4: Correlation detection between lab and imaging findings
- 8.5: Error handling when one analysis component fails
- 8.6: Integrated diagnosis generation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

# Import the correlation function from main.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.main import _correlate_findings
from backend.services.analysis_engine import AnalysisEngine, AnalysisResult, RiskIndicator
from backend.services.vision_analyzer import MedicalImageAnalyzer


class TestMultiModalAnalysis:
    """Test suite for multi-modal analysis functionality."""
    
    @pytest.fixture
    def mock_report_analysis(self):
        """Mock lab report analysis result."""
        return {
            "summary": "Lab results show elevated glucose and cholesterol levels",
            "key_findings": [
                "Blood Glucose: 250 mg/dL (High)",
                "Total Cholesterol: 280 mg/dL (High)",
                "WBC: 12.5 ×10³/μL (Elevated)"
            ],
            "risk_indicators": [
                {
                    "finding": "Blood Glucose: 250 mg/dL",
                    "category": "metabolic",
                    "severity": "high",
                    "description": "Significantly elevated glucose level"
                },
                {
                    "finding": "Total Cholesterol: 280 mg/dL",
                    "category": "cardiovascular",
                    "severity": "medium",
                    "description": "Elevated cholesterol level"
                },
                {
                    "finding": "WBC: 12.5 ×10³/μL",
                    "category": "hematologic",
                    "severity": "medium",
                    "description": "Elevated white blood cell count"
                }
            ],
            "follow_up_suggestions": [
                "Consult with endocrinologist",
                "Monitor glucose levels closely"
            ],
            "medical_disclaimer": "This is for informational purposes only",
            "analysis_timestamp": "2024-01-01T00:00:00"
        }
    
    @pytest.fixture
    def mock_image_analysis(self):
        """Mock medical image analysis result."""
        return {
            "success": True,
            "diagnosis": "Chest X-ray shows possible infiltrate in right lower lobe",
            "issues": [
                "Opacity in right lower lobe",
                "Possible infiltrate",
                "Mild cardiomegaly"
            ],
            "suggestions": [
                "Consider CT scan for detailed evaluation",
                "Follow-up imaging in 2-4 weeks"
            ],
            "confidence": 85,
            "urgency": "medium"
        }
    
    @pytest.fixture
    def mock_image_analysis_retinal(self):
        """Mock retinal image analysis result."""
        return {
            "success": True,
            "diagnosis": "Retinal examination shows microaneurysms and hemorrhages",
            "issues": [
                "Retinal microaneurysms",
                "Dot and blot hemorrhages",
                "Hard exudates present"
            ],
            "suggestions": [
                "Ophthalmology referral recommended",
                "Diabetic retinopathy screening"
            ],
            "confidence": 90,
            "urgency": "high"
        }
    
    @pytest.fixture
    def mock_report_analysis_normal(self):
        """Mock normal lab report analysis result."""
        return {
            "summary": "All lab results within normal limits",
            "key_findings": [
                "Blood Glucose: 95 mg/dL (Normal)",
                "Total Cholesterol: 180 mg/dL (Normal)"
            ],
            "risk_indicators": [],
            "follow_up_suggestions": [
                "Continue routine healthcare maintenance"
            ],
            "medical_disclaimer": "This is for informational purposes only",
            "analysis_timestamp": "2024-01-01T00:00:00"
        }
    
    @pytest.fixture
    def mock_image_analysis_normal(self):
        """Mock normal medical image analysis result."""
        return {
            "success": True,
            "diagnosis": "Chest X-ray appears normal",
            "issues": [],
            "suggestions": [
                "No immediate follow-up required"
            ],
            "confidence": 95,
            "urgency": "low"
        }


    # Test 3.2: Multi-modal analysis scenarios
    
    def test_multimodal_with_both_report_and_image(self, mock_report_analysis, mock_image_analysis):
        """
        Test multi-modal analysis with both report and image provided.
        
        Requirements: 8.1
        """
        # Call correlation function with both inputs
        correlation = _correlate_findings(mock_report_analysis, mock_image_analysis)
        
        # Verify correlation result structure
        assert "integrated_diagnosis" in correlation
        assert "correlations" in correlation
        assert "confidence" in correlation
        assert "recommendations" in correlation
        
        # Verify correlation is a dictionary
        assert isinstance(correlation, dict)
        assert isinstance(correlation["correlations"], list)
        assert isinstance(correlation["recommendations"], list)
        
        # Verify confidence is a number
        assert isinstance(correlation["confidence"], (int, float))
        assert 0 <= correlation["confidence"] <= 100
    
    def test_multimodal_with_only_report(self, mock_report_analysis):
        """
        Test multi-modal analysis with only report provided.
        
        Requirements: 8.2
        """
        # Create empty image analysis
        empty_image_analysis = {
            "success": True,
            "diagnosis": "",
            "issues": [],
            "suggestions": [],
            "confidence": 0,
            "urgency": "low"
        }
        
        # Call correlation function with only report
        correlation = _correlate_findings(mock_report_analysis, empty_image_analysis)
        
        # Verify correlation result structure exists
        assert "integrated_diagnosis" in correlation
        assert "correlations" in correlation
        assert "confidence" in correlation
        assert "recommendations" in correlation
        
        # With no image findings, should have no correlations
        assert len(correlation["correlations"]) == 0
    
    def test_multimodal_with_only_image(self, mock_image_analysis):
        """
        Test multi-modal analysis with only image provided.
        
        Requirements: 8.3
        """
        # Create empty report analysis
        empty_report_analysis = {
            "summary": "",
            "key_findings": [],
            "risk_indicators": [],
            "follow_up_suggestions": [],
            "medical_disclaimer": "",
            "analysis_timestamp": "2024-01-01T00:00:00"
        }
        
        # Call correlation function with only image
        correlation = _correlate_findings(empty_report_analysis, mock_image_analysis)
        
        # Verify correlation result structure exists
        assert "integrated_diagnosis" in correlation
        assert "correlations" in correlation
        assert "confidence" in correlation
        assert "recommendations" in correlation
        
        # With no report findings, should have no correlations
        assert len(correlation["correlations"]) == 0
    
    def test_multimodal_with_no_findings(self, mock_report_analysis_normal, mock_image_analysis_normal):
        """
        Test multi-modal analysis when both report and image are normal.
        
        Requirements: 8.1
        """
        # Call correlation function with normal results
        correlation = _correlate_findings(mock_report_analysis_normal, mock_image_analysis_normal)
        
        # Verify correlation result structure
        assert "integrated_diagnosis" in correlation
        assert "correlations" in correlation
        assert "confidence" in correlation
        assert "recommendations" in correlation
        
        # With no abnormal findings, should have no correlations
        assert len(correlation["correlations"]) == 0
        
        # Should have a message about no correlations
        assert "no" in correlation["integrated_diagnosis"].lower() or \
               "not found" in correlation["integrated_diagnosis"].lower()


    # Test 3.3: Correlation logic
    
    def test_correlation_detection_glucose_retinal(self, mock_report_analysis, mock_image_analysis_retinal):
        """
        Test correlation detection between high glucose and retinal changes.
        
        Requirements: 8.4
        """
        # Call correlation function with glucose and retinal findings
        correlation = _correlate_findings(mock_report_analysis, mock_image_analysis_retinal)
        
        # Verify correlations were found
        assert len(correlation["correlations"]) > 0
        
        # Verify correlation contains expected information
        found_diabetic_correlation = False
        for corr in correlation["correlations"]:
            if "diabetic" in corr.get("diagnosis", "").lower() or \
               "retinal" in corr.get("diagnosis", "").lower():
                found_diabetic_correlation = True
                assert "confidence" in corr
                assert corr["confidence"] > 0
        
        assert found_diabetic_correlation, "Expected diabetic retinopathy correlation not found"
    
    def test_correlation_detection_wbc_infiltrate(self):
        """
        Test correlation detection between elevated WBC and lung infiltrate.
        
        Requirements: 8.4
        """
        # Create report with elevated WBC
        report_with_wbc = {
            "summary": "Elevated white blood cell count",
            "key_findings": ["WBC: 15.0 ×10³/μL (Elevated)"],
            "risk_indicators": [
                {
                    "finding": "WBC: 15.0 ×10³/μL",
                    "category": "hematologic",
                    "severity": "medium",
                    "description": "Elevated white blood cell count"
                }
            ],
            "follow_up_suggestions": [],
            "medical_disclaimer": "",
            "analysis_timestamp": "2024-01-01T00:00:00"
        }
        
        # Create image with infiltrate
        image_with_infiltrate = {
            "success": True,
            "diagnosis": "Chest X-ray shows infiltrate in right lung",
            "issues": [
                "Infiltrate in right lower lobe",
                "Possible pneumonia"
            ],
            "suggestions": ["Consider antibiotic therapy"],
            "confidence": 85,
            "urgency": "high"
        }
        
        # Call correlation function
        correlation = _correlate_findings(report_with_wbc, image_with_infiltrate)
        
        # Verify correlations were found
        assert len(correlation["correlations"]) > 0
        
        # Verify correlation mentions infection or pneumonia
        found_infection_correlation = False
        for corr in correlation["correlations"]:
            diagnosis_lower = corr.get("diagnosis", "").lower()
            if "infection" in diagnosis_lower or "pneumonia" in diagnosis_lower or "wbc" in diagnosis_lower:
                found_infection_correlation = True
                assert corr["confidence"] > 0
        
        assert found_infection_correlation, "Expected infection correlation not found"
    
    def test_correlation_detection_cholesterol_heart(self):
        """
        Test correlation detection between high cholesterol and cardiac findings.
        
        Requirements: 8.4
        """
        # Create report with high cholesterol
        report_with_cholesterol = {
            "summary": "Elevated cholesterol levels",
            "key_findings": ["Total Cholesterol: 280 mg/dL (High)"],
            "risk_indicators": [
                {
                    "finding": "Total Cholesterol: 280 mg/dL",
                    "category": "cardiovascular",
                    "severity": "high",
                    "description": "Significantly elevated cholesterol"
                }
            ],
            "follow_up_suggestions": [],
            "medical_disclaimer": "",
            "analysis_timestamp": "2024-01-01T00:00:00"
        }
        
        # Create image with cardiac findings
        image_with_heart = {
            "success": True,
            "diagnosis": "Chest X-ray shows cardiomegaly",
            "issues": [
                "Enlarged heart shadow",
                "Cardiomegaly present"
            ],
            "suggestions": ["Echocardiogram recommended"],
            "confidence": 90,
            "urgency": "medium"
        }
        
        # Call correlation function
        correlation = _correlate_findings(report_with_cholesterol, image_with_heart)
        
        # Verify correlations were found
        assert len(correlation["correlations"]) > 0
        
        # Verify correlation mentions cardiovascular risk
        found_cardiac_correlation = False
        for corr in correlation["correlations"]:
            diagnosis_lower = corr.get("diagnosis", "").lower()
            if "cardiovascular" in diagnosis_lower or "heart" in diagnosis_lower or "cholesterol" in diagnosis_lower:
                found_cardiac_correlation = True
                assert corr["confidence"] > 0
        
        assert found_cardiac_correlation, "Expected cardiovascular correlation not found"
    
    def test_integrated_diagnosis_generation(self, mock_report_analysis, mock_image_analysis_retinal):
        """
        Test that integrated diagnosis is generated when correlations are found.
        
        Requirements: 8.6
        """
        # Call correlation function
        correlation = _correlate_findings(mock_report_analysis, mock_image_analysis_retinal)
        
        # Verify integrated diagnosis is present and not empty
        assert "integrated_diagnosis" in correlation
        assert correlation["integrated_diagnosis"]
        assert len(correlation["integrated_diagnosis"]) > 0
        
        # Verify confidence is higher when correlations are found
        if len(correlation["correlations"]) > 0:
            assert correlation["confidence"] > 50
        
        # Verify recommendations are provided
        assert "recommendations" in correlation
        assert len(correlation["recommendations"]) > 0
    
    def test_no_correlation_message(self, mock_report_analysis_normal, mock_image_analysis_normal):
        """
        Test that appropriate message is shown when no correlations are found.
        
        Requirements: 8.6
        """
        # Call correlation function with normal results
        correlation = _correlate_findings(mock_report_analysis_normal, mock_image_analysis_normal)
        
        # Verify integrated diagnosis mentions no correlations
        assert "integrated_diagnosis" in correlation
        diagnosis_lower = correlation["integrated_diagnosis"].lower()
        assert "no" in diagnosis_lower or "not found" in diagnosis_lower
        
        # Verify confidence is lower when no correlations
        assert correlation["confidence"] <= 50
        
        # Verify recommendations are still provided
        assert "recommendations" in correlation
        assert len(correlation["recommendations"]) > 0


    # Test 3.4: Error handling
    
    def test_error_handling_missing_report_fields(self, mock_image_analysis):
        """
        Test error handling when report analysis has missing fields.
        
        Requirements: 8.5
        """
        # Create report with missing fields
        incomplete_report = {
            "summary": "Incomplete report"
            # Missing risk_indicators, key_findings, etc.
        }
        
        # Call correlation function - should not crash
        correlation = _correlate_findings(incomplete_report, mock_image_analysis)
        
        # Verify correlation result structure is still valid
        assert "integrated_diagnosis" in correlation
        assert "correlations" in correlation
        assert "confidence" in correlation
        assert "recommendations" in correlation
        
        # Should handle gracefully with no correlations
        assert isinstance(correlation["correlations"], list)
    
    def test_error_handling_missing_image_fields(self, mock_report_analysis):
        """
        Test error handling when image analysis has missing fields.
        
        Requirements: 8.5
        """
        # Create image with missing fields
        incomplete_image = {
            "success": True,
            "diagnosis": "Incomplete image analysis"
            # Missing issues, suggestions, etc.
        }
        
        # Call correlation function - should not crash
        correlation = _correlate_findings(mock_report_analysis, incomplete_image)
        
        # Verify correlation result structure is still valid
        assert "integrated_diagnosis" in correlation
        assert "correlations" in correlation
        assert "confidence" in correlation
        assert "recommendations" in correlation
        
        # Should handle gracefully with no correlations
        assert isinstance(correlation["correlations"], list)
    
    def test_error_handling_none_values(self):
        """
        Test error handling when None values are provided.
        
        Requirements: 8.5
        """
        # Create report and image with None in lists
        report_with_none = {
            "summary": "Test report",
            "key_findings": [],
            "risk_indicators": [None, {"finding": "test"}],
            "follow_up_suggestions": [],
            "medical_disclaimer": "",
            "analysis_timestamp": "2024-01-01T00:00:00"
        }
        
        image_with_none = {
            "success": True,
            "diagnosis": "Test image",
            "issues": [None, "test issue"],
            "suggestions": [],
            "confidence": 80,
            "urgency": "low"
        }
        
        # Call correlation function - should not crash
        correlation = _correlate_findings(report_with_none, image_with_none)
        
        # Verify correlation result structure is still valid
        assert "integrated_diagnosis" in correlation
        assert "correlations" in correlation
        assert "confidence" in correlation
        assert "recommendations" in correlation
    
    def test_error_handling_empty_strings(self):
        """
        Test error handling when empty strings are provided.
        
        Requirements: 8.5
        """
        # Create report and image with empty strings
        report_empty = {
            "summary": "",
            "key_findings": [""],
            "risk_indicators": [],
            "follow_up_suggestions": [],
            "medical_disclaimer": "",
            "analysis_timestamp": ""
        }
        
        image_empty = {
            "success": True,
            "diagnosis": "",
            "issues": [""],
            "suggestions": [],
            "confidence": 0,
            "urgency": ""
        }
        
        # Call correlation function - should not crash
        correlation = _correlate_findings(report_empty, image_empty)
        
        # Verify correlation result structure is still valid
        assert "integrated_diagnosis" in correlation
        assert "correlations" in correlation
        assert "confidence" in correlation
        assert "recommendations" in correlation
        
        # Should have no correlations with empty data
        assert len(correlation["correlations"]) == 0
    
    def test_correlation_function_exception_handling(self):
        """
        Test that correlation function handles exceptions gracefully.
        
        Requirements: 8.5
        """
        # Create malformed data that might cause exceptions
        malformed_report = {
            "risk_indicators": "not a list"  # Should be a list
        }
        
        malformed_image = {
            "issues": 123  # Should be a list
        }
        
        # Call correlation function - should not crash
        try:
            correlation = _correlate_findings(malformed_report, malformed_image)
            
            # If it doesn't crash, verify it returns valid structure
            assert "integrated_diagnosis" in correlation
            assert "correlations" in correlation
            assert "confidence" in correlation
            assert "recommendations" in correlation
        except Exception as e:
            # If it does raise an exception, it should be caught and handled
            # The function should ideally not raise exceptions
            pytest.fail(f"Correlation function should handle errors gracefully, but raised: {e}")
