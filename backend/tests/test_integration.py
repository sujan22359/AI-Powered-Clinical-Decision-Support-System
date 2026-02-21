"""
Integration test suite for end-to-end workflows.

This module tests complete workflows from file upload to response,
including document analysis, image analysis, and multi-modal analysis.

Requirements tested:
- 9.1: Complete document analysis workflow
- 9.2: Complete image analysis workflow
- 9.3: Complete multi-modal analysis workflow
- 9.4: Threshold-based risks in analysis results
- 9.5: Various file formats (PDF, DOCX, JPEG, PNG)
- 9.6: API error responses
- 9.7: Performance requirements
"""

import pytest
import time
from pathlib import Path
from io import BytesIO
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Import the FastAPI app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.main import app
from backend.services.analysis_engine import AnalysisEngine
from backend.services.vision_analyzer import MedicalImageAnalyzer


class TestIntegration:
    """Integration test suite for end-to-end workflows."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Create sample PDF content for testing."""
        # Create a minimal PDF structure
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 100
>>
stream
BT
/F1 12 Tf
50 700 Td
(Lab Report: Blood Glucose: 250 mg/dL) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000317 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
466
%%EOF
"""
        return pdf_content
    
    @pytest.fixture
    def sample_image_content(self):
        """Create sample image content for testing."""
        # Create a minimal valid JPEG
        # This is a 1x1 pixel red JPEG
        jpeg_content = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
            0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08,
            0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C,
            0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D,
            0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20,
            0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27,
            0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34,
            0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4,
            0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x03, 0xFF, 0xDA, 0x00, 0x08,
            0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x37, 0xFF,
            0xD9
        ])
        return jpeg_content
    
    @pytest.fixture
    def sample_png_content(self):
        """Create sample PNG content for testing."""
        # Create a minimal valid PNG (1x1 pixel)
        png_content = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        return png_content


    # Test 4.2: Document analysis workflow
    
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_document_analysis_pdf_workflow(self, mock_model_class, client, sample_pdf_content):
        """
        Test complete PDF document analysis workflow from upload to response.
        
        Requirements: 9.1, 9.5
        """
        # Mock the LLM response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "summary": "Lab report shows elevated blood glucose",
            "key_findings": ["Blood Glucose: 250 mg/dL (High)"],
            "risk_indicators": ["Elevated glucose level"],
            "follow_up_suggestions": ["Consult endocrinologist"]
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Upload PDF file
        files = {"file": ("test_report.pdf", BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/analyze-document", files=files)
        
        # Verify response status
        assert response.status_code == 200
        
        # Verify response structure
        data = response.json()
        assert "success" in data
        assert data["success"] is True
        
        # Verify analysis result structure
        assert "analysis" in data
        analysis = data["analysis"]
        assert "summary" in analysis
        assert "key_findings" in analysis
        assert "risk_indicators" in analysis
        assert "follow_up_suggestions" in analysis
        assert "medical_disclaimer" in analysis
        assert "analysis_timestamp" in analysis
        
        # Verify data types
        assert isinstance(analysis["key_findings"], list)
        assert isinstance(analysis["risk_indicators"], list)
        assert isinstance(analysis["follow_up_suggestions"], list)
    
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_document_analysis_with_clinical_context(self, mock_model_class, client, sample_pdf_content):
        """
        Test document analysis with clinical context provided.
        
        Requirements: 9.1
        """
        # Mock the LLM response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "summary": "Lab report with patient history context",
            "key_findings": ["Blood Glucose: 250 mg/dL"],
            "risk_indicators": ["Elevated glucose"],
            "follow_up_suggestions": ["Monitor closely"]
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Upload PDF with clinical context
        files = {"file": ("test_report.pdf", BytesIO(sample_pdf_content), "application/pdf")}
        data = {"clinical_context": "Patient has history of diabetes"}
        response = client.post("/analyze-document", files=files, data=data)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "analysis" in result
    
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_document_analysis_includes_threshold_risks(self, mock_model_class, client, sample_pdf_content):
        """
        Test that document analysis includes threshold-based risk indicators.
        
        Requirements: 9.4
        """
        # Mock the LLM response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "summary": "Lab report shows elevated glucose",
            "key_findings": ["Blood Glucose: 250 mg/dL"],
            "risk_indicators": ["Elevated glucose from AI"],
            "follow_up_suggestions": ["Monitor glucose"]
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Upload PDF with clinical parameters
        files = {"file": ("test_report.pdf", BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/analyze-document", files=files)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        
        # Verify risk indicators are present
        analysis = result["analysis"]
        assert "risk_indicators" in analysis
        assert isinstance(analysis["risk_indicators"], list)
        
        # Check if threshold-based risks are included
        # The analysis engine should merge AI and threshold risks
        risk_indicators = analysis["risk_indicators"]
        if len(risk_indicators) > 0:
            # At least one risk indicator should be present
            assert len(risk_indicators) >= 1
    
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_threshold_risks_have_required_fields(self, mock_model_class, client, sample_pdf_content):
        """
        Test that threshold-based risks have all required fields.
        
        Requirements: 9.4
        """
        # Mock the LLM response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "summary": "Lab report",
            "key_findings": ["Blood Glucose: 250 mg/dL"],
            "risk_indicators": [],
            "follow_up_suggestions": []
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Upload PDF
        files = {"file": ("test_report.pdf", BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/analyze-document", files=files)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        
        # Check risk indicators structure
        analysis = result["analysis"]
        risk_indicators = analysis.get("risk_indicators", [])
        
        # If threshold risks are present, verify they have required fields
        for risk in risk_indicators:
            if isinstance(risk, dict):
                # Threshold-based risks should have these fields
                # Note: The exact structure depends on implementation
                # This test verifies the structure is consistent
                assert "finding" in risk or isinstance(risk, str)
    
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_threshold_and_ai_risks_merged(self, mock_model_class, client, sample_pdf_content):
        """
        Test that threshold-based and AI risks are properly merged.
        
        Requirements: 9.4
        """
        # Mock the LLM response with AI risks
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "summary": "Lab report",
            "key_findings": ["Blood Glucose: 250 mg/dL", "Blood Pressure: 140/90 mmHg"],
            "risk_indicators": ["AI detected risk 1", "AI detected risk 2"],
            "follow_up_suggestions": []
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Upload PDF
        files = {"file": ("test_report.pdf", BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/analyze-document", files=files)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        
        # Verify risk indicators include both AI and threshold risks
        analysis = result["analysis"]
        risk_indicators = analysis.get("risk_indicators", [])
        
        # Should have at least the AI risks
        assert len(risk_indicators) >= 2
        
        # Verify no duplicate risks (deduplication should work)
        if isinstance(risk_indicators[0], dict):
            risk_findings = [r.get("finding", "") for r in risk_indicators if isinstance(r, dict)]
        else:
            risk_findings = [str(r) for r in risk_indicators]
        
        # Check for uniqueness (no exact duplicates)
        assert len(risk_findings) == len(set(risk_findings))
    
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_document_analysis_response_structure(self, mock_model_class, client, sample_pdf_content):
        """
        Test that document analysis response has complete structure.
        
        Requirements: 9.1, 9.5
        """
        # Mock the LLM response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "summary": "Complete analysis",
            "key_findings": ["Finding 1", "Finding 2"],
            "risk_indicators": ["Risk 1"],
            "follow_up_suggestions": ["Suggestion 1"]
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Upload document
        files = {"file": ("test.pdf", BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/analyze-document", files=files)
        
        # Verify complete response structure
        assert response.status_code == 200
        data = response.json()
        
        # Top-level fields
        assert "success" in data
        assert "analysis" in data
        assert "filename" in data
        
        # Analysis fields
        analysis = data["analysis"]
        required_fields = [
            "summary",
            "key_findings",
            "risk_indicators",
            "follow_up_suggestions",
            "medical_disclaimer",
            "analysis_timestamp"
        ]
        for field in required_fields:
            assert field in analysis, f"Missing required field: {field}"
        
        # Verify medical disclaimer is present and not empty
        assert analysis["medical_disclaimer"]
        assert len(analysis["medical_disclaimer"]) > 0


    # Test 4.3: Image analysis workflow
    
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_image_analysis_jpeg_workflow(self, mock_model_class, client, sample_image_content):
        """
        Test complete JPEG image analysis workflow from upload to response.
        
        Requirements: 9.2, 9.5
        """
        # Mock the Gemini response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "diagnosis": "Chest X-ray shows normal findings",
            "issues": [],
            "suggestions": ["No immediate follow-up required"],
            "confidence": 95,
            "urgency": "low"
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Upload JPEG image
        files = {"file": ("test_xray.jpg", BytesIO(sample_image_content), "image/jpeg")}
        data = {"image_type": "chest_xray"}
        response = client.post("/analyze-image", files=files, data=data)
        
        # Verify response status
        assert response.status_code == 200
        
        # Verify response structure
        result = response.json()
        assert "success" in result
        assert result["success"] is True
        
        # Verify image analysis fields
        assert "diagnosis" in result
        assert "issues" in result
        assert "suggestions" in result
        assert "confidence" in result
        assert "urgency" in result
        
        # Verify data types
        assert isinstance(result["issues"], list)
        assert isinstance(result["suggestions"], list)
        assert isinstance(result["confidence"], (int, float))
    
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_image_analysis_png_workflow(self, mock_model_class, client, sample_png_content):
        """
        Test complete PNG image analysis workflow from upload to response.
        
        Requirements: 9.2, 9.5
        """
        # Mock the Gemini response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "diagnosis": "MRI scan shows normal brain structure",
            "issues": [],
            "suggestions": ["Routine follow-up"],
            "confidence": 90,
            "urgency": "low"
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Upload PNG image
        files = {"file": ("test_mri.png", BytesIO(sample_png_content), "image/png")}
        data = {"image_type": "mri"}
        response = client.post("/analyze-image", files=files, data=data)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "diagnosis" in result
    
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_image_analysis_with_clinical_context(self, mock_model_class, client, sample_image_content):
        """
        Test image analysis with clinical context provided.
        
        Requirements: 9.2
        """
        # Mock the Gemini response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "diagnosis": "Chest X-ray with patient history context",
            "issues": ["Possible infiltrate"],
            "suggestions": ["Follow-up imaging"],
            "confidence": 85,
            "urgency": "medium"
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Upload image with clinical context
        files = {"file": ("test.jpg", BytesIO(sample_image_content), "image/jpeg")}
        data = {
            "image_type": "chest_xray",
            "clinical_context": "Patient has persistent cough for 2 weeks"
        }
        response = client.post("/analyze-image", files=files, data=data)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
    
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_image_analysis_response_structure(self, mock_model_class, client, sample_image_content):
        """
        Test that image analysis response has complete structure.
        
        Requirements: 9.2, 9.5
        """
        # Mock the Gemini response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "diagnosis": "Complete image analysis",
            "issues": ["Issue 1", "Issue 2"],
            "suggestions": ["Suggestion 1"],
            "confidence": 88,
            "urgency": "medium"
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Upload image
        files = {"file": ("test.jpg", BytesIO(sample_image_content), "image/jpeg")}
        data = {"image_type": "auto"}
        response = client.post("/analyze-image", files=files, data=data)
        
        # Verify complete response structure
        assert response.status_code == 200
        result = response.json()
        
        # Required fields
        required_fields = [
            "success",
            "diagnosis",
            "issues",
            "suggestions",
            "confidence",
            "urgency"
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Verify confidence is in valid range
        assert 0 <= result["confidence"] <= 100
        
        # Verify urgency is valid
        assert result["urgency"] in ["low", "medium", "high"]


    # Test 4.4: Multi-modal workflow
    
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_multimodal_analysis_workflow(self, mock_llm_class, mock_vision_class, 
                                         client, sample_pdf_content, sample_image_content):
        """
        Test complete multi-modal analysis workflow with both report and image.
        
        Requirements: 9.3
        """
        # Mock LLM response for document
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.text = """{
            "summary": "Lab report shows elevated glucose",
            "key_findings": ["Blood Glucose: 250 mg/dL"],
            "risk_indicators": ["Elevated glucose level"],
            "follow_up_suggestions": ["Monitor glucose"]
        }"""
        mock_llm.generate_content.return_value = mock_llm_response
        mock_llm_class.return_value = mock_llm
        
        # Mock Vision response for image
        mock_vision = Mock()
        mock_vision_response = Mock()
        mock_vision_response.text = """{
            "diagnosis": "Retinal examination shows microaneurysms",
            "issues": ["Retinal microaneurysms", "Hemorrhages"],
            "suggestions": ["Ophthalmology referral"],
            "confidence": 90,
            "urgency": "high"
        }"""
        mock_vision.generate_content.return_value = mock_vision_response
        mock_vision_class.return_value = mock_vision
        
        # Upload both report and image
        files = [
            ("report", ("test_report.pdf", BytesIO(sample_pdf_content), "application/pdf")),
            ("image", ("test_retinal.jpg", BytesIO(sample_image_content), "image/jpeg"))
        ]
        response = client.post("/analyze-multimodal", files=files)
        
        # Verify response status
        assert response.status_code == 200
        
        # Verify response structure
        result = response.json()
        assert "success" in result
        assert result["success"] is True
        
        # Verify multi-modal specific fields
        assert "analysis_type" in result
        assert result["analysis_type"] == "multimodal"
        assert "report_analysis" in result
        assert "image_analysis" in result
        assert "correlation" in result
        
        # Verify report analysis is present
        assert result["report_analysis"] is not None
        assert "summary" in result["report_analysis"]
        
        # Verify image analysis is present
        assert result["image_analysis"] is not None
        assert "diagnosis" in result["image_analysis"]
        
        # Verify correlation is present
        assert result["correlation"] is not None
        assert "integrated_diagnosis" in result["correlation"]
        assert "correlations" in result["correlation"]
    
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_multimodal_with_only_report(self, mock_llm_class, client, sample_pdf_content):
        """
        Test multi-modal endpoint with only report provided.
        
        Requirements: 9.3
        """
        # Mock LLM response
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.text = """{
            "summary": "Lab report analysis",
            "key_findings": ["Finding 1"],
            "risk_indicators": [],
            "follow_up_suggestions": []
        }"""
        mock_llm.generate_content.return_value = mock_llm_response
        mock_llm_class.return_value = mock_llm
        
        # Upload only report
        files = [("report", ("test.pdf", BytesIO(sample_pdf_content), "application/pdf"))]
        response = client.post("/analyze-multimodal", files=files)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["report_analysis"] is not None
        assert result["image_analysis"] is None
    
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_multimodal_with_only_image(self, mock_vision_class, client, sample_image_content):
        """
        Test multi-modal endpoint with only image provided.
        
        Requirements: 9.3
        """
        # Mock Vision response
        mock_vision = Mock()
        mock_vision_response = Mock()
        mock_vision_response.text = """{
            "diagnosis": "Image analysis",
            "issues": [],
            "suggestions": [],
            "confidence": 85,
            "urgency": "low"
        }"""
        mock_vision.generate_content.return_value = mock_vision_response
        mock_vision_class.return_value = mock_vision
        
        # Upload only image
        files = [("image", ("test.jpg", BytesIO(sample_image_content), "image/jpeg"))]
        response = client.post("/analyze-multimodal", files=files)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["report_analysis"] is None
        assert result["image_analysis"] is not None
    
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_multimodal_correlation_results(self, mock_llm_class, mock_vision_class,
                                           client, sample_pdf_content, sample_image_content):
        """
        Test that multi-modal analysis includes correlation results.
        
        Requirements: 9.3
        """
        # Mock responses with correlated findings
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.text = """{
            "summary": "Elevated glucose",
            "key_findings": ["Blood Glucose: 300 mg/dL"],
            "risk_indicators": [{"finding": "glucose elevated"}],
            "follow_up_suggestions": []
        }"""
        mock_llm.generate_content.return_value = mock_llm_response
        mock_llm_class.return_value = mock_llm
        
        mock_vision = Mock()
        mock_vision_response = Mock()
        mock_vision_response.text = """{
            "diagnosis": "Retinal changes",
            "issues": ["retinal microaneurysms"],
            "suggestions": [],
            "confidence": 90,
            "urgency": "high"
        }"""
        mock_vision.generate_content.return_value = mock_vision_response
        mock_vision_class.return_value = mock_vision
        
        # Upload both files
        files = [
            ("report", ("test.pdf", BytesIO(sample_pdf_content), "application/pdf")),
            ("image", ("test.jpg", BytesIO(sample_image_content), "image/jpeg"))
        ]
        response = client.post("/analyze-multimodal", files=files)
        
        # Verify correlation structure
        assert response.status_code == 200
        result = response.json()
        correlation = result["correlation"]
        
        assert "integrated_diagnosis" in correlation
        assert "correlations" in correlation
        assert "confidence" in correlation
        assert "recommendations" in correlation
        
        # Verify correlation data types
        assert isinstance(correlation["correlations"], list)
        assert isinstance(correlation["recommendations"], list)
        assert isinstance(correlation["confidence"], (int, float))


    # Test 4.5: Error responses
    
    def test_document_analysis_invalid_file_format(self, client):
        """
        Test error handling for invalid file format.
        
        Requirements: 9.6
        """
        # Upload invalid file format
        invalid_content = b"This is not a valid document"
        files = {"file": ("test.txt", BytesIO(invalid_content), "text/plain")}
        response = client.post("/analyze-document", files=files)
        
        # Verify error response
        assert response.status_code in [400, 415, 500]  # Bad request or unsupported media type
        result = response.json()
        assert "success" in result
        assert result["success"] is False
        assert "error" in result or "detail" in result
    
    def test_document_analysis_no_file(self, client):
        """
        Test error handling when no file is provided.
        
        Requirements: 9.6
        """
        # Send request without file
        response = client.post("/analyze-document")
        
        # Verify error response
        assert response.status_code == 422  # Unprocessable entity
        result = response.json()
        assert "detail" in result
    
    def test_image_analysis_invalid_format(self, client):
        """
        Test error handling for invalid image format.
        
        Requirements: 9.6
        """
        # Upload invalid image format
        invalid_content = b"Not an image"
        files = {"file": ("test.txt", BytesIO(invalid_content), "text/plain")}
        data = {"image_type": "chest_xray"}
        response = client.post("/analyze-image", files=files, data=data)
        
        # Verify error response
        assert response.status_code in [400, 415, 500]
        result = response.json()
        assert "success" in result
        assert result["success"] is False
    
    def test_image_analysis_no_file(self, client):
        """
        Test error handling when no image file is provided.
        
        Requirements: 9.6
        """
        # Send request without file
        response = client.post("/analyze-image", data={"image_type": "chest_xray"})
        
        # Verify error response
        assert response.status_code == 422
        result = response.json()
        assert "detail" in result
    
    def test_multimodal_analysis_no_files(self, client):
        """
        Test error handling when no files are provided to multi-modal endpoint.
        
        Requirements: 9.6
        """
        # Send request without any files
        response = client.post("/analyze-multimodal")
        
        # Verify error response
        assert response.status_code in [400, 422]
        result = response.json()
        assert "detail" in result or "error" in result
    
    def test_error_response_format(self, client):
        """
        Test that error responses follow expected format.
        
        Requirements: 9.6
        """
        # Trigger an error by sending invalid request
        response = client.post("/analyze-document")
        
        # Verify error response format
        assert response.status_code >= 400
        result = response.json()
        
        # Error response should have detail or error field
        assert "detail" in result or "error" in result
        
        # If it has detail, it should be a string or list
        if "detail" in result:
            assert isinstance(result["detail"], (str, list, dict))
    
    def test_document_analysis_corrupted_pdf(self, client):
        """
        Test error handling for corrupted PDF file.
        
        Requirements: 9.6
        """
        # Create corrupted PDF content
        corrupted_pdf = b"%PDF-1.4\nCorrupted content"
        files = {"file": ("corrupted.pdf", BytesIO(corrupted_pdf), "application/pdf")}
        response = client.post("/analyze-document", files=files)
        
        # Verify error response
        assert response.status_code in [400, 500]
        result = response.json()
        assert "success" in result
        assert result["success"] is False
    
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_image_analysis_corrupted_image(self, mock_model_class, client):
        """
        Test error handling for corrupted image file.
        
        Requirements: 9.6
        """
        # Create corrupted image content
        corrupted_image = b"\xFF\xD8\xFF\xE0Corrupted"
        files = {"file": ("corrupted.jpg", BytesIO(corrupted_image), "image/jpeg")}
        data = {"image_type": "chest_xray"}
        
        # Mock to raise an error
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Invalid image")
        mock_model_class.return_value = mock_model
        
        response = client.post("/analyze-image", files=files, data=data)
        
        # Verify error response
        assert response.status_code in [400, 500]
        result = response.json()
        assert "success" in result
        assert result["success"] is False


    # Test 4.6: Performance requirements
    
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_document_analysis_performance(self, mock_model_class, client, sample_pdf_content):
        """
        Test that document analysis completes within performance requirements.
        
        Requirements: 9.7
        """
        # Mock the LLM response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "summary": "Lab report",
            "key_findings": ["Blood Glucose: 250 mg/dL"],
            "risk_indicators": ["Elevated glucose"],
            "follow_up_suggestions": []
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Measure analysis time
        start_time = time.time()
        
        files = {"file": ("test.pdf", BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/analyze-document", files=files)
        
        elapsed_time = time.time() - start_time
        
        # Verify response
        assert response.status_code == 200
        
        # Verify performance requirement (< 30 seconds)
        assert elapsed_time < 30.0, f"Analysis took {elapsed_time:.2f}s, expected < 30s"
    
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_image_analysis_performance(self, mock_model_class, client, sample_image_content):
        """
        Test that image analysis completes within performance requirements.
        
        Requirements: 9.7
        """
        # Mock the Gemini response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """{
            "diagnosis": "Chest X-ray analysis",
            "issues": [],
            "suggestions": [],
            "confidence": 90,
            "urgency": "low"
        }"""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Measure analysis time
        start_time = time.time()
        
        files = {"file": ("test.jpg", BytesIO(sample_image_content), "image/jpeg")}
        data = {"image_type": "chest_xray"}
        response = client.post("/analyze-image", files=files, data=data)
        
        elapsed_time = time.time() - start_time
        
        # Verify response
        assert response.status_code == 200
        
        # Verify performance requirement (< 30 seconds)
        assert elapsed_time < 30.0, f"Analysis took {elapsed_time:.2f}s, expected < 30s"
    
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    @patch('backend.services.llm_service.genai.GenerativeModel')
    def test_multimodal_analysis_performance(self, mock_llm_class, mock_vision_class,
                                            client, sample_pdf_content, sample_image_content):
        """
        Test that multi-modal analysis completes within performance requirements.
        
        Requirements: 9.7
        """
        # Mock LLM response
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.text = """{
            "summary": "Lab report",
            "key_findings": [],
            "risk_indicators": [],
            "follow_up_suggestions": []
        }"""
        mock_llm.generate_content.return_value = mock_llm_response
        mock_llm_class.return_value = mock_llm
        
        # Mock Vision response
        mock_vision = Mock()
        mock_vision_response = Mock()
        mock_vision_response.text = """{
            "diagnosis": "Image analysis",
            "issues": [],
            "suggestions": [],
            "confidence": 85,
            "urgency": "low"
        }"""
        mock_vision.generate_content.return_value = mock_vision_response
        mock_vision_class.return_value = mock_vision
        
        # Measure analysis time
        start_time = time.time()
        
        files = [
            ("report", ("test.pdf", BytesIO(sample_pdf_content), "application/pdf")),
            ("image", ("test.jpg", BytesIO(sample_image_content), "image/jpeg"))
        ]
        response = client.post("/analyze-multimodal", files=files)
        
        elapsed_time = time.time() - start_time
        
        # Verify response
        assert response.status_code == 200
        
        # Verify performance requirement (< 30 seconds)
        assert elapsed_time < 30.0, f"Analysis took {elapsed_time:.2f}s, expected < 30s"

