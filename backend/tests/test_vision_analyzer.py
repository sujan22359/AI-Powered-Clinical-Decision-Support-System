"""
Unit tests for Medical Image Analyzer (Vision Analyzer Service)

Tests cover:
- Image analysis for different image types
- Image format handling (JPEG, PNG)
- Error handling for corrupted/invalid images
- Analysis result structure validation
- Clinical context handling
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

# Import the service to test
from backend.services.vision_analyzer import MedicalImageAnalyzer


class TestMedicalImageAnalyzer:
    """Test suite for MedicalImageAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a MedicalImageAnalyzer instance for testing"""
        # Use a test API key
        api_key = os.getenv('GEMINI_API_KEY', 'test_api_key')
        return MedicalImageAnalyzer(api_key=api_key)
    
    @pytest.fixture
    def sample_image_bytes(self, fixtures_dir):
        """Load sample X-ray image as bytes"""
        image_path = fixtures_dir / "sample_xray.jpg"
        with open(image_path, 'rb') as f:
            return f.read()
    
    @pytest.fixture
    def sample_png_bytes(self, fixtures_dir):
        """Load sample PNG image as bytes"""
        image_path = fixtures_dir / "sample_image.png"
        with open(image_path, 'rb') as f:
            return f.read()
    
    @pytest.fixture
    def sample_ct_bytes(self, fixtures_dir):
        """Load sample CT scan image as bytes"""
        image_path = fixtures_dir / "sample_ct_brain.jpg"
        with open(image_path, 'rb') as f:
            return f.read()
    
    @pytest.fixture
    def corrupted_image_bytes(self, fixtures_dir):
        """Load corrupted image file"""
        image_path = fixtures_dir / "corrupted_image.jpg"
        with open(image_path, 'rb') as f:
            return f.read()
    
    @pytest.fixture
    def mock_gemini_response(self):
        """Create a mock Gemini API response"""
        mock_response = Mock()
        mock_response.text = """
**DIAGNOSIS:**
Test chest X-ray shows normal cardiac and pulmonary findings.

**ISSUES IDENTIFIED:**
- No significant abnormalities detected
- Image quality is adequate for interpretation

**CONFIDENCE LEVEL:**
85%

**URGENCY:**
LOW

**DETAILED FINDINGS:**
1. Image Quality: Good technical quality
2. Normal Structures: Heart size normal, lung fields clear
3. Abnormal Findings: None identified
4. Measurements: Cardiothoracic ratio within normal limits
5. Comparison: Consistent with normal chest anatomy

**FOLLOW-UP SUGGESTIONS:**
1. No immediate action required
2. Continue routine health maintenance
3. Follow up as scheduled with primary care
4. Monitor for any new symptoms
5. Maintain healthy lifestyle
6. Return if symptoms develop
"""
        return mock_response
    
    # Test 2.1: Basic test structure
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly"""
        assert analyzer is not None
        assert analyzer.model is not None
        assert hasattr(analyzer, 'analyze_medical_image')
    
    # Test 2.2: Supported image types
    def test_get_supported_image_types(self, analyzer):
        """Test that all supported image types are returned"""
        supported_types = analyzer.get_supported_image_types()
        
        assert len(supported_types) > 0
        assert any(t['type'] == 'chest_xray' for t in supported_types)
        assert any(t['type'] == 'ct_brain' for t in supported_types)
        assert any(t['type'] == 'bone_xray' for t in supported_types)
        assert any(t['type'] == 'mri' for t in supported_types)
        assert any(t['type'] == 'ultrasound' for t in supported_types)
        
        # Check structure
        for img_type in supported_types:
            assert 'type' in img_type
            assert 'name' in img_type
            assert 'description' in img_type
    
    # Test 2.2: Image type analysis - Chest X-ray
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analyze_chest_xray(self, mock_model_class, analyzer, sample_image_bytes, mock_gemini_response):
        """Test chest X-ray analysis"""
        # Mock the model's generate_content method
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        result = analyzer.analyze_medical_image(
            image_data=sample_image_bytes,
            image_type="chest_xray"
        )
        
        assert result['success'] is True
        assert result['image_type'] == 'chest_xray'
        assert 'diagnosis' in result
        assert 'issues' in result
        assert 'suggestions' in result
        assert 'confidence' in result
        assert 'urgency' in result
    
    # Test 2.2: Image type analysis - CT Brain
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analyze_ct_brain(self, mock_model_class, analyzer, sample_ct_bytes, mock_gemini_response):
        """Test CT brain scan analysis"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        result = analyzer.analyze_medical_image(
            image_data=sample_ct_bytes,
            image_type="ct_brain"
        )
        
        assert result['success'] is True
        assert result['image_type'] == 'ct_brain'
    
    # Test 2.2: Image type analysis - Bone X-ray
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analyze_bone_xray(self, mock_model_class, analyzer, sample_image_bytes, mock_gemini_response):
        """Test bone X-ray analysis"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        result = analyzer.analyze_medical_image(
            image_data=sample_image_bytes,
            image_type="bone_xray"
        )
        
        assert result['success'] is True
        assert result['image_type'] == 'bone_xray'
    
    # Test 2.2: Image type analysis - MRI
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analyze_mri(self, mock_model_class, analyzer, sample_image_bytes, mock_gemini_response):
        """Test MRI scan analysis"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        result = analyzer.analyze_medical_image(
            image_data=sample_image_bytes,
            image_type="mri"
        )
        
        assert result['success'] is True
        assert result['image_type'] == 'mri'
    
    # Test 2.2: Image type analysis - Ultrasound
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analyze_ultrasound(self, mock_model_class, analyzer, sample_image_bytes, mock_gemini_response):
        """Test ultrasound analysis"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        result = analyzer.analyze_medical_image(
            image_data=sample_image_bytes,
            image_type="ultrasound"
        )
        
        assert result['success'] is True
        assert result['image_type'] == 'ultrasound'
    
    # Test 2.3: JPEG format handling
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analyze_jpeg_format(self, mock_model_class, analyzer, sample_image_bytes, mock_gemini_response):
        """Test JPEG format analysis"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        result = analyzer.analyze_medical_image(
            image_data=sample_image_bytes,
            image_type="chest_xray"
        )
        
        assert result['success'] is True
        # Verify image was loaded correctly
        assert 'diagnosis' in result
    
    # Test 2.3: PNG format handling
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analyze_png_format(self, mock_model_class, analyzer, sample_png_bytes, mock_gemini_response):
        """Test PNG format analysis"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        result = analyzer.analyze_medical_image(
            image_data=sample_png_bytes,
            image_type="auto"
        )
        
        assert result['success'] is True
    
    # Test 2.3: Corrupted image handling
    def test_analyze_corrupted_image(self, analyzer, corrupted_image_bytes):
        """Test error handling for corrupted images"""
        result = analyzer.analyze_medical_image(
            image_data=corrupted_image_bytes,
            image_type="chest_xray"
        )
        
        assert result['success'] is False
        assert 'error' in result
        assert result['diagnosis'] == 'Analysis failed'
    
    # Test 2.3: Unsupported format handling
    def test_analyze_unsupported_format(self, analyzer):
        """Test error handling for unsupported image formats"""
        # Create invalid image data
        invalid_data = b"This is not an image"
        
        result = analyzer.analyze_medical_image(
            image_data=invalid_data,
            image_type="chest_xray"
        )
        
        assert result['success'] is False
        assert 'error' in result
    
    # Test 2.4: Analysis result structure
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analysis_result_structure(self, mock_model_class, analyzer, sample_image_bytes, mock_gemini_response):
        """Test that analysis results include all required fields"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        result = analyzer.analyze_medical_image(
            image_data=sample_image_bytes,
            image_type="chest_xray"
        )
        
        # Verify all required fields are present
        assert 'success' in result
        assert 'diagnosis' in result
        assert 'issues' in result
        assert 'suggestions' in result
        assert 'confidence' in result
        assert 'urgency' in result
        assert 'findings' in result
        
        # Verify field types
        assert isinstance(result['diagnosis'], str)
        assert isinstance(result['issues'], list)
        assert isinstance(result['suggestions'], list)
        assert isinstance(result['confidence'], int)
        assert isinstance(result['urgency'], str)
        
        # Verify diagnosis is non-empty
        assert len(result['diagnosis']) > 0
    
    # Test 2.5: Clinical context handling - with context
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analyze_with_clinical_context(self, mock_model_class, analyzer, sample_image_bytes, mock_gemini_response):
        """Test analysis with clinical context provided"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        clinical_context = "Patient has persistent cough for 2 weeks, fever, and shortness of breath"
        
        result = analyzer.analyze_medical_image(
            image_data=sample_image_bytes,
            image_type="chest_xray",
            clinical_context=clinical_context
        )
        
        assert result['success'] is True
        # Verify the model was called with a prompt
        assert mock_model.generate_content.called
    
    # Test 2.5: Clinical context handling - without context
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analyze_without_clinical_context(self, mock_model_class, analyzer, sample_image_bytes, mock_gemini_response):
        """Test analysis without clinical context"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        result = analyzer.analyze_medical_image(
            image_data=sample_image_bytes,
            image_type="chest_xray",
            clinical_context=None
        )
        
        assert result['success'] is True
    
    # Test prompt creation
    def test_create_medical_analysis_prompt(self, analyzer):
        """Test prompt creation for different image types"""
        prompt = analyzer._create_medical_analysis_prompt(
            image_type="chest_xray",
            clinical_context="Patient has cough",
            patient_info={"age": 45, "gender": "Male"}
        )
        
        assert "chest" in prompt.lower() or "x-ray" in prompt.lower()
        assert "Patient has cough" in prompt
        assert "45" in prompt
    
    # Test response parsing
    def test_parse_medical_response(self, analyzer, mock_gemini_response):
        """Test parsing of Gemini response"""
        parsed = analyzer._parse_medical_response(
            mock_gemini_response.text,
            "chest_xray"
        )
        
        assert 'diagnosis' in parsed
        assert 'issues' in parsed
        assert 'suggestions' in parsed
        assert 'confidence' in parsed
        assert 'urgency' in parsed
        
        # Verify parsed values
        assert parsed['confidence'] == 85
        assert parsed['urgency'] == 'low'
        assert len(parsed['issues']) > 0
        assert len(parsed['suggestions']) > 0
    
    # Test multiple image analysis
    @patch('backend.services.vision_analyzer.genai.GenerativeModel')
    def test_analyze_multiple_images(self, mock_model_class, analyzer, sample_image_bytes, sample_ct_bytes, mock_gemini_response):
        """Test analyzing multiple images"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        analyzer.model = mock_model
        
        images = [
            (sample_image_bytes, "chest_xray"),
            (sample_ct_bytes, "ct_brain")
        ]
        
        results = analyzer.analyze_multiple_images(images)
        
        assert len(results) == 2
        assert results[0]['image_type'] == 'chest_xray'
        assert results[1]['image_type'] == 'ct_brain'
    
    # Test image type instructions
    def test_get_image_type_instructions(self, analyzer):
        """Test that image type instructions are generated"""
        instructions = analyzer._get_image_type_instructions("chest_xray")
        assert len(instructions) > 0
        assert "chest" in instructions.lower() or "lung" in instructions.lower()
        
        instructions = analyzer._get_image_type_instructions("ct_brain")
        assert "brain" in instructions.lower()
        
        instructions = analyzer._get_image_type_instructions("auto")
        assert len(instructions) > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
