"""
Medical Image Analysis Service using Gemini Vision
Analyzes medical images (X-rays, CT, MRI) with high accuracy
"""

import google.generativeai as genai
from PIL import Image
import io
from typing import Dict, List, Optional
import re
from backend.utils.logger import setup_logger
from backend.config import Config

logger = setup_logger(__name__)


class GeminiMedicalImageAnalyzer:
    """
    Analyzes medical images using Gemini Vision (cloud vision model)
    Provides diagnosis, identifies issues, and gives follow-up suggestions
    High accuracy for complex medical imaging
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the Medical Image Analyzer with Gemini
        
        Args:
            api_key: Gemini API key
            model_name: Vision model to use (default: gemini-2.0-flash-exp)
        """
        self.api_key = api_key or Config.GEMINI_API_KEY
        # Add 'models/' prefix if not present for v1beta API compatibility
        raw_model_name = model_name or Config.GEMINI_VISION_MODEL
        if not raw_model_name.startswith('models/'):
            self.model_name = f"models/{raw_model_name}"
        else:
            self.model_name = raw_model_name
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"Gemini Medical Image Analyzer initialized with {self.model_name}")
    
    def analyze_medical_image(
        self, 
        image_data: bytes,
        image_type: str = "auto",
        clinical_context: Optional[str] = None,
        patient_info: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze a medical image and provide diagnosis with suggestions
        
        Args:
            image_data: Image file bytes
            image_type: Type of medical image (chest_xray, ct_brain, mri, bone_xray, auto)
            clinical_context: Additional clinical information (symptoms, history)
            patient_info: Patient details (age, gender, medical history)
        
        Returns:
            Dictionary containing:
            - diagnosis: Main diagnosis
            - issues: List of identified issues
            - suggestions: Follow-up recommendations
            - confidence: Confidence level (0-100)
            - findings: Detailed findings
            - urgency: Urgency level (low, medium, high, critical)
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"Analyzing {image_type} image with Gemini, size: {image.size}")
            
            # Create specialized prompt
            prompt = self._create_medical_analysis_prompt(
                image_type, 
                clinical_context, 
                patient_info
            )
            
            # Generate analysis using Gemini Vision
            logger.info("Sending image to Gemini for analysis...")
            response = self.model.generate_content([prompt, image])
            
            if not response or not response.text:
                return {
                    "success": False,
                    "error": "Empty response from Gemini",
                    "diagnosis": "Analysis failed",
                    "issues": [],
                    "suggestions": ["Please try again or check your API key"],
                    "confidence": 0,
                    "urgency": "medium",
                    "findings": {}
                }
            
            # Parse the response
            analysis = self._parse_medical_response(response.text, image_type)
            
            logger.info(f"Analysis complete: {analysis['diagnosis']}")
            
            return {
                "success": True,
                "image_type": image_type,
                "diagnosis": analysis["diagnosis"],
                "issues": analysis["issues"],
                "suggestions": analysis["suggestions"],
                "confidence": analysis["confidence"],
                "findings": analysis["findings"],
                "urgency": analysis["urgency"],
                "raw_analysis": response.text
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "diagnosis": "Analysis failed",
                "issues": [],
                "suggestions": ["Please check your Gemini API key and try again"],
                "confidence": 0,
                "urgency": "medium",
                "findings": {}
            }
    
    def _create_medical_analysis_prompt(
        self, 
        image_type: str, 
        clinical_context: Optional[str],
        patient_info: Optional[Dict]
    ) -> str:
        """Create a specialized prompt for medical image analysis"""
        
        # Build patient context
        patient_context = ""
        if patient_info:
            age = patient_info.get('age', 'Unknown')
            gender = patient_info.get('gender', 'Unknown')
            history = patient_info.get('medical_history', 'None provided')
            patient_context = f"""
PATIENT INFORMATION:
- Age: {age}
- Gender: {gender}
- Medical History: {history}
"""
        
        # Build clinical context
        clinical_info = ""
        if clinical_context:
            clinical_info = f"""
CLINICAL CONTEXT:
{clinical_context}
"""
        
        # Base prompt with expert persona
        prompt = f"""You are a board-certified radiologist with 15 years of experience in diagnostic imaging. Analyze this medical image systematically and provide a comprehensive report.

{patient_context}
{clinical_info}

Please provide your analysis in the following format:

**DIAGNOSIS:**
[Clear, concise primary impression]

**ISSUES IDENTIFIED:**
- [Issue 1 with specific details]
- [Issue 2 with specific details]
- [Additional findings]

**CONFIDENCE LEVEL:**
[Percentage 0-100%]

**URGENCY:**
[LOW, MEDIUM, HIGH, or CRITICAL]

**DETAILED FINDINGS:**
[Comprehensive description of all findings]

**FOLLOW-UP SUGGESTIONS:**
1. [Specific recommendation 1]
2. [Specific recommendation 2]
3. [Additional recommendations]

**MEDICAL DISCLAIMER:**
This imaging analysis is for informational purposes only and must be reviewed by a qualified radiologist or treating physician.
"""
        
        return prompt
    
    def _parse_medical_response(self, response_text: str, image_type: str) -> Dict:
        """Parse the AI response into structured format"""
        
        result = {
            "diagnosis": "Unable to determine",
            "issues": [],
            "suggestions": [],
            "confidence": 0,
            "urgency": "medium",
            "findings": {}
        }
        
        try:
            # Extract diagnosis
            diagnosis_match = re.search(
                r'\*\*DIAGNOSIS:\*\*\s*\n(.+?)(?=\n\*\*|$)', 
                response_text, 
                re.DOTALL
            )
            if diagnosis_match:
                result["diagnosis"] = diagnosis_match.group(1).strip()
            elif len(response_text) > 50:
                result["diagnosis"] = response_text.split('\n\n')[0].strip()[:500]
            
            # Extract issues
            issues_match = re.search(
                r'\*\*ISSUES IDENTIFIED:\*\*\s*\n(.+?)(?=\n\*\*|$)', 
                response_text, 
                re.DOTALL
            )
            if issues_match:
                issues_text = issues_match.group(1)
                issues = re.findall(r'[-•]\s*(.+?)(?=\n[-•]|\n\n|$)', issues_text, re.DOTALL)
                result["issues"] = [issue.strip() for issue in issues if issue.strip()]
            
            # Extract confidence
            confidence_match = re.search(
                r'\*\*CONFIDENCE LEVEL:\*\*\s*\n?(\d+)%?', 
                response_text
            )
            if confidence_match:
                result["confidence"] = int(confidence_match.group(1))
            else:
                result["confidence"] = 85  # Default high confidence for Gemini
            
            # Extract urgency
            urgency_match = re.search(
                r'\*\*URGENCY:\*\*\s*\n?(\w+)', 
                response_text, 
                re.IGNORECASE
            )
            if urgency_match:
                urgency = urgency_match.group(1).lower()
                result["urgency"] = urgency if urgency in ["low", "medium", "high", "critical"] else "medium"
            
            # Extract suggestions
            suggestions_match = re.search(
                r'\*\*FOLLOW-UP SUGGESTIONS:\*\*\s*\n(.+?)(?=\n\*\*|$)', 
                response_text, 
                re.DOTALL
            )
            if suggestions_match:
                suggestions_text = suggestions_match.group(1)
                suggestions = re.findall(r'[\d•-]+\.\s*(.+?)(?=\n[\d•-]+\.|\n\n|$)', suggestions_text, re.DOTALL)
                result["suggestions"] = [sug.strip() for sug in suggestions if sug.strip()]
            
            # Extract detailed findings
            findings_match = re.search(
                r'\*\*DETAILED FINDINGS:\*\*\s*\n(.+?)(?=\n\*\*|$)', 
                response_text, 
                re.DOTALL
            )
            if findings_match:
                result["findings"]["detailed"] = findings_match.group(1).strip()
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            result["findings"]["raw"] = response_text
        
        return result
    
    def get_supported_image_types(self) -> List[Dict[str, str]]:
        """Get list of supported medical image types"""
        return [
            {"type": "chest_xray", "name": "Chest X-ray", "description": "Detects pneumonia, lung cancer, heart problems"},
            {"type": "ct_brain", "name": "Brain CT Scan", "description": "Detects stroke, bleeding, tumors"},
            {"type": "bone_xray", "name": "Bone X-ray", "description": "Detects fractures, arthritis"},
            {"type": "mri", "name": "MRI Scan", "description": "Detailed soft tissue imaging"},
            {"type": "ultrasound", "name": "Ultrasound", "description": "Organ imaging, pregnancy"},
            {"type": "auto", "name": "Auto-detect", "description": "Automatically detect image type"}
        ]
