"""
Gemini LLM Service for Clinical Report Analyzer
Cloud-based AI with high accuracy for medical analysis
"""

import google.generativeai as genai
from typing import Dict, Any, Optional
from dataclasses import dataclass

from backend.config import Config
from backend.utils.logger import setup_logger


@dataclass
class LLMResponse:
    """Response from LLM service"""
    content: str
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0


class GeminiServiceError(Exception):
    """Custom exception for Gemini service errors"""
    pass


class GeminiService:
    """
    Gemini API service for clinical text analysis.
    
    Features:
    - Cloud-based high-accuracy analysis
    - Medical text analysis with specialized prompts
    - Comprehensive error handling
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize Gemini service.
        
        Args:
            api_key: Gemini API key
            model_name: Model to use (default: gemini-2.0-flash-exp)
        """
        self.logger = setup_logger(__name__)
        
        # Configuration
        self.api_key = api_key or Config.GEMINI_API_KEY
        # Add 'models/' prefix if not present for v1beta API compatibility
        raw_model_name = model_name or Config.GEMINI_MODEL
        if not raw_model_name.startswith('models/'):
            self.model_name = f"models/{raw_model_name}"
        else:
            self.model_name = raw_model_name
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        self.logger.info(f"Gemini service initialized with model: {self.model_name}")
    
    def analyze_clinical_text(self, clinical_text: str) -> LLMResponse:
        """
        Analyze clinical text and generate structured medical insights.
        
        Args:
            clinical_text: Raw clinical text to analyze
            
        Returns:
            LLMResponse with analysis results or error information
        """
        if not clinical_text or not clinical_text.strip():
            return LLMResponse(
                content="",
                success=False,
                error_message="Clinical text is empty or invalid"
            )
        
        # Prepare the prompt for medical analysis
        prompt = self._prepare_medical_analysis_prompt(clinical_text)
        
        # Execute API call
        try:
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                self.logger.info("Successfully generated response from Gemini")
                return LLMResponse(
                    content=response.text,
                    success=True
                )
            else:
                error_msg = "Empty response from Gemini"
                self.logger.warning(error_msg)
                return LLMResponse(
                    content="",
                    success=False,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            self.logger.error(error_msg)
            return LLMResponse(
                content="",
                success=False,
                error_message=error_msg
            )
    
    def _prepare_medical_analysis_prompt(self, clinical_text: str) -> str:
        """
        Prepare specialized prompt for medical text analysis.
        
        Args:
            clinical_text: Clinical text to analyze
            
        Returns:
            Formatted prompt for medical analysis
        """
        prompt = f"""You are a senior medical AI consultant with 15 years of experience in clinical laboratory medicine, diagnostic pathology, and patient care. Your expertise includes interpreting complex lab results, identifying critical values, and providing clear, actionable insights for healthcare professionals and patients.

PROFESSIONAL CONTEXT:
- You have extensive experience reviewing thousands of clinical reports
- You understand the clinical significance of laboratory values and their interrelationships
- You can identify subtle patterns that may indicate underlying conditions
- You communicate complex medical information in clear, accessible language
- You prioritize patient safety and clinical accuracy in all assessments

ANALYSIS APPROACH:
- Systematically review all parameters and their reference ranges
- Identify critical values that require immediate attention
- Recognize patterns suggesting specific conditions or organ dysfunction
- Consider the clinical context and potential differential diagnoses
- Provide evidence-based follow-up recommendations

CRITICAL SAFETY REQUIREMENTS:
- This is an informational analysis, NOT a medical diagnosis
- Always recommend consultation with qualified healthcare professionals
- Use precise medical terminology while remaining accessible
- Clearly distinguish between normal, borderline, and abnormal findings
- Highlight urgent findings that require immediate medical attention

Please analyze the following clinical laboratory report and provide a comprehensive structured response in JSON format:

REQUIRED JSON STRUCTURE:
{{
  "summary": "A clear, professional summary of the overall findings (3-4 sentences). Include the most significant findings and their clinical implications.",
  
  "key_findings": [
    "List each significant finding with specific values and reference ranges",
    "Include both abnormal and notable normal findings",
    "Use precise medical terminology with explanations"
  ],
  
  "risk_indicators": [
    "List any abnormal values with clinical significance",
    "Specify the degree of abnormality (mildly/moderately/severely elevated or decreased)",
    "Indicate potential clinical implications",
    "Prioritize by urgency (critical, high, moderate, low)"
  ],
  
  "follow_up_suggestions": [
    "Specific recommendations for immediate actions if critical values present",
    "Suggested additional tests or evaluations based on findings",
    "Lifestyle or monitoring recommendations",
    "Timeline for follow-up (urgent, within 1 week, routine)",
    "Specialist consultations if indicated"
  ],
  
  "medical_disclaimer": "IMPORTANT: This analysis is for informational purposes only and does not constitute medical advice, diagnosis, or treatment. All findings should be reviewed by a qualified healthcare professional who can consider your complete medical history, physical examination, and clinical context. If you have concerning symptoms or critical values, seek immediate medical attention."
}}

CLINICAL LABORATORY REPORT:
{clinical_text}

Provide your analysis as valid JSON only. Be thorough, precise, and clinically relevant. Focus on actionable insights that support better patient care.
"""
        return prompt
    
    def validate_api_connection(self) -> bool:
        """
        Validate Gemini API connection.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            self.logger.debug("Validating Gemini API connection")
            # Try a simple generation to test the API
            response = self.model.generate_content("Test")
            return response is not None
        except Exception as e:
            self.logger.error(f"Gemini API validation failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "provider": "Gemini (Google Cloud)",
            "type": "cloud"
        }
