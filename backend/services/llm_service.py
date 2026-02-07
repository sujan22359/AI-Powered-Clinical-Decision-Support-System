"""
LLM Service for Clinical Report Analyzer

This module provides Google Gemini API integration for analyzing clinical text
and generating medical insights with robust error handling and retry logic.
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions

from backend.config import Config
from backend.utils.logger import setup_logger


@dataclass
class LLMResponse:
    """Response from LLM service"""
    content: str
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0


class LLMServiceError(Exception):
    """Custom exception for LLM service errors"""
    pass


class LLMService:
    """
    Google Gemini API service for clinical text analysis.
    
    Features:
    - Medical text analysis with specialized prompts
    - Retry logic with exponential backoff
    - Comprehensive error handling
    - Safety settings for medical content
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM service with Google Gemini API.
        
        Args:
            api_key: Google Gemini API key (uses config if not provided)
        """
        self.logger = setup_logger(__name__)
        
        # Use provided API key or get from config
        self.api_key = api_key or Config.GEMINI_API_KEY
        if not self.api_key:
            raise LLMServiceError("Google Gemini API key is required")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize model with safety settings
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 30.0  # Maximum delay in seconds
        
        self.logger.info("LLM service initialized with Google Gemini API")
    
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
        
        # Execute with retry logic
        return self._execute_with_retry(prompt)
    
    def _prepare_medical_analysis_prompt(self, clinical_text: str) -> str:
        """
        Prepare specialized prompt for medical text analysis.
        
        Args:
            clinical_text: Clinical text to analyze
            
        Returns:
            Formatted prompt for medical analysis
        """
        prompt = f"""
You are a medical AI assistant analyzing clinical reports for informational purposes only. 
Your role is to provide patient-friendly summaries and insights while maintaining strict safety guidelines.

CRITICAL SAFETY REQUIREMENTS:
- NEVER provide medical diagnoses or treatment recommendations
- ALWAYS include appropriate medical disclaimers
- Focus on informational content only
- Use clear, patient-friendly language
- Highlight any concerning findings as "may require attention" rather than diagnostic terms

Please analyze the following clinical text and provide a structured response in JSON format with these sections:

1. "summary": A patient-friendly summary in plain language (2-3 sentences)
2. "key_findings": List of important medical findings from the text
3. "risk_indicators": Any abnormal values or findings that may require attention
4. "follow_up_suggestions": Non-diagnostic suggestions for patient care or monitoring
5. "medical_disclaimer": Required disclaimer about non-diagnostic nature

Clinical Text to Analyze:
{clinical_text}

Respond with valid JSON only. Ensure all sections are present even if empty (use empty arrays [] for lists).
"""
        return prompt
    
    def _execute_with_retry(self, prompt: str) -> LLMResponse:
        """
        Execute API call with retry logic and exponential backoff.
        
        Args:
            prompt: Formatted prompt to send to the API
            
        Returns:
            LLMResponse with results or error information
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Attempting API call (attempt {attempt + 1}/{self.max_retries + 1})")
                
                # Generate response
                response = self.model.generate_content(prompt)
                
                # Check if response was blocked
                if response.candidates and response.candidates[0].finish_reason.name == "SAFETY":
                    error_msg = "Content was blocked by safety filters"
                    self.logger.warning(error_msg)
                    return LLMResponse(
                        content="",
                        success=False,
                        error_message=error_msg,
                        retry_count=attempt
                    )
                
                # Extract text content
                if response.text:
                    self.logger.info(f"Successfully generated response on attempt {attempt + 1}")
                    return LLMResponse(
                        content=response.text,
                        success=True,
                        retry_count=attempt
                    )
                else:
                    error_msg = "Empty response received from API"
                    self.logger.warning(error_msg)
                    last_error = LLMServiceError(error_msg)
                
            except google_exceptions.ResourceExhausted as e:
                # Rate limiting - wait longer before retry
                error_msg = f"Rate limit exceeded: {str(e)}"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                last_error = LLMServiceError(error_msg)
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (3 ** attempt), self.max_delay)
                    self.logger.info(f"Rate limited, waiting {delay:.1f} seconds before retry")
                    time.sleep(delay)
                
            except google_exceptions.ServiceUnavailable as e:
                # Service unavailable - retry with backoff
                error_msg = f"Service unavailable: {str(e)}"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                last_error = LLMServiceError(error_msg)
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    self.logger.info(f"Service unavailable, waiting {delay:.1f} seconds before retry")
                    time.sleep(delay)
                
            except google_exceptions.InvalidArgument as e:
                # Invalid request - don't retry
                error_msg = f"Invalid request: {str(e)}"
                self.logger.error(error_msg)
                return LLMResponse(
                    content="",
                    success=False,
                    error_message=error_msg,
                    retry_count=attempt
                )
                
            except google_exceptions.Unauthenticated as e:
                # Authentication error - don't retry
                error_msg = f"Authentication failed: {str(e)}"
                self.logger.error(error_msg)
                return LLMResponse(
                    content="",
                    success=False,
                    error_message=error_msg,
                    retry_count=attempt
                )
                
            except Exception as e:
                # Unexpected error
                error_msg = f"Unexpected error: {str(e)}"
                self.logger.error(f"{error_msg} (attempt {attempt + 1})")
                last_error = LLMServiceError(error_msg)
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    self.logger.info(f"Unexpected error, waiting {delay:.1f} seconds before retry")
                    time.sleep(delay)
        
        # All retries exhausted
        final_error = f"Failed after {self.max_retries + 1} attempts. Last error: {str(last_error)}"
        self.logger.error(final_error)
        
        return LLMResponse(
            content="",
            success=False,
            error_message=final_error,
            retry_count=self.max_retries
        )
    
    def validate_api_connection(self) -> bool:
        """
        Validate API connection and authentication.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            self.logger.debug("Validating API connection")
            
            # Simple test prompt
            test_prompt = "Respond with 'OK' if you can process this message."
            response = self.model.generate_content(test_prompt)
            
            if response.text and "OK" in response.text:
                self.logger.info("API connection validated successfully")
                return True
            else:
                self.logger.warning("API connection validation failed - unexpected response")
                return False
                
        except Exception as e:
            self.logger.error(f"API connection validation failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": "gemini-2.0-flash-exp",
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "api_configured": bool(self.api_key)
        }