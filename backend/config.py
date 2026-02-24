import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Gemini configuration (cloud, high accuracy)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # Text model - Latest stable
    GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")  # Vision model - Supports multimodal
    
    # File upload limits
    MAX_FILE_SIZE_MB = 10
    ALLOWED_EXTENSIONS = {".pdf", ".docx"}
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration values"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required. Please add it to your .env file")
        return True
    
    @classmethod
    def get_provider_info(cls):
        """Get information about the current AI provider"""
        return {
            "provider": "Gemini (Google)",
            "type": "cloud",
            "privacy": "Data sent to Google servers",
            "accuracy": "High",
            "text_model": cls.GEMINI_MODEL,
            "vision_model": cls.GEMINI_VISION_MODEL
        }
