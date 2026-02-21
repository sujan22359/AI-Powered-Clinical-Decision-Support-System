import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Google Gemini API configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
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
            raise ValueError("GEMINI_API_KEY environment variable is required")
        return True