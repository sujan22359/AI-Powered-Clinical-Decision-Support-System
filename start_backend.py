#!/usr/bin/env python3
"""
Startup script for Clinical Report Analyzer backend
"""

import uvicorn
from backend.config import Config

if __name__ == "__main__":
    # Validate configuration
    try:
        Config.validate_config()
        print("✅ Configuration validated successfully")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        exit(1)
    
    print("🚀 Starting Clinical Report Analyzer API...")
    print(f"📍 API will be available at: http://{Config.API_HOST}:{Config.API_PORT}")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🏥 Streamlit UI: Run 'streamlit run frontend/app.py' in another terminal")
    
    uvicorn.run(
        "backend.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,
        log_level=Config.LOG_LEVEL.lower()
    )