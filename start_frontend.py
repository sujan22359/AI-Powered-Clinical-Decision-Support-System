#!/usr/bin/env python3
"""
Startup script for Clinical Report Analyzer frontend
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    print("🏥 Starting Clinical Report Analyzer Frontend...")
    print("📍 Frontend will be available at: http://localhost:8501")
    print("🔗 Make sure the backend is running at: http://localhost:8000")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "frontend/app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")