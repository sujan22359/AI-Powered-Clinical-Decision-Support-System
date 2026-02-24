#!/usr/bin/env python3
"""
Simple startup script for Clinical Report Analyzer
Runs both backend and frontend
"""

import subprocess
import sys
import time
from pathlib import Path

def print_banner():
    print("\n" + "="*70)
    print("🏥 Clinical Report Analyzer - AI-Powered Medical Analysis")
    print("="*70 + "\n")

def main():
    print_banner()
    
    print("🚀 Starting Clinical Report Analyzer...")
    print("="*70 + "\n")
    
    # Start backend
    print("📍 Starting Backend API on http://localhost:8000")
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "localhost",
        "--port", "8000",
        "--reload"
    ]
    
    # Start frontend with headless mode to prevent auto-opening browser
    print("🎨 Starting Frontend UI on http://localhost:8501")
    frontend_cmd = [
        sys.executable, "-m", "streamlit",
        "run", "frontend/app.py",
        "--server.headless", "true"
    ]
    
    # On Windows, use CREATE_NEW_CONSOLE to open in new windows
    if sys.platform == "win32":
        CREATE_NEW_CONSOLE = 0x00000010
        
        # Start backend in new console
        subprocess.Popen(
            backend_cmd,
            creationflags=CREATE_NEW_CONSOLE,
            cwd=Path.cwd()
        )
        
        print("⏳ Waiting for backend to start...")
        time.sleep(5)
        
        # Start frontend in new console
        subprocess.Popen(
            frontend_cmd,
            creationflags=CREATE_NEW_CONSOLE,
            cwd=Path.cwd()
        )
        
        print("⏳ Waiting for frontend to start...")
        time.sleep(5)
        
    else:
        # On Linux/Mac
        subprocess.Popen(backend_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        subprocess.Popen(frontend_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
    
    print("\n" + "="*70)
    print("✅ Clinical Report Analyzer is running!")
    print("="*70)
    print("\n📍 Access Points:")
    print("   • Frontend UI:  http://localhost:8501")
    print("   • Backend API:  http://localhost:8000")
    print("   • API Docs:     http://localhost:8000/docs")
    print("\n💡 Both services are running in separate windows")
    print("   Close those windows to stop the services")
    print("="*70 + "\n")
    
    input("Press Enter to exit this window (services will keep running)...")

if __name__ == "__main__":
    main()
