@echo off
echo ======================================================================
echo    MediScan AI - Clinical Decision Support System
echo ======================================================================
echo.

REM Check if Ollama is running
echo Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama is not running!
    echo Please start Ollama first: ollama serve
    echo.
    pause
    exit /b 1
)
echo [OK] Ollama is running
echo.

echo ======================================================================
echo Starting Backend API...
echo ======================================================================
start "MediScan Backend" cmd /k "python -m uvicorn backend.main:app --host localhost --port 8000 --reload"

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo ======================================================================
echo Starting Frontend UI...
echo ======================================================================
start "MediScan Frontend" cmd /k "streamlit run frontend/app.py"

echo.
echo ======================================================================
echo MediScan AI is starting...
echo ======================================================================
echo.
echo Frontend UI:  http://localhost:8501
echo Backend API:  http://localhost:8000
echo API Docs:     http://localhost:8000/docs
echo.
echo Both services are running in separate windows.
echo Close those windows to stop the services.
echo ======================================================================
echo.
pause
