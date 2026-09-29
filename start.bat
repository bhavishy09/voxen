@echo off
echo Starting AI Voice Sales Agent...
echo.

@REM # Before running your start.bat, use:
@REM taskkill /F /IM python.exe
@REM start.bat

REM Check if port 8000 is in use and kill if needed
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    echo Stopping existing server on port 8000...
    taskkill /F /PID %%a >nul 2>&1
)

REM Check if virtual environment exists, create if not
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Starting the server...
echo Open http://localhost:8000 in your browser
echo.

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
