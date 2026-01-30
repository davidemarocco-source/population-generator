@echo off
cd /d "%~dp0"
echo ==========================================
echo Psychometric Population Generator Launcher
echo ==========================================

echo [1/2] Checking/Installing dependencies...
py -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies.
    echo Please ensure Python is installed.
    echo.
    pause
    exit /b
)

echo.
echo [2/2] Starting GUI...
py gui.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The application crashed.
    echo Please check the error message above.
    echo.
    pause
    exit /b
)

echo.
echo Application closed.
pause
