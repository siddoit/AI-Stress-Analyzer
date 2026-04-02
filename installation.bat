@echo off
title Stress Analyzer Initialization
echo ========================================================
echo SYSTEM BOOT: Stress Analyzer
echo ========================================================

:: 1. Check if Python actually exists and is the right version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Python is not detected on this system.
    echo [INFO] Downloading Python 3.12.0...
    curl -o python_installer.exe https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe
    
    echo [INFO] Installing Python silently...
    start /wait python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    
    del python_installer.exe
    echo [SUCCESS] Python installed. Close this command window and run this script again to continue.
    pause
    exit /b
)

:: 2. Create the virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Building virtual environment...
    python -m venv venv
)

:: 3. Activate the environment (Explicitly calling the .bat file)
echo [INFO] Activating environment...
call venv\Scripts\activate.bat

:: 4. Safety Check: Verify we are actually inside the venv
if "%VIRTUAL_ENV%"=="" (
    echo [FATAL ERROR] Failed to activate the virtual environment. 
    echo Halting installation to protect your global system packages.
    pause
    exit /b
)

:: 5. Install dependencies (Optimized to skip if already installed)
if not exist "venv\.installed" (
    echo [INFO] Updating pip...
    python -m pip install --upgrade pip >nul 2>&1

    echo [INFO] Installing required dependencies. This will download a few gigabytes for the AI models. Grab a coffee.
    pip install -r requirements.txt
    
    :: Check if pip install succeeded before creating the marker
    if %errorlevel% equ 0 (
        type nul > venv\.installed
        echo [SUCCESS] Dependencies installed successfully.
    ) else (
        echo [FATAL ERROR] Failed to install dependencies. Check your internet connection.
        pause
        exit /b
    )
) else (
    echo [INFO] Dependencies verified. Booting system...
)

:: 6. Launch the app (Updated to run.py)
echo [INFO] Launching the application...
python run.py

echo.
echo [SYSTEM] Application closed or crashed. Check the logs above.
pause