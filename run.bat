@echo off
setlocal enabledelayedexpansion
title MNIST PCA Digit Classifier - One-Click Launcher

:: Change to script directory
cd /d "%~dp0"

echo =====================================================================
echo  MACSE504 Machine Learning - Course Based Design Project (Q7)
echo  Student: Rohan Yadav ^| Reg No: 26MML0020
echo  MNIST Complexity Reduction ^& PCA Edge Classifier
echo =====================================================================
echo.

:: 1. Check for Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
    set "PY_CMD=python"
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
    set "PY_CMD=python"
) else (
    echo [INFO] No local virtual environment found. Searching for system Python...
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=python"
    ) else (
        where py >nul 2>&1
        if !errorlevel! equ 0 (
            set "PY_CMD=py"
        ) else (
            echo [ERROR] Python was not found in PATH or a local virtual environment!
            echo Please install Python 3.10+ or add Python to your system PATH.
            echo.
            pause
            exit /b 1
        )
    )
)

:: 2. Verify / Install Dependencies
echo [INFO] Verifying core packages...
!PY_CMD! -c "import streamlit, sklearn, plotly, PIL" >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARNING] Missing dependencies detected. Installing requirements from requirements.txt...
    !PY_CMD! -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [ERROR] Dependency installation failed! Check your internet connection or Python configuration.
        pause
        exit /b 1
    )
)

:: 3. Launch Streamlit Application
echo.
echo =====================================================================
echo  Starting Streamlit Application (app.py)...
echo  Your default web browser will open automatically.
echo  To stop the application, press Ctrl+C in this terminal.
echo =====================================================================
echo.

!PY_CMD! -m streamlit run app.py

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Application terminated unexpectedly.
    pause
)
