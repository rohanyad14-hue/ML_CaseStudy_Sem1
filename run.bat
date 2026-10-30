@echo off
title ML_CaseStudy
echo =======================================================================
echo        ML_CaseStudy 
echo =======================================================================
echo.s

echo [1/2] Checking and installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements. Please ensure Python is installed and added to PATH.
    pause
    exit /b %errorlevel%
)
echo.

echo [2/2] Launching Interactive ML Classifyer Dashboard...
echo Opening in default browser at http://localhost:8501
python -m streamlit run app.py
pause
