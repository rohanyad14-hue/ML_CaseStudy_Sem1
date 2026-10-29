#!/usr/bin/env bash
# ==============================================================================
# MACSE504 Machine Learning - Course Based Design Project (Q7)
# Student: Rohan Yadav | Reg No: 26MML0020
# MNIST Complexity Reduction & PCA Edge Classifier
# One-Click Run Script for Linux & macOS
# ==============================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================================="
echo " MACSE504 Machine Learning - Course Based Design Project (Q7)"
echo " Student: Rohan Yadav | Reg No: 26MML0020"
echo " MNIST Complexity Reduction & PCA Edge Classifier"
echo "====================================================================="
echo ""

# 1. Activate Virtual Environment or Locate Python
PY_CMD=""
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
    PY_CMD="python"
elif [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (venv)..."
    source venv/bin/activate
    PY_CMD="python"
elif command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python 3 was not found on your system!"
    echo "Please install Python 3.10 or higher."
    exit 1
fi

# 2. Check and Install Dependencies
echo "[INFO] Verifying core packages..."
if ! "$PY_CMD" -c "import streamlit, sklearn, plotly, PIL" &>/dev/null; then
    echo "[WARNING] Missing required dependencies. Installing from requirements.txt..."
    "$PY_CMD" -m pip install -r requirements.txt
fi

# 3. Launch Streamlit Application
echo ""
echo "====================================================================="
echo " Starting Streamlit Application (app.py)..."
echo " Your default web browser will open automatically."
echo " To stop the application, press Ctrl+C in this terminal."
echo "====================================================================="
echo ""

"$PY_CMD" -m streamlit run app.py
