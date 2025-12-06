@echo off
echo ============================================================
echo DIABETES PREDICTION CHALLENGE - WINDOWS SETUP
echo ============================================================

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Checking if Jupyter is installed...
jupyter --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Jupyter...
    pip install jupyter
)

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo You can now run:
echo 1. jupyter notebook diabetes_prediction.ipynb
echo 2. python diabetes_prediction.py
echo ============================================================
pause