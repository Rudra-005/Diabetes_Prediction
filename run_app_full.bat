@echo off
echo ============================================================
echo Diabetes Prediction Web App - Full Setup
echo ============================================================
echo.
echo Step 1: Training model...
python train_model.py
echo.
echo Step 2: Launching web app...
streamlit run app.py
pause