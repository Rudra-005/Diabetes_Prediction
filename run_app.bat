@echo off
echo ============================================================
echo Starting Diabetes Prediction Web App
echo ============================================================
echo.
echo Installing Streamlit if not already installed...
pip install streamlit
echo.
echo Launching web application...
echo The app will open in your browser automatically.
echo.
streamlit run app.py
pause