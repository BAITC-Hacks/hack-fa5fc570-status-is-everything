@echo off
chcp 65001 > nul
echo ========================================================
echo   Samruk WindPilot AI - Launching Dashboard
echo   AO "Samruk-Kazyna" / Samruk-Energy
echo ========================================================
echo.
echo Checking dependencies...
pip install -r requirements.txt
echo.
echo Starting Streamlit Application...
streamlit run app.py
pause
