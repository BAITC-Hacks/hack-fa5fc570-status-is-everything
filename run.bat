@echo off
chcp 65001 > nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
if not exist "models\validated\turbine_1.joblib" goto train
if not exist "models\validated\turbine_2.joblib" goto train
goto launch
:train
  ".venv\Scripts\python.exe" -m src.train
  if errorlevel 1 exit /b 1
:launch
".venv\Scripts\python.exe" -m streamlit run app.py
