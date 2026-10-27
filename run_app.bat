@echo off
title AI Supply Chain Disruption Forecast

echo.
echo  ================================================
echo   AI Supply Chain Disruption Forecast
echo  ================================================
echo.

cd /d "%~dp0"

py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found. Install from https://www.python.org/
    pause & exit /b 1
)

py -m streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [INFO] Installing dependencies...
    py -m pip install -r requirements.txt
)

if not exist "data\supply_chain_data.csv" (
    echo  [ERROR] Dataset missing: data\supply_chain_data.csv
    pause & exit /b 1
)

if not exist "models\best_model.pkl" (
    echo  [INFO] Training model for the first time...
    py src\train_model.py
)

echo  [INFO] Open browser at: http://localhost:8502
echo  [INFO] Press Ctrl+C to stop.
echo.

py -m streamlit run src/app.py --server.port 8502

pause
