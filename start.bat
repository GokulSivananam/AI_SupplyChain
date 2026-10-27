@echo off
title AI Supply Chain Disruption Prediction

echo.
echo  ================================================
echo   AI Supply Chain Disruption Prediction
echo  ================================================
echo.

:: Move to the project root (folder where this .bat lives)
cd /d "%~dp0"

:: Check Python is available
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python launcher ^(py^) not found.
    echo  Install Python from https://www.python.org/downloads/
    echo  Make sure "Add Python to PATH" is checked during install.
    pause
    exit /b 1
)

:: Check Streamlit is installed
py -m streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [INFO] Streamlit not found. Installing dependencies...
    py -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

:: Check the dataset exists
if not exist "data\supply_chain_data.csv" (
    echo  [ERROR] Dataset not found at data\supply_chain_data.csv
    echo  Please copy your CSV file into the data\ folder.
    pause
    exit /b 1
)

:: Train model if not already trained
if not exist "models\best_model.pkl" (
    echo  [INFO] No trained model found. Running training first...
    echo.
    py src\train_model.py
    if %errorlevel% neq 0 (
        echo  [ERROR] Model training failed.
        pause
        exit /b 1
    )
    echo.
    echo  [INFO] Training complete.
)

:: Launch the app
echo  [INFO] Starting Streamlit app...
echo  [INFO] Open your browser at: http://localhost:8501
echo  [INFO] Press Ctrl+C in this window to stop the app.
echo.

py -m streamlit run src/app.py --server.port 8501

pause
