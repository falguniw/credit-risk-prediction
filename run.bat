@echo off
echo ============================================
echo   Credit Risk Prediction System - Setup
echo ============================================
echo.

echo [1/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)

echo.
echo [2/3] Training model (this takes ~30 seconds)...
python train_model.py
if errorlevel 1 (
    echo ERROR: Model training failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting server...
echo.
echo ============================================
echo   Open your browser at: http://localhost:5000
echo   Press Ctrl+C to stop the server
echo ============================================
echo.
python backend/app.py
pause
