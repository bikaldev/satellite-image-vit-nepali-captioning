@echo off
echo ==========================================
echo Starting Satellite Image Training Pipeline
echo ==========================================
echo.

if exist "env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call env\Scripts\activate.bat
) else (
    echo Warning: env\Scripts\activate.bat not found. Trying global python...
)

echo.
echo Step 1: Training Classification Model
echo -------------------------------------
python train_classifier.py --config configs/config.yaml --output_dir outputs/classification
if %errorlevel% neq 0 (
    echo Error: Classification training failed!
    exit /b %errorlevel%
)

echo.
echo Step 2: Training Captioning Model
echo ---------------------------------
python train_captioner.py --config configs/config.yaml --output_dir outputs/captioning
if %errorlevel% neq 0 (
    echo Error: Captioning training failed!
    exit /b %errorlevel%
)

echo.
echo ==========================================
echo Training Pipeline Completed Successfully!
echo ==========================================
echo.
echo Press any key to exit...
pause >nul
