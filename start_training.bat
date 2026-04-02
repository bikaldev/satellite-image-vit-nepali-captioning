@echo off
setlocal
set PYTORCH_ALLOC_CONF=expandable_segments:True

echo ===========================================
echo Starting Satellite Image Captioning Training
echo ===========================================
echo.


:: Check for environment
if not exist "env\Scripts\activate.bat" (
    echo Error: Python environment not found in /env.
    echo Please create it first using: python -m venv env
    pause
    exit /b 1
)

:: Activate environment
echo Activating environment...
call env\Scripts\activate.bat

:: Install missing dependencies (if any)
echo Checking dependencies...
pip install tensorboard rouge-score nltk pyyaml pandas matplotlib tqdm transformers accelerate torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128

:: Start training (standard)
python train_captioner.py --config configs/config.yaml --output_dir outputs/captioning_refined

:: Use this line if you want to resume from the latest checkpoint
:: python train_captioner.py --config configs/config.yaml --output_dir outputs/captioning_refined --resume

:: To fix the reported error (starting from batch 293 of epoch 1):
:: echo Resuming training from last epoch (Epoch 2)...
:: python train_captioner.py --config configs/config.yaml --output_dir outputs/captioning_refined --resume --resume_from outputs/captioning_refined/checkpoints/last_epoch --resume_epoch 1

if %errorlevel% neq 0 (
    echo.
    echo Training failed with error code %errorlevel%.
    pause
    exit /b %errorlevel%
)

echo.
echo ===========================================
echo Training Finished Successfully!
echo ===========================================
pause
