@echo off
setlocal

echo ===========================================
echo Starting Comparative Model Evaluation
echo ===========================================
echo.

:: Check for environment
if not exist "env\Scripts\activate.bat" (
    echo Error: Python environment not found in /env.
    pause
    exit /b 1
)

:: Activate environment
echo Activating environment...
call env\Scripts\activate.bat

:: Install any missing dependency
echo Checking dependencies...
pip install rouge-score nltk torchvision torch transformers matplotlib seaborn scikit-learn tqdm pyyaml pandas --extra-index-url https://download.pytorch.org/whl/cu128

:: Start comparative training for all models
:: Models used: 
:: 1. ResNet50 Classifier (resnet50_clf)
:: 2. VGG16 Classifier (vgg16_clf)
:: 3. ResNet50 + LSTM Captioner (resnet50_lstm)
:: 4. VGG16 + LSTM Captioner (vgg16_lstm)
:: 5. ResNet50 + mGPT Captioner (resnet50_mgpt)

echo Starting comprehensive evaluation (this may take some time)...
python train_comparative.py --models resnet50_clf vgg16_clf resnet50_lstm vgg16_lstm resnet50_mgpt --config configs/config.yaml --output_dir outputs/comparative_results

if %errorlevel% neq 0 (
    echo.
    echo Comparative evaluation failed with error code %errorlevel%.
    pause
    exit /b %errorlevel%
)

echo.
echo ===========================================
echo Evaluation Finished Successfully!
echo Results stored in: outputs/comparative_results
echo ===========================================
pause
