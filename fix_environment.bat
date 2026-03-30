@echo off
echo ==========================================
echo Reinstalling Virtual Environment
echo ==========================================

if exist env (
    echo Removing existing env...
    rmdir /s /q env
)

echo Creating new virtual environment...
python -m venv env

echo Upgrading pip...
call env\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt
pip install openpyxl

echo.
echo ==========================================
echo Environment setup complete!
echo ==========================================
pause
