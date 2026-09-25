@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python nahi mila. Pehle https://www.python.org/downloads/ se Python 3.12 install karo. Install karte waqt Add python.exe to PATH tick karo.
  pause
  exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
  echo Node.js nahi mila. Pehle https://nodejs.org se LTS version install karo.
  pause
  exit /b 1
)
cd backend
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install onnxruntime tokenizers
cd ..\frontend
call npm install
call npm run build
cd ..
echo.
echo Setup complete. Ab start.bat par double-click karo.
pause
