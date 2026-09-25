@echo off
cd /d "%~dp0backend"
call .venv\Scripts\activate
start "" cmd /c "timeout /t 5 >nul & start http://127.0.0.1:8000"
python run.py
pause
