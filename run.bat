@echo off
cd /d "%~dp0"

if not exist "venv" (
    echo First run — creating sandbox and installing everything (1-2 min)...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
    echo Ready! Next time will be instant.
) else (
    call venv\Scripts\activate.bat
)

echo Starting CamOCR web UI...
echo Open in browser → http://127.0.0.1:7860
python app.py
pause
