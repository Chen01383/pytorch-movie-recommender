@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" demo.py
) else (
    echo Create the Python environment and install requirements.txt first.
)
pause
