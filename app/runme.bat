@echo off
cd /d "%~dp0"

set PORT=8000

start powershell -NoExit -Command "cd '%~dp0'; uvicorn main:app --port %PORT% --reload"

timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:%PORT%"

exit