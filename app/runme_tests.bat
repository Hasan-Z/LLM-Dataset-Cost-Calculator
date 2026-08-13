@echo off
echo ============================================================
echo   LLM Cost Calculator - Test Suite
echo ============================================================
echo.

REM Run pytest from the project root
python -m pytest tests/ -v --tb=short 2>&1

echo.
echo ============================================================
echo   Done. See output above for results.
echo ============================================================
pause
