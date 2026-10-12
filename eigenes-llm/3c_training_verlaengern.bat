@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Verlaengert ein FERTIGES Training: setzt beim Stand vor dem Abkuehlen an
echo und trainiert insgesamt 16 Stunden (also 8 Stunden mehr als vorher).
echo.
".venv\Scripts\python.exe" -m llm trainieren --preset klein --verlaengern --stunden 16
pause
