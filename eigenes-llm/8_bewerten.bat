@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Deutsch-Test: Grammatik, Wissen und Logik fuer Text- und Chat-Modell.
echo.
".venv\Scripts\python.exe" -m llm bewerten --modell arbeit\modell_klein --modell arbeit\chat_klein
pause
