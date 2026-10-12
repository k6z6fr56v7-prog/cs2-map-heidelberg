@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Bringt dem Modell bei, wie ein Assistent auf Fragen zu antworten (ca. 1-2 Stunden).
echo.
".venv\Scripts\python.exe" -m llm sft --basis arbeit\modell_klein --name "Eigen-LM" --ersteller "Leo"
pause
