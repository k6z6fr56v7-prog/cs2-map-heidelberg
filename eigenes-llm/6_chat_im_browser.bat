@echo off
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" -m llm web --modell arbeit\chat_klein --name "Eigen-LM" --oeffnen
pause
