@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Exportiert das Modell als GGUF und richtet es in Ollama ein (falls installiert).
echo Braucht Git (https://git-scm.com) fuer den einmaligen Download von llama.cpp.
echo.
".venv\Scripts\python.exe" -m llm export --modell arbeit\chat_klein --name eigen-lm --ollama
pause
