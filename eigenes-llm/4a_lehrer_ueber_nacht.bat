@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Dein Ollama-Modell (z.B. Qwen) schreibt 8 Stunden lang deutsche Chat-Beispiele
echo fuer dein eigenes Modell. Ollama muss laufen. Nicht gleichzeitig trainieren!
echo Beenden mit Strg+C - beim naechsten Start geht es weiter.
echo.
".venv\Scripts\python.exe" -m llm lehrer --stunden 8
pause
