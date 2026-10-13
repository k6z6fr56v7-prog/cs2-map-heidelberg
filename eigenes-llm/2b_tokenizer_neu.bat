@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Trainiert den Tokenizer neu und tokenisiert die bereits geladenen Texte noch einmal.
echo Nichts wird neu heruntergeladen. Dauer ca. 15-30 Minuten.
echo.
".venv\Scripts\python.exe" -m llm tokenizer
if errorlevel 1 goto ende
".venv\Scripts\python.exe" -m llm tokenisieren
:ende
pause
