@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Trainiert das Sprachmodell (Preset "klein") fuer 8 Stunden.
echo Mit Strg+C kannst du jederzeit abbrechen - der Stand wird gespeichert.
echo Fortsetzen: 3b_weitertrainieren.bat
echo.
".venv\Scripts\python.exe" -m llm trainieren --preset klein --stunden 8
pause
