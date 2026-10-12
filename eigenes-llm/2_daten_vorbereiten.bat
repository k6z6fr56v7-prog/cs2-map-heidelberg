@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Laedt deutschen Webtext (70 %%) und Wikipedia (30 %%), trainiert den Tokenizer und bereitet alles vor.
echo Das dauert je nach Internet 30-90 Minuten und braucht ca. 5 GB Platz.
echo.
".venv\Scripts\python.exe" -m llm vorbereiten --preset klein
pause
