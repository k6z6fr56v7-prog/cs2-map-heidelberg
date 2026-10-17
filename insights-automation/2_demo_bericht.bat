@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Erzeugt Beispieldaten (erfundene Marken) und baut daraus Excel-Report und PowerPoint-Deck.
echo.
".venv\Scripts\python.exe" -m insights demo --ordner demo
if not errorlevel 1 start "" "demo\berichte"
pause
