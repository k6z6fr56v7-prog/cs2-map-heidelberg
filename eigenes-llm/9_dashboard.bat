@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Zeigt den Trainingsfortschritt im Browser - einfach neben dem Training starten.
echo Die Adresse fuer das Handy (gleiches WLAN) steht gleich unten.
echo.
".venv\Scripts\python.exe" -m llm dashboard --netzwerk --oeffnen
pause
