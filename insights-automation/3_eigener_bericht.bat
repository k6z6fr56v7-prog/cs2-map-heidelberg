@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Baut den Bericht aus deinen eigenen Daten im Ordner "daten".
echo Erwartet daten\mapping.json (siehe README) oder daten\daten.csv im internen Format.
echo.
if exist "daten\mapping.json" (
  ".venv\Scripts\python.exe" -m insights bericht --daten daten\mapping.json --ausgabe berichte
) else (
  ".venv\Scripts\python.exe" -m insights bericht --daten daten\daten.csv --ausgabe berichte
)
if not errorlevel 1 start "" "berichte"
pause
