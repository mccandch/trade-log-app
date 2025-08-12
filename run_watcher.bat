@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
setlocal enabledelayedexpansion

rem === CONFIG ===
set "BASE=C:\Users\Administrator\Documents\Trade Stats"
set "PY=%BASE%\.venv\Scripts\python.exe"
set "SCRIPT=watcher_db.py"
set "LOG=%BASE%\watcher.log"
rem ==============

rem Sanity checks
if not exist "%PY%" (
  echo [%DATE% %TIME%] ERROR: Python not found: "%PY%" > "%LOG%"
  echo Python not found: "%PY%"
  goto :fail
)
if not exist "%BASE%\%SCRIPT%" (
  echo [%DATE% %TIME%] ERROR: Script not found: "%BASE%\%SCRIPT%" > "%LOG%"
  echo Script not found: "%BASE%\%SCRIPT%"
  goto :fail
)

rem Optional: check your creds/db paths inside watcher_db.py
rem if not exist "C:\path\to\service_account.json" echo Missing service_account.json>>"%LOG%"
rem if not exist "C:\path\to\data.db3" echo Missing data.db3>>"%LOG%"

echo [%DATE% %TIME%] Starting watcher >> "%LOG%"
cd /d "%BASE%"
"%PY%" -u "%SCRIPT%" >> "%LOG%" 2>&1
set ERR=%ERRORLEVEL%
echo [%DATE% %TIME%] Exit code %ERR% >> "%LOG%"

if %ERR% NEQ 0 goto :fail
exit /b 0

:fail
echo.
echo Watcher failed (exit %ERR%). See log:
echo   "%LOG%"
echo.
pause
