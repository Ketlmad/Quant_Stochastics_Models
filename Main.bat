@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Single entry point: prepare Barchart once, then run SS and SSX (default).
call "%~dp0Bootstrap.bat"
if errorlevel 1 exit /b 1

set "VENV_PY=%~dp0..\.venv\Scripts\python.exe"
"%VENV_PY%" "%~dp0modele.py" %*
set "RUN_EXIT_CODE=%ERRORLEVEL%"
if not "%RUN_EXIT_CODE%"=="0" echo [ERROR] Schwartz-Smith execution failed (exit %RUN_EXIT_CODE%).
exit /b %RUN_EXIT_CODE%
