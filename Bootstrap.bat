@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Shared Python environment at G:\Mickael\Projets\.venv
rem when this project lives in G:\Mickael\Projets\Stochastic_Models.
set "VENV_DIR=%~dp0..\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [BOOTSTRAP] Creating shared virtual environment: %VENV_DIR%
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        where python >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Python not found. Install Python 3.10+ with venv support.
            exit /b 1
        )
        python -m venv "%VENV_DIR%"
    )
    if errorlevel 1 (
        echo [ERROR] Virtual environment creation failed.
        exit /b 1
    )
)

"%VENV_PY%" -c "import sys; assert sys.version_info >= (3, 10), 'Python >= 3.10 required'"
if errorlevel 1 (
    echo [ERROR] This virtual environment requires Python 3.10 or newer.
    exit /b 1
)

echo [BOOTSTRAP] Installing/verifying requirements...
"%VENV_PY%" -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [ERROR] Python dependencies could not be installed.
    exit /b 1
)
echo [BOOTSTRAP] Ready.
exit /b 0
