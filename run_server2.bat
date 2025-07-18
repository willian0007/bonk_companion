@echo off
setlocal

:: =================================================================
:: ==            PORTABLE UVICORN SERVER LAUNCHER                 ==
:: =================================================================
:: This script is designed to be run from any location. It uses
:: paths relative to its own location to find the necessary files
:: and virtual environment.

echo.
echo --- Locating Script and Project Root ---
REM %~dp0 is the magic variable: it's the Drive and Path of this batch file.
SET "PROJECT_DIR=%~dp0"
echo Batch file is located in: "%PROJECT_DIR%"

REM --- Attempt to Activate Virtual Environment ---
SET "VENV_ACTIVATION_SCRIPT=%PROJECT_DIR%..\venv\Scripts\activate.bat"
echo Checking for venv activation script at: "%VENV_ACTIVATION_SCRIPT%"
IF EXIST "%VENV_ACTIVATION_SCRIPT%" (
    echo Found venv activation script. Activating...
    CALL "%VENV_ACTIVATION_SCRIPT%"
) ELSE (
    echo --------------------------------------------------------------------
    echo WARNING: venv activation script NOT FOUND at the expected location.
    echo Expected Path: "%VENV_ACTIVATION_SCRIPT%"
    echo Please ensure the 'venv' folder is located one level above the
    echo folder containing this script.
    echo --------------------------------------------------------------------
    GOTO :error
)

echo.
echo --- Setting Environment Variables (Relative Paths) ---
set "ROBOT_MODEL_PATH=%PROJECT_DIR%ckpts\model_650000_FP16.pt"
set "ROBOT_VOCAB_PATH=%PROJECT_DIR%vocab\vocab.txt"
set "ROBOT_REF_AUDIO_PATH=%PROJECT_DIR%soundtest\welp.wav"
set "GEMINI_API_KEY=ใส่ gemini api key ลงตรงนี้"
set "VOCODER_MODEL_PATH=%PROJECT_DIR%vocoder"
REM ====================================================================
REM  NEW: Path to the local Whisper model folder
REM ====================================================================
set "WHISPER_MODEL_PATH=%PROJECT_DIR%whisper"


echo Model Path set to: %ROBOT_MODEL_PATH%
echo Vocab Path set to: %ROBOT_VOCAB_PATH%
echo Ref Audio Path set to: %ROBOT_REF_AUDIO_PATH%
echo Whisper Model Path set to: %WHISPER_MODEL_PATH%
if defined GEMINI_API_KEY (
    echo GEMINI_API_KEY is set.
) else (
    echo WARNING: GEMINI_API_KEY IS NOT SET!
)
echo Environment variables set.

echo.
echo --- Path and Python Diagnostics ---
echo Python executable being used:
where python
echo.
echo Uvicorn executable being used:
where uvicorn
echo.

echo --- Attempting to Start Uvicorn Server ---
SET "PYTHON_EXE=%PROJECT_DIR%..\venv\Scripts\python.exe"

echo Running command:
echo "%PYTHON_EXE%" -m uvicorn main_api:app --reload --host 0.0.0.0 --port 8000 --app-dir "%PROJECT_DIR%"
echo =================================================================
"%PYTHON_EXE%" -m uvicorn main_api:app --reload --host 0.0.0.0 --port 8000 --app-dir "%PROJECT_DIR%"

echo.
echo Server execution finished, or failed to start.
GOTO :end

:error
echo.
echo Script aborted due to an error.

:end
echo Press any key to exit.
pause
