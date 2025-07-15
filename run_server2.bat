@echo off
echo Current batch file directory: %~dp0
echo Parent directory of batch file's dir: %~dp0..\

REM --- Attempt to Activate Virtual Environment ---
SET VENV_ACTIVATION_SCRIPT=%~dp0..\venv\Scripts\activate.bat
echo Checking for venv activation script at: "%VENV_ACTIVATION_SCRIPT%"
IF EXIST "%VENV_ACTIVATION_SCRIPT%" (
    echo Found venv activation script. Activating...
    CALL "%VENV_ACTIVATION_SCRIPT%"
) ELSE (
    echo --------------------------------------------------------------------
    echo WARNING: venv activation script NOT FOUND at "%VENV_ACTIVATION_SCRIPT%"
    echo Please ensure your venv is active manually before running this script,
    echo OR correct the VENV_ACTIVATION_SCRIPT path within this .bat file.
    echo If venv is not active, Python dependencies might not be found.
    echo --------------------------------------------------------------------
)

echo.
echo --- Setting Environment Variables ---
set ROBOT_MODEL_PATH=D:\f5tts\F5-TTS-THAI\ckpts\model_1000000.pt
set ROBOT_VOCAB_PATH=D:\f5tts\F5-TTS-THAI\vocab\vocab.txt
set ROBOT_REF_AUDIO_PATH=D:\f5tts\F5-TTS-THAI\src\f5_tts\infer\examples\thai_examples\welp.wav
set GEMINI_API_KEY="my api"
REM Make sure to replace YOUR_ACTUAL_GEMINI_KEY_HERE with your real API key above
echo Environment variables set.

echo.
echo --- Path and Python Diagnostics ---
echo Current Working Directory (should be D:\f5tts\F5-TTS-THAI\): %CD%
echo.
echo Python executable being used by this script (after venv activation attempt):
where python
echo.
echo Uvicorn executable being used by this script:
where uvicorn
echo.

echo --- Attempting to Start Uvicorn Server (Explicit Python Invocation) ---
echo Running: "%~dp0..\venv\Scripts\python.exe" -m uvicorn main_api:app --reload --host 0.0.0.0 --port 8000 --app-dir "%~dp0"
"%~dp0..\venv\Scripts\python.exe" -m uvicorn main_api:app --reload --host 0.0.0.0 --port 8000 --app-dir "%~dp0"

echo.
echo Server execution finished, or failed to start. Press any key to exit.
pause