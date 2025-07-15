@echo off

REM Define key paths
set "CODE_REPO_DIR=D:\f5tts\F5-TTS-THAI"
set "VENV_ACTIVATE_SCRIPT=D:\f5tts\venv\Scripts\activate.bat"
set "PYTHON_SCRIPT_IN_REPO=src\f5_tts\f5_tts_webui.py"

REM Change to the Code Repository directory
REM This is important for relative paths used in the Python script (e.g., Gradio examples)
echo Changing directory to: %CODE_REPO_DIR%
cd /D "%CODE_REPO_DIR%"

REM Check if directory change was successful
if errorlevel 1 (
    echo ERROR: Could not change directory to %CODE_REPO_DIR%.
    echo Please ensure the path is correct and the directory exists.
    pause
    exit /b 1
)

REM Activate the virtual environment
echo Activating virtual environment from: %VENV_ACTIVATE_SCRIPT%
if not exist "%VENV_ACTIVATE_SCRIPT%" (
    echo ERROR: Virtual environment activate script not found at %VENV_ACTIVATE_SCRIPT%
    pause
    exit /b 1
)
call "%VENV_ACTIVATE_SCRIPT%"

REM Set PYTHONPATH to include the 'src' directory.
REM This allows Python to find the 'f5_tts' package correctly (e.g., "from f5_tts.model import ...")
REM %CD% here will be D:\f5tts\F5-TTS-THAI
set "PYTHONPATH_ADDITION=%CD%\src"
echo Setting PYTHONPATH to include: %PYTHONPATH_ADDITION%
set "PYTHONPATH=%PYTHONPATH_ADDITION%;%PYTHONPATH%" 
REM Prepending to give it priority, and including existing PYTHONPATH if any.
REM Simpler if you don't have an existing PYTHONPATH to preserve: set "PYTHONPATH=%PYTHONPATH_ADDITION%"

REM Run the Python script
echo Starting Python WebUI: %PYTHON_SCRIPT_IN_REPO%
if not exist "%PYTHON_SCRIPT_IN_REPO%" (
    echo ERROR: Python script not found at %CD%\%PYTHON_SCRIPT_IN_REPO%
    pause
    exit /b 1
)
python "%PYTHON_SCRIPT_IN_REPO%"

echo.
echo Python script finished or exited.

REM Optional: Clean up PYTHONPATH if it was set by this script and not pre-existing.
REM This is more complex to do perfectly, so often omitted for simplicity.
REM For now, we'll leave it set as the cmd session will close anyway.
REM If you needed to unset only what you added:
REM (Requires knowing if PYTHONPATH was empty before)
REM set "PYTHONPATH=" 

pause