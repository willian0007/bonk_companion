@echo off

REM Save the current directory
set "current_dir=%CD%"

REM Change to the location of Python executable
call venv/scripts/activate

REM Run the Python script using the activated Python environment
f5-tts_finetune-gradio

REM Return to the original directory
cd /d %current_dir%

REM Pause to keep the command prompt window open (optional)
pause
