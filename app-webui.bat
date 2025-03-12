@echo off

set "current_dir=%CD%"

call venv/scripts/activate

python src/f5_tts/f5_tts_webui.py

pause
