@echo off

echo Creating virtual environment...
python -m venv venv

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install required dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git

REM Run the application
echo Running f5_tts_webui.py...
python src/f5_tts/f5_tts_webui.py

REM Deactivate the virtual environment
echo Deactivating virtual environment...
deactivate

echo Setup complete. The virtual environment is ready.
pause