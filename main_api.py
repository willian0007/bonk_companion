# main_api.py
import os
import sys
import shutil
import tempfile
from contextlib import asynccontextmanager
import random
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Tuple, Optional
import traceback
import json
from fastapi.responses import Response, FileResponse
# --- NEW/MODIFIED IMPORTS ---
from threading import Thread
import pandas as pd

# --- Adjust sys.path and import robot_module ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
try:
    import robot_module as ai_robot
except ImportError as e:
    print(f"CRITICAL ERROR in main_api.py: Could not import robot_module.py. Error: {e}")
    sys.exit(1)

# --- Pydantic Models ---
class TextIn(BaseModel):
    text: str
    history: Optional[List[Tuple[str, str]]] = None

class STTResponse(BaseModel):
    transcribed_text: str
    error: Optional[str] = None
    stt_method: Optional[str] = None

class LLMRequest(BaseModel):
    text_input: str
    history: Optional[List[Tuple[str, str]]] = None

class LLMResponse(BaseModel):
    ai_response: str
    error: Optional[str] = None

class InitStatusResponse(BaseModel):
    initialized: bool
    message: str
    details: Optional[str] = None
    gemini_model_name: Optional[str] = None
    f5_tts_ref_text: Optional[str] = None
    stt_method: Optional[str] = None

class VoiceSettings(BaseModel):
    ref_audio_path: str
    ref_text_hint: Optional[str] = ""

class TTSSettings(BaseModel):
    speed: Optional[float] = 0.8
    max_chars: Optional[int] = 250
    nfe_step: Optional[int] = 16
    cfg_strength: Optional[float] = 2.0

class GeminiSettings(BaseModel):
    model_name: Optional[str] = "models/gemini-2.5-flash-lite-preview-06-17"
    system_prompt: Optional[str] = "เป็นเพื่อนคุยแก้เหงา ไม่พิมพ์ อักขระ ไม่ใช้ตัว ๆ ให้พิมพ์ซ้ำแทน ไม่พิมพ์ภาษาอังกฤษ พิมพ์แค่คำทับศัพท์ ไม่ใช้เลขอารบิก เช่น 1 2 แต่ใช้คำแทน เช่น หนึ่ง สอง "

class AllSettings(BaseModel):
    voice_settings: Optional[VoiceSettings] = None
    tts_settings: Optional[TTSSettings] = None
    gemini_settings: Optional[GeminiSettings] = None

# ========== NEW SUBTITLE MODEL AND GLOBAL VARIABLE ==========
class SubtitleIn(BaseModel):
    text: str

CURRENT_SUBTITLE = ""
# =========================================================

# --- Lifespan event handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global TEMP_API_DIR
    log_func = getattr(ai_robot, 'DEBUG_LOG', print)
    log_func("FastAPI lifespan.startup: Beginning...")
    try:
        TEMP_API_DIR = tempfile.mkdtemp(prefix="waifu_api_uploads_")
        
        # --- NEW DETAILED CHECKING ---
        log_func("--- Checking Environment Variables and Paths ---")
        model_p = os.getenv("ROBOT_MODEL_PATH")
        vocab_p = os.getenv("ROBOT_VOCAB_PATH")
        ref_audio_p = os.getenv("ROBOT_REF_AUDIO_PATH")
        gemini_key = os.getenv("GEMINI_API_KEY")

        missing = []
        # Check Model Path
        log_func(f"Checking ROBOT_MODEL_PATH: {model_p}")
        if not model_p or not os.path.exists(model_p):
            missing.append("ROBOT_MODEL_PATH")
            log_func("--> STATUS: NOT FOUND!")
        else:
            log_func("--> STATUS: Found.")

        # Check Vocab Path
        log_func(f"Checking ROBOT_VOCAB_PATH: {vocab_p}")
        if not vocab_p or not os.path.exists(vocab_p):
            missing.append("ROBOT_VOCAB_PATH")
            log_func("--> STATUS: NOT FOUND!")
        else:
            log_func("--> STATUS: Found.")

        # Check Ref Audio Path
        log_func(f"Checking ROBOT_REF_AUDIO_PATH: {ref_audio_p}")
        if not ref_audio_p or not os.path.exists(ref_audio_p):
            missing.append("ROBOT_REF_AUDIO_PATH")
            log_func("--> STATUS: NOT FOUND!")
        else:
            log_func("--> STATUS: Found.")

        # Check Gemini Key
        log_func(f"Checking GEMINI_API_KEY: {'Set' if gemini_key else 'Not Set'}")
        if not gemini_key:
            missing.append("GEMINI_API_KEY")
            log_func("--> STATUS: NOT FOUND!")
        else:
            log_func("--> STATUS: Found.")
        
        log_func("--------------------------------------------------")

        if missing:
            error_message = "Initialization pre-check failed for: " + ", ".join(missing)
            ai_robot.initialization_error_msg_global = error_message
            log_func(f"FATAL: {error_message}")
        else:
            log_func("FastAPI lifespan.startup: All paths and keys found. Initializing all AI models...")
            ai_robot.initialize_all_models(model_path=model_p, vocab_path=vocab_p, ref_audio_path=ref_audio_p, api_key=gemini_key)
            if ai_robot.initialization_error_msg_global is None:
                log_func("FastAPI lifespan.startup: AI models initialized successfully.")
                ai_robot.start_default_livelink_animation_stream()
            else:
                log_func(f"ERROR during model initialization: {ai_robot.initialization_error_msg_global}")

    except Exception as e:
        error_details = f"Fatal error during startup: {e}\n{traceback.format_exc()}"
        ai_robot.initialization_error_msg_global = error_details
        log_func(error_details)
    
    yield
    
    log_func("FastAPI lifespan.shutdown: Cleaning up...")
    if TEMP_API_DIR and os.path.exists(TEMP_API_DIR):
        try: shutil.rmtree(TEMP_API_DIR)
        except Exception: pass

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Waifu AI Brain API",
    description="API to interact with the AI Waifu's STT, LLM, TTS and LiveLink Animation capabilities.",
    version="5.1.0-SubtitleStream",
    lifespan=lifespan
)

def check_models_initialized():
    if ai_robot.initialization_error_msg_global is not None:
        raise HTTPException(status_code=503, detail=f"AI Models not initialized: {ai_robot.initialization_error_msg_global}")

# --- API Endpoints ---
@app.get("/status", response_model=InitStatusResponse)
async def get_status_endpoint():
    try:
        check_models_initialized()
        stt_method = "Gemini API" if ai_robot.USE_GEMINI_STT else "Local Whisper"
        return InitStatusResponse(
            initialized=True, 
            message="All AI models initialized successfully.",
            stt_method=stt_method
        )
    except HTTPException as http_exc:
        return InitStatusResponse(
            initialized=False, 
            message="AI models NOT initialized.", 
            details=str(http_exc.detail)
        )

# --- PRIMARY STREAMING ENDPOINT ---
@app.post("/generate_streamed_response")
async def generate_streamed_response_endpoint(request: TextIn):
    check_models_initialized()
    log_func = getattr(ai_robot, 'DEBUG_LOG', print)
    try:
        stream_thread = Thread(
            target=ai_robot.generate_and_stream_response_threaded,
            args=(request.text, request.history)
        )
        stream_thread.start()

        log_func("API: Triggered background streaming process.")
        return {"status": "success", "message": "AI response stream initiated."}
    except Exception as e:
        log_func(f"API Error: Failed to start stream thread: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to start streaming process: {str(e)}")


# ========== NEW SUBTITLE ENDPOINTS ==========
@app.post("/update_current_subtitle")
async def update_subtitle(subtitle: SubtitleIn):
    """Internal endpoint for the producer thread to update the subtitle."""
    global CURRENT_SUBTITLE
    CURRENT_SUBTITLE = subtitle.text
    return {"status": "success", "current_subtitle": CURRENT_SUBTITLE}

@app.get("/get_current_subtitle")
async def get_subtitle():
    """Endpoint for Unreal Engine to poll for the latest subtitle."""
    global CURRENT_SUBTITLE
    return {"subtitle": CURRENT_SUBTITLE}
# ==========================================


# --- The following endpoints are kept for debugging or other workflows ---

@app.post("/transcribe", response_model=STTResponse)
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...), purpose: str = Form("command")):
    check_models_initialized()
    temp_audio_path = os.path.join(TEMP_API_DIR, "stt_temp.wav")
    log_func = getattr(ai_robot, 'DEBUG_LOG', print)
    
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        stt_method_used = "Gemini API" if ai_robot.USE_GEMINI_STT else "Local Whisper"
        log_func(f"STT: Processing with {stt_method_used}, purpose: {purpose}")
        transcribed_text = ai_robot.perform_stt(temp_audio_path, purpose=purpose)
        
        if ai_robot.USE_GEMINI_STT and ai_robot.ASR_PIPE is not None:
            actual_method = "Gemini API (with Whisper fallback available)"
        elif ai_robot.USE_GEMINI_STT:
            actual_method = "Gemini API"
        else:
            actual_method = "Local Whisper"
        
        log_func(f"STT: Successfully transcribed using {actual_method}: '{transcribed_text}'")
        return STTResponse(transcribed_text=transcribed_text, stt_method=actual_method)
    except Exception as e:
        log_func(f"STT: Error during transcription: {e}")
        error_msg = str(e)
        if "Gemini" in error_msg and ai_robot.ASR_PIPE is None:
            error_msg += " (No Whisper fallback available)"
        elif "Gemini" in error_msg:
            error_msg += " (Whisper fallback also failed)"
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except: pass

# ... (The rest of the file remains unchanged) ...
# ... (get_llm_response, trigger_tts_and_livelink, tests, settings endpoints, etc.) ...
@app.post("/get_llm_response", response_model=LLMResponse, deprecated=True)
async def get_llm_response_api_endpoint(request: LLMRequest):
    """DEPRECATED for main workflow. Use /generate_streamed_response instead."""
    check_models_initialized()
    try:
        return LLMResponse(ai_response=ai_robot.get_gemini_response(request.text_input, history=request.history))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trigger_tts_and_livelink", deprecated=True)
async def trigger_tts_and_livelink_endpoint(request: TextIn):
    """DEPRECATED for main workflow. Use /generate_streamed_response instead."""
    check_models_initialized()
    log_func = getattr(ai_robot, 'DEBUG_LOG', print)
    try:
        audio_file_path = ai_robot.perform_tts(request.text)
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(audio_file_path)
        blendshape_data = ai_robot.perform_blendshape_generation(audio_bytes)
        stream_thread = Thread(target=ai_robot.play_and_stream_animation, args=(audio_bytes, blendshape_data))
        stream_thread.start()
        log_func("LIVELINK: Triggered background streaming process.")
        return {"status": "success", "message": "TTS and LiveLink stream initiated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# (The rest of your endpoints like /test_english_wav, settings endpoints, etc., remain unchanged)
@app.post("/test_english_wav")
async def test_english_wav_endpoint():
    check_models_initialized()
    log_func = getattr(ai_robot, 'DEBUG_LOG', print)
    english_wav_path = r"D:\f5tts\NeuroSync_Player-main\wav_input\audio.wav"
    try:
        if not os.path.exists(english_wav_path):
            log_func(f"TEST_WAV: File not found at {english_wav_path}")
            raise HTTPException(status_code=404, detail="Test WAV file not found on server.")
        with open(english_wav_path, "rb") as f:
            audio_bytes = f.read()
        log_func(f"TEST_WAV: Generating blendshapes for {english_wav_path}")
        blendshape_data = ai_robot.perform_blendshape_generation(audio_bytes)
        stream_thread = Thread(target=ai_robot.play_and_stream_animation, args=(audio_bytes, blendshape_data))
        stream_thread.start()
        log_func("TEST_WAV: Triggered background streaming process.")
        return {"status": "success", "message": "English WAV test and LiveLink stream initiated."}
    except Exception as e:
        log_func(f"TEST_WAV Endpoint Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/test_generate_and_play")
async def test_generate_and_play_endpoint():
    """
    Loads a hardcoded audio file, generates blendshapes on the fly,
    and then streams both to Unreal Engine.
    This is a pure test of the blendshape model's performance.
    """
    check_models_initialized()
    log_func = getattr(ai_robot, 'DEBUG_LOG', print)

    # --- HARDCODED PATH TO YOUR TEST AUDIO FILE ---
    # Make sure this file exists on your server machine
    test_audio_path = r"D:\ue5\bonk_companion\soundtest\welp.wav"
    # ----------------------------------------------

    try:
        if not os.path.exists(test_audio_path):
            log_func(f"TEST_GENERATE: File not found at {test_audio_path}")
            raise HTTPException(status_code=404, detail="Test audio file not found on server.")

        # Step 1: Read the audio file into memory
        with open(test_audio_path, "rb") as f:
            audio_bytes = f.read()

        # Step 2: Generate blendshapes ON THE FLY
        log_func(f"TEST_GENERATE: Generating blendshapes for {os.path.basename(test_audio_path)}...")
        blendshape_data = ai_robot.perform_blendshape_generation(audio_bytes)
        log_func(f"TEST_GENERATE: Blendshape generation complete.")

        # Step 3: Start the audio playback and Live Link stream in a background thread
        stream_thread = Thread(target=ai_robot.play_and_stream_animation, args=(audio_bytes, blendshape_data))
        stream_thread.start()

        log_func("TEST_GENERATE: Triggered background streaming process.")
        return {"status": "success", "message": "On-the-fly blendshape generation test initiated."}

    except Exception as e:
        log_func(f"TEST_GENERATE Endpoint Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/debug_stt_methods")
async def debug_stt_methods_endpoint(audio_file: UploadFile = File(...)):
    check_models_initialized()
    temp_audio_path = os.path.join(TEMP_API_DIR, "debug_stt.wav")
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        results = {}
        if ai_robot.gemini_model is not None:
            try:
                results["gemini_stt"] = {"result": ai_robot.perform_stt_gemini(temp_audio_path), "status": "success"}
            except Exception as e:
                results["gemini_stt"] = {"result": None, "status": "failed", "error": str(e)}
        else:
            results["gemini_stt"] = {"result": None, "status": "not_available", "error": "Gemini model not initialized"}
        if ai_robot.ASR_PIPE is not None:
            try:
                results["whisper_stt"] = {"result": ai_robot.perform_stt_whisper_fallback(temp_audio_path), "status": "success"}
            except Exception as e:
                results["whisper_stt"] = {"result": None, "status": "failed", "error": str(e)}
        else:
            results["whisper_stt"] = {"result": None, "status": "not_available", "error": "Whisper model not initialized"}
        results["config"] = {"use_gemini_stt": ai_robot.USE_GEMINI_STT, "gemini_available": ai_robot.gemini_model is not None, "whisper_available": ai_robot.ASR_PIPE is not None}
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except: pass

@app.post("/toggle_stt_method")
async def toggle_stt_method_endpoint():
    try:
        ai_robot.USE_GEMINI_STT = not ai_robot.USE_GEMINI_STT
        new_method = "Gemini API" if ai_robot.USE_GEMINI_STT else "Local Whisper"
        return {"status": "success", "message": f"STT method switched to: {new_method}", "current_method": new_method, "use_gemini_stt": ai_robot.USE_GEMINI_STT}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_current_settings")
async def get_current_settings():
    return ai_robot.get_current_settings_dict()

@app.post("/update_voice_settings")
async def update_voice_settings(settings: VoiceSettings):
    check_models_initialized()
    try:
        error = ai_robot.update_reference_audio_globals(settings.ref_audio_path, settings.ref_text_hint)
        if error: raise HTTPException(status_code=400, detail=f"Failed to update voice settings: {error}")
        ai_robot.current_ref_audio_path = settings.ref_audio_path
        return {"status": "success", "message": "Voice settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_tts_settings")
async def update_tts_settings(settings: TTSSettings):
    check_models_initialized()
    try:
        if settings.speed is not None: ai_robot.DEFAULT_TTS_SPEED = settings.speed
        if settings.max_chars is not None: ai_robot.DEFAULT_TTS_MAX_CHARS = settings.max_chars
        if settings.nfe_step is not None: ai_robot.DEFAULT_NFE_STEP = settings.nfe_step
        if settings.cfg_strength is not None: ai_robot.DEFAULT_CFG_STRENGTH = settings.cfg_strength
        return {"status": "success", "message": "TTS settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_gemini_settings")
async def update_gemini_settings(settings: GeminiSettings):
    check_models_initialized()
    try:
        if settings.model_name is not None:
            ai_robot.SPECIFIC_GEMINI_MODEL_NAME = settings.model_name
            import google.generativeai as genai
            ai_robot.gemini_model = genai.GenerativeModel(settings.model_name)
        if settings.system_prompt is not None:
            ai_robot.SYSTEM_PROMPT = settings.system_prompt
        return {"status": "success", "message": "Gemini settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_all_settings")
async def update_all_settings(settings: AllSettings):
    check_models_initialized()
    results = {}
    try:
        if settings.voice_settings:
            error = ai_robot.update_reference_audio_globals(settings.voice_settings.ref_audio_path, settings.voice_settings.ref_text_hint)
            if error: results["voice_settings"] = {"status": "error", "message": error}
            else:
                ai_robot.current_ref_audio_path = settings.voice_settings.ref_audio_path
                results["voice_settings"] = {"status": "success"}
        if settings.tts_settings:
            if settings.tts_settings.speed is not None: ai_robot.DEFAULT_TTS_SPEED = settings.tts_settings.speed
            if settings.tts_settings.max_chars is not None: ai_robot.DEFAULT_TTS_MAX_CHARS = settings.tts_settings.max_chars
            if settings.tts_settings.nfe_step is not None: ai_robot.DEFAULT_NFE_STEP = settings.tts_settings.nfe_step
            if settings.tts_settings.cfg_strength is not None: ai_robot.DEFAULT_CFG_STRENGTH = settings.tts_settings.cfg_strength
            results["tts_settings"] = {"status": "success"}
        if settings.gemini_settings:
            if settings.gemini_settings.model_name is not None:
                ai_robot.SPECIFIC_GEMINI_MODEL_NAME = settings.gemini_settings.model_name
                import google.generativeai as genai
                ai_robot.gemini_model = genai.GenerativeModel(settings.gemini_settings.model_name)
            if settings.gemini_settings.system_prompt is not None:
                ai_robot.SYSTEM_PROMPT = settings.gemini_settings.system_prompt
            results["gemini_settings"] = {"status": "success"}
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_available_models")
async def list_available_models():
    try:
        available_models = ["models/gemini-2.5-flash", "models/gemini-2.5-flash-lite-preview-06-17", "models/gemini-1.5-pro", "models/gemini-1.0-pro"]
        return {"models": available_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_to_default_settings")
async def reset_settings_to_default():
    check_models_initialized()
    try:
        ai_robot.reset_to_default_settings()
        return {"status": "success", "message": "All settings have been reset to their default values."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
