# robot_module.py
import os
from livelink.animations.default_animation import pause_default_animation, resume_default_animation
import sys
import random
import tempfile
import shutil
import numpy as np
import atexit
import warnings
import time
import datetime
# --- NEW/MODIFIED IMPORTS ---
from threading import Thread, Event
import librosa
import soundfile as sf
import io
import pygame
try:
    from utils.model.model import load_model as load_blendshape_model
    from utils.config import config as neurosync_config
    from utils.generate_face_shapes import generate_facial_data_from_bytes
    from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
    from livelink.send_to_unreal import pre_encode_facial_data, send_pre_encoded_data_to_unreal
    from livelink.animations.default_animation import default_animation_loop, stop_default_animation
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import Neuro-Sync/Livelink utility files. Have you copied the 'utils' and 'livelink' folders? Error: {e}")
    sys.exit(1)

def DEBUG_LOG(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[DEBUG {timestamp}] {message}")

def global_exception_hook(exctype, value, tb):
    DEBUG_LOG(f"Global Unhandled Exception Type: {exctype}")
    DEBUG_LOG(f"Global Unhandled Exception Value: {value}")
    import traceback
    DEBUG_LOG("Global Unhandled Exception Traceback:\n" + "".join(traceback.format_tb(tb)))
    sys.__excepthook__(exctype, value, tb)
    sys.exit(1)

sys.excepthook = global_exception_hook

DEBUG_LOG("robot_module.py: Script starting.")
project_root_robot_module = os.path.dirname(os.path.abspath(__file__))
src_path_robot_module = os.path.join(project_root_robot_module, 'src')

if os.path.exists(src_path_robot_module) and os.path.isdir(src_path_robot_module):
    if src_path_robot_module not in sys.path:
        sys.path.insert(0, src_path_robot_module)
else:
    alt_src_path_if_in_subdir = os.path.abspath(os.path.join(project_root_robot_module, '..', 'src'))
    if os.path.exists(alt_src_path_if_in_subdir) and os.path.isdir(alt_src_path_if_in_subdir) and 'f5_tts' in os.listdir(alt_src_path_if_in_subdir):
        if alt_src_path_if_in_subdir not in sys.path:
            sys.path.insert(0, alt_src_path_if_in_subdir)

try:
    import torch, torchaudio, google.generativeai as genai
    from transformers import pipeline
    from vocos import Vocos
    from f5_tts.infer.utils_infer import (
        infer_batch_process, load_model, load_vocoder, preprocess_ref_audio_text,
        remove_silence_for_generated_wav, chunk_text
    )
    from f5_tts.model import DiT
    DEBUG_LOG("robot_module.py: Core AI/TTS libraries imported successfully.")
except ImportError as e:
    DEBUG_LOG(f"robot_module.py: Fatal Error importing libraries: {e}")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

f5tts_model = None
vocoder = None
ref_audio_data_loaded = None
ref_text_processed = None
gemini_model = None
gemini_model_name_used = None
initialization_error_msg_global = None
temp_dir = None
ASR_PIPE = None
blendshape_model_global = None

DEFAULT_NFE_STEP = 32
DEFAULT_CFG_STRENGTH = 2.0
DEFAULT_TTS_MAX_CHARS = 250
DEFAULT_TTS_SPEED = 0.6
SPECIFIC_GEMINI_MODEL_NAME = "models/gemini-2.5-flash"
SYSTEM_PROMPT = "ตอบคำถามสั้นๆ ไม่เกิน 15 คำ"
WAKE_WORD_LANG = "english"
COMMAND_LANG = "thai"

def initialize_all_models(model_path, vocab_path, ref_audio_path, api_key):
    global f5tts_model, vocoder, ref_audio_data_loaded, ref_text_processed, gemini_model, gemini_model_name_used
    global initialization_error_msg_global, temp_dir, ASR_PIPE, device, blendshape_model_global

    DEBUG_LOG("robot_module.py: Starting full model initialization...")
    try:
        if not temp_dir:
            temp_dir = tempfile.mkdtemp(prefix="robot_module_")

        if not api_key: raise ValueError("Google API key not provided.")
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(SPECIFIC_GEMINI_MODEL_NAME)
        
        whisper_model_name = "openai/whisper-large-v3"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ASR_PIPE = pipeline("automatic-speech-recognition", model=whisper_model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device=device)
        
        vocoder = load_vocoder(vocoder_name="vocos", device=device)
        
        if not os.path.exists(model_path): raise FileNotFoundError(f"TTS model not found: {model_path}")
        if not os.path.exists(vocab_path): raise FileNotFoundError(f"TTS vocab not found: {vocab_path}")
        
        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        f5tts_model = load_model(DiT, F5TTS_model_cfg, model_path, vocab_file=vocab_path, use_ema=True, device=device)

        error_ref_audio = update_reference_audio_globals(ref_audio_path)
        if error_ref_audio: raise RuntimeError(f"Failed to process initial ref audio: {error_ref_audio}")
        
        DEBUG_LOG("robot_module.py: Loading Neuro-Sync blendshape model...")
        neurosync_model_path = "utils/model/model.pth"
        if not os.path.exists(neurosync_model_path):
            raise FileNotFoundError(f"Neuro-Sync model not found at: {neurosync_model_path}.")
        blendshape_model_global = load_blendshape_model(neurosync_model_path, neurosync_config, device)
        DEBUG_LOG("robot_module.py: Neuro-Sync blendshape model loaded successfully.")

        DEBUG_LOG("robot_module.py: --- ALL MODELS INITIALIZED SUCCESSFULLY ---")
    except Exception as e:
        initialization_error_msg_global = str(e)
        DEBUG_LOG(f"robot_module.py: --- MODEL INITIALIZATION FAILED: {traceback.format_exc()} ---")

def update_reference_audio_globals(audio_path, text_hint=""):
    global ref_audio_data_loaded, ref_text_processed, device, ASR_PIPE
    try:
        if not os.path.exists(audio_path):
            if not os.path.isabs(audio_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(script_dir, audio_path)
                if os.path.exists(resolved_path): audio_path = resolved_path
                else: raise FileNotFoundError(f"Reference audio file not found: '{audio_path}'")
            else: raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
        temp_processed_ref_audio_path, new_ref_text = preprocess_ref_audio_text(ref_audio_orig=audio_path, ref_text=text_hint, device=device, show_info=DEBUG_LOG)
        ref_audio_tensor, ref_audio_sr = torchaudio.load(temp_processed_ref_audio_path)
        if ref_audio_tensor.shape[0] > 1: ref_audio_tensor = torch.mean(ref_audio_tensor, dim=0, keepdim=True)
        if ref_audio_sr != 24000:
            resampler = torchaudio.transforms.Resample(ref_audio_sr, 24000)
            ref_audio_tensor = resampler(ref_audio_tensor)
        ref_audio_data_loaded = (ref_audio_tensor.to(device), 24000)
        ref_text_processed = new_ref_text
        os.remove(temp_processed_ref_audio_path)
        return None
    except Exception as e:
        return str(e)

@atexit.register
def cleanup_temp_dir_on_exit():
    global temp_dir
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def perform_stt(audio_path, purpose="command"):
    global ASR_PIPE, WAKE_WORD_LANG, COMMAND_LANG
    if ASR_PIPE is None: raise RuntimeError("ASR not initialized.")
    lang_to_use = COMMAND_LANG if purpose == "command" else WAKE_WORD_LANG
    result = ASR_PIPE(audio_path, generate_kwargs={"language": lang_to_use, "task": "transcribe"})
    return result["text"].strip() if result and "text" in result else ""

def get_gemini_response(text_input, history=None):
    global gemini_model, SYSTEM_PROMPT
    if gemini_model is None: raise RuntimeError("Gemini not initialized.")
    _history = history or []
    gemini_history_formatted = []
    if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
        gemini_history_formatted.append({"role": "user", "parts": [SYSTEM_PROMPT.strip()]})
        gemini_history_formatted.append({"role": "model", "parts": ["Understood."]})
    for user_msg, ai_msg in _history:
        gemini_history_formatted.append({"role": "user", "parts": [user_msg or ""]})
        gemini_history_formatted.append({"role": "model", "parts": [ai_msg or ""]})
    chat_session = gemini_model.start_chat(history=gemini_history_formatted)
    response = chat_session.send_message(text_input)
    return response.text.strip()

def perform_tts(text_to_speak):
    global f5tts_model, vocoder, ref_audio_data_loaded, ref_text_processed, temp_dir, device
    if not all([f5tts_model, vocoder, ref_audio_data_loaded, ref_text_processed]):
        raise RuntimeError("TTS components not initialized.")
    gen_text_batches = chunk_text(text_to_speak, max_chars=DEFAULT_TTS_MAX_CHARS)
    if not gen_text_batches and text_to_speak.strip():
        gen_text_batches = [text_to_speak.strip()]
    if not gen_text_batches:
        raise ValueError("No text provided for TTS.")

    tts_generator = infer_batch_process(ref_audio=ref_audio_data_loaded, ref_text=ref_text_processed, gen_text_batches=gen_text_batches, model_obj=f5tts_model, vocoder=vocoder, speed=DEFAULT_TTS_SPEED, nfe_step=DEFAULT_NFE_STEP, cfg_strength=DEFAULT_CFG_STRENGTH, device=device)
    final_wave, final_sample_rate, _ = next(tts_generator)
    if final_wave is None: raise RuntimeError("TTS generation returned no audio.")
    
    audio_output_filename = f"tts_output_{random.randint(0, 10000000)}.wav"
    audio_output_path = os.path.join(temp_dir, audio_output_filename)
    sf.write(audio_output_path, final_wave, final_sample_rate)
    try:
        remove_silence_for_generated_wav(audio_output_path)
    except Exception: pass
    return audio_output_path

def perform_blendshape_generation(audio_bytes):
    global blendshape_model_global, device
    if blendshape_model_global is None:
        raise RuntimeError("Neuro-Sync blendshape model is not initialized.")
    
    DEBUG_LOG("robot_module.py: Resampling audio for Neuro-Sync model...")
    try:
        original_audio, original_sr = sf.read(io.BytesIO(audio_bytes))
        target_sr = 88200

        if original_audio.ndim > 1:
            original_audio = np.mean(original_audio, axis=1)

        if original_sr != target_sr:
            resampled_audio = librosa.resample(y=original_audio, orig_sr=original_sr, target_sr=target_sr)
            DEBUG_LOG(f"robot_module.py: Resampled audio from {original_sr} Hz to {target_sr} Hz.")
        else:
            resampled_audio = original_audio

        resampled_audio_bytes_io = io.BytesIO()
        sf.write(resampled_audio_bytes_io, resampled_audio, target_sr, format='WAV', subtype='PCM_16')
        resampled_audio_bytes = resampled_audio_bytes_io.getvalue()
    except Exception as e:
        DEBUG_LOG(f"robot_module.py: Audio resampling failed: {e}. Falling back to original audio bytes.")
        resampled_audio_bytes = audio_bytes

    DEBUG_LOG("robot_module.py: Generating blendshapes from resampled audio bytes...")
    facial_data = generate_facial_data_from_bytes(
        resampled_audio_bytes,
        blendshape_model_global,
        device,
        neurosync_config
    )
    
    # CRITICAL FIX: Ensure facial data is not being zeroed inappropriately
    if isinstance(facial_data, np.ndarray):
        DEBUG_LOG(f"robot_module.py: Generated {facial_data.shape[0]} frames with {facial_data.shape[1]} blendshapes")
        # Verify critical blendshapes are not all zeros
        mouth_indices = list(range(17, 41))  # Mouth blendshapes
        eyebrow_indices = [41, 42, 43, 44, 45]  # Eyebrow blendshapes
        
        mouth_data = facial_data[:, mouth_indices]
        eyebrow_data = facial_data[:, eyebrow_indices]
        
        DEBUG_LOG(f"robot_module.py: Mouth data range: {mouth_data.min():.4f} to {mouth_data.max():.4f}")
        DEBUG_LOG(f"robot_module.py: Eyebrow data range: {eyebrow_data.min():.4f} to {eyebrow_data.max():.4f}")
        
        return facial_data.tolist()
    else:
        return facial_data

    DEBUG_LOG("robot_module.py: Generating blendshapes from resampled audio bytes...")
    facial_data = generate_facial_data_from_bytes(
        resampled_audio_bytes,
        blendshape_model_global,
        device,
        neurosync_config
    )
    return facial_data.tolist() if isinstance(facial_data, np.ndarray) else facial_data
def debug_facial_data(facial_data, label=""):
    """Debug function to analyze facial data"""
    if isinstance(facial_data, list) and len(facial_data) > 0:
        sample_frame = facial_data[0]
        DEBUG_LOG(f"DEBUG {label}: Frame 0 sample values:")
        DEBUG_LOG(f"  Mouth Close (18): {sample_frame[18]:.4f}")
        DEBUG_LOG(f"  Jaw Open (17): {sample_frame[17]:.4f}")
        DEBUG_LOG(f"  Brow Down Left (41): {sample_frame[41]:.4f}")
        DEBUG_LOG(f"  Brow Down Right (42): {sample_frame[42]:.4f}")
        
        # Check if data looks reasonable
        mouth_active = any(abs(frame[i]) > 0.01 for frame in facial_data for i in range(17, 41))
        brow_active = any(abs(frame[i]) > 0.01 for frame in facial_data for i in range(41, 46))
        DEBUG_LOG(f"  Mouth activity detected: {mouth_active}")
        DEBUG_LOG(f"  Eyebrow activity detected: {brow_active}")
def play_audio_from_memory(audio_bytes, start_event):
    try:
        pygame.mixer.init()
        audio_file = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_file)
        start_event.wait()
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        DEBUG_LOG(f"LIVELINK: Error playing audio: {e}")

def play_and_stream_animation(audio_bytes, facial_data):
    DEBUG_LOG("LIVELINK: Starting audio and animation stream...")
    try:
        # CRITICAL FIX: Pause default animation to prevent conflicts
        pause_default_animation()
        DEBUG_LOG("LIVELINK: Default animation paused")
        
        py_face = initialize_py_face()
        socket_connection = create_socket_connection()
        encoded_facial_data = pre_encode_facial_data(facial_data, py_face)
        start_event = Event()
        audio_thread = Thread(target=play_audio_from_memory, args=(audio_bytes, start_event))
        data_thread = Thread(target=send_pre_encoded_data_to_unreal, args=(encoded_facial_data, start_event, 60, socket_connection))
        audio_thread.start()
        data_thread.start()
        start_event.set()
        audio_thread.join()
        data_thread.join()
        socket_connection.close()
        DEBUG_LOG("LIVELINK: Audio and animation stream finished.")
        
    except Exception as e:
        DEBUG_LOG(f"LIVELINK: Error during stream: {e}")
    finally:
        # CRITICAL FIX: Resume default animation after facial animation completes
        resume_default_animation()
        DEBUG_LOG("LIVELINK: Default animation resumed")

def start_default_livelink_animation_stream():
    DEBUG_LOG("LIVELINK: Starting default idle animation stream...")
    try:
        py_face = initialize_py_face()
        default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
        default_animation_thread.daemon = True
        default_animation_thread.start()
        DEBUG_LOG("LIVELINK: Default idle animation thread started.")
    except Exception as e:
        DEBUG_LOG(f"LIVELINK: Failed to start default animation stream: {e}")