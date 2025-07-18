# robot_module.py
import os
import sys

# This ensures that no matter how the script is run, it knows where the 'utils' and 'livelink' folders are.
project_root_dir = os.path.dirname(os.path.abspath(__file__))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import random
import tempfile
import shutil
import numpy as np
import atexit
import warnings
import time
import datetime
import base64
import traceback
import json
import requests
from threading import Thread, Event
from queue import Queue
import librosa
import soundfile as sf
import io
import pygame

# Using the standard PyYAML library
import yaml

try:
    from utils.model.model import load_model as load_blendshape_model
    from utils.config import config as neurosync_config
    from utils.generate_face_shapes import generate_facial_data_from_bytes
    from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
    from livelink.send_to_unreal import pre_encode_facial_data, send_pre_encoded_data_to_unreal
    from livelink.animations.default_animation import default_animation_loop, stop_default_animation
    from livelink.animations.default_animation import pause_default_animation, resume_default_animation
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
    from vocos.feature_extractors import MelSpectrogramFeatures
    from vocos.models import VocosBackbone
    from vocos.heads import ISTFTHead
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
initialization_error_msg_global = None
temp_dir = None
ASR_PIPE = None
blendshape_model_global = None
current_ref_audio_path = None

DEFAULT_NFE_STEP = 16
DEFAULT_CFG_STRENGTH = 1.5
DEFAULT_TTS_MAX_CHARS = 200
DEFAULT_TTS_SPEED = 0.8
SPECIFIC_GEMINI_MODEL_NAME = "models/gemini-2.5-flash-lite-preview-06-17"
SYSTEM_PROMPT = "เป็นเพื่อนผู้หญิงคุยแก้เหงา ชื่อ ฟ้า ไม่พิมพ์ อักขระ ไม่ใช้ตัว ๆ ให้พิมพ์ซ้ำแทน ไม่พิมพ์ภาษาอังกฤษ พิมพ์แค่คำทับศัพท์ ไม่ใช้เลขอารบิก เช่น 1 2 แต่ใช้คำแทน เช่น หนึ่ง สอง พิมพ์คำตอบให้มีคำ 50-100 คำ ไม่มากไม่น้อยไปกว่านั้น "
WAKE_WORD_LANG = "english"
COMMAND_LANG = "thai"

STREAMING_CHUNK_WORD_LIMIT = 5
# This global variable now controls the STT method used in real-time
USE_GEMINI_STT = False

def perform_stt_gemini(audio_path, purpose="command"):
    global gemini_model, WAKE_WORD_LANG, COMMAND_LANG
    if gemini_model is None: raise RuntimeError("Gemini not initialized.")
    try:
        DEBUG_LOG(f"Gemini STT: Processing {audio_path}")
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        language = COMMAND_LANG if purpose == "command" else WAKE_WORD_LANG
        prompt = "โปรดแปลงเสียงพูดนี้เป็นข้อความภาษาไทย ตอบเฉพาะข้อความที่ได้ยินเท่านั้น:" if language == "thai" else "Please transcribe this audio to text. Return only the transcribed text:"
        audio_part = {"mime_type": "audio/wav", "data": audio_data}
        response = gemini_model.generate_content([prompt, audio_part])
        return response.text.strip()
    except Exception as e:
        DEBUG_LOG(f"Gemini STT Error: {e}")
        raise e

def perform_stt_whisper_fallback(audio_path, purpose="command"):
    global ASR_PIPE, WAKE_WORD_LANG, COMMAND_LANG
    if ASR_PIPE is None: raise RuntimeError("ASR not initialized.")
    lang_to_use = COMMAND_LANG if purpose == "command" else WAKE_WORD_LANG
    result = ASR_PIPE(audio_path, generate_kwargs={"language": lang_to_use, "task": "transcribe"})
    return result["text"].strip() if result and "text" in result else ""

# This is the main, switchable STT function for real-time use
def perform_stt(audio_path, purpose="command"):
    if USE_GEMINI_STT:
        try:
            return perform_stt_gemini(audio_path, purpose)
        except Exception as e:
            DEBUG_LOG(f"Gemini STT failed, falling back to Whisper: {e}")
            return perform_stt_whisper_fallback(audio_path, purpose)
    else:
        return perform_stt_whisper_fallback(audio_path, purpose)

def initialize_all_models(model_path, vocab_path, ref_audio_path, api_key):
    global f5tts_model, vocoder, ref_audio_data_loaded, ref_text_processed, gemini_model
    global initialization_error_msg_global, temp_dir, ASR_PIPE, device, blendshape_model_global, current_ref_audio_path

    DEBUG_LOG("robot_module.py: Starting full model initialization...")
    try:
        if not temp_dir:
            temp_dir = tempfile.mkdtemp(prefix="robot_module_")
        if not api_key: raise ValueError("Google API key not provided.")
        
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(SPECIFIC_GEMINI_MODEL_NAME)
        DEBUG_LOG(f"Gemini model '{SPECIFIC_GEMINI_MODEL_NAME}' initialized successfully.")
        
        local_whisper_path = os.getenv("WHISPER_MODEL_PATH")
        if not local_whisper_path or not os.path.isdir(local_whisper_path):
            raise FileNotFoundError(f"Whisper model path not found. Check WHISPER_MODEL_PATH in runserver.bat. Path: '{local_whisper_path}'")
        DEBUG_LOG(f"Initializing Whisper STT from local path: {local_whisper_path}")
        model_kwargs = {"local_files_only": True}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ASR_PIPE = pipeline("automatic-speech-recognition", model=local_whisper_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device=device, model_kwargs=model_kwargs)
        DEBUG_LOG("Whisper initialized successfully from local files.")
        
        local_vocoder_path = os.getenv("VOCODER_MODEL_PATH")
        if not local_vocoder_path or not os.path.isdir(local_vocoder_path):
            raise FileNotFoundError(f"Vocoder model path not found. Check VOCODER_MODEL_PATH in runserver.bat. Path: '{local_vocoder_path}'")
        DEBUG_LOG(f"Initializing Vocoder MANUALLY from local path: {local_vocoder_path}")
        config_path = os.path.join(local_vocoder_path, "config.yaml")
        model_path_vocoder = os.path.join(local_vocoder_path, "pytorch_model.bin")
        if not os.path.exists(config_path) or not os.path.exists(model_path_vocoder):
            raise FileNotFoundError("Vocoder config.yaml or pytorch_model.bin not found in the specified directory.")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        feature_extractor = MelSpectrogramFeatures(**config['feature_extractor']['init_args'])
        backbone = VocosBackbone(**config['backbone']['init_args'])
        head = ISTFTHead(**config['head']['init_args'])
        vocoder_instance = Vocos(feature_extractor=feature_extractor, backbone=backbone, head=head)
        state_dict = torch.load(model_path_vocoder, map_location="cpu")
        vocoder_instance.load_state_dict(state_dict)
        vocoder = vocoder_instance.to(device)
        DEBUG_LOG("Vocoder initialized successfully from local files.")
        
        if not os.path.exists(model_path): raise FileNotFoundError(f"TTS model (F5) not found: {model_path}")
        if not os.path.exists(vocab_path): raise FileNotFoundError(f"TTS vocab not found: {vocab_path}")
        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        f5tts_model = load_model(DiT, F5TTS_model_cfg, model_path, vocab_file=vocab_path, use_ema=True, device=device)

        # This now correctly uses the local Whisper model to transcribe the reference audio if needed
        error_ref_audio = update_reference_audio_globals(ref_audio_path)
        if error_ref_audio: raise RuntimeError(f"Failed to process initial ref audio: {error_ref_audio}")
        
        current_ref_audio_path = ref_audio_path
        
        DEBUG_LOG("robot_module.py: Loading Neuro-Sync blendshape model...")
        neurosync_model_path = os.path.join(project_root_dir, "utils", "model", "model.pth")
        if not os.path.exists(neurosync_model_path):
            raise FileNotFoundError(f"Neuro-Sync model not found at: {neurosync_model_path}.")
        blendshape_model_global = load_blendshape_model(neurosync_model_path, neurosync_config, device)
        DEBUG_LOG("robot_module.py: Neuro-Sync blendshape model loaded successfully.")

        DEBUG_LOG("robot_module.py: --- ALL MODELS INITIALIZED SUCCESSFULLY ---")
            
    except Exception as e:
        initialization_error_msg_global = str(e)
        DEBUG_LOG(f"robot_module.py: --- MODEL INITIALIZATION FAILED: {traceback.format_exc()} ---")

def get_current_settings_dict():
    # ... (function is unchanged)
    global current_ref_audio_path, ref_text_processed, DEFAULT_TTS_SPEED, DEFAULT_TTS_MAX_CHARS, DEFAULT_NFE_STEP, DEFAULT_CFG_STRENGTH, SPECIFIC_GEMINI_MODEL_NAME, SYSTEM_PROMPT
    return {"voice_settings": {"ref_audio_path": current_ref_audio_path or "Not set", "ref_text_hint": ref_text_processed or "Not set"},"tts_settings": {"speed": DEFAULT_TTS_SPEED, "max_chars": DEFAULT_TTS_MAX_CHARS, "nfe_step": DEFAULT_NFE_STEP, "cfg_strength": DEFAULT_CFG_STRENGTH},"gemini_settings": {"model_name": SPECIFIC_GEMINI_MODEL_NAME, "system_prompt": SYSTEM_PROMPT}}

def validate_audio_file_path(file_path):
    # ... (function is unchanged)
    if not file_path: return False, "No file path provided"
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)
    if not os.path.exists(file_path): return False, f"File does not exist: {file_path}"
    supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    if os.path.splitext(file_path)[1].lower() not in supported_formats: return False, f"Unsupported audio format."
    return True, file_path

# This version of the function intelligently uses the local Whisper model for transcription
def update_reference_audio_globals(audio_path, text_hint=""):
    global ref_audio_data_loaded, ref_text_processed, device, ASR_PIPE, current_ref_audio_path
    try:
        is_valid, result = validate_audio_file_path(audio_path)
        if not is_valid: return result
        audio_path = result

        if not text_hint:
            DEBUG_LOG("No reference text hint provided. Transcribing reference audio using LOCAL Whisper model...")
            if ASR_PIPE is None:
                return "Cannot transcribe reference audio because local Whisper model is not loaded yet."
            # This directly calls the function that uses our local model, bypassing Gemini logic
            text_hint = perform_stt_whisper_fallback(audio_path, purpose="command") # Use command lang for Thai ref audio
            DEBUG_LOG(f"Locally transcribed reference text: '{text_hint}'")

        temp_processed_ref_audio_path, new_ref_text = preprocess_ref_audio_text(ref_audio_orig=audio_path, ref_text=text_hint, device=device, show_info=DEBUG_LOG)
        ref_audio_tensor, ref_audio_sr = torchaudio.load(temp_processed_ref_audio_path)
        if ref_audio_tensor.shape[0] > 1: ref_audio_tensor = torch.mean(ref_audio_tensor, dim=0, keepdim=True)
        if ref_audio_sr != 24000:
            resampler = torchaudio.transforms.Resample(ref_audio_sr, 24000)
            ref_audio_tensor = resampler(ref_audio_tensor)
        ref_audio_data_loaded = (ref_audio_tensor.to(device), 24000)
        ref_text_processed = new_ref_text
        current_ref_audio_path = audio_path
        os.remove(temp_processed_ref_audio_path)
        return None
    except Exception as e:
        return str(e)

# ... (The rest of the file: reset_to_default_settings, get_gemini_response, perform_tts, etc. remains the same as your original)
def reset_to_default_settings():
    global DEFAULT_TTS_SPEED, DEFAULT_TTS_MAX_CHARS, DEFAULT_NFE_STEP, DEFAULT_CFG_STRENGTH, SPECIFIC_GEMINI_MODEL_NAME, SYSTEM_PROMPT, USE_GEMINI_STT
    DEFAULT_TTS_SPEED = 0.8
    DEFAULT_TTS_MAX_CHARS = 200
    DEFAULT_NFE_STEP = 16
    DEFAULT_CFG_STRENGTH = 1.5
    SPECIFIC_GEMINI_MODEL_NAME = "models/gemini-2.5-flash-lite-preview-06-17"
    SYSTEM_PROMPT = "เป็นเพื่อนผู้หญิงคุยแก้เหงา ชื่อ ฟ้า ไม่พิมพ์ อักขระ ไม่ใช้ตัว ๆ ให้พิมพ์ซ้ำแทน ไม่พิมพ์ภาษาอังกฤษ พิมพ์แค่คำทับศัพท์ ไม่ใช้เลขอารบิก เช่น 1 2 แต่ใช้คำแทน เช่น หนึ่ง สอง พิมพ์คำตอบให้มีคำ 50-100 คำ ไม่มากไม่น้อยไปกว่านั้น "
    USE_GEMINI_STT = False
    DEBUG_LOG("robot_module.py: Reset all settings to defaults")

@atexit.register
def cleanup_temp_dir_on_exit():
    global temp_dir
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

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
    if not gen_text_batches and text_to_speak.strip(): gen_text_batches = [text_to_speak.strip()]
    if not gen_text_batches: raise ValueError("No text provided for TTS.")
    tts_generator = infer_batch_process(ref_audio=ref_audio_data_loaded, ref_text=ref_text_processed, gen_text_batches=gen_text_batches, model_obj=f5tts_model, vocoder=vocoder, speed=DEFAULT_TTS_SPEED, nfe_step=DEFAULT_NFE_STEP, cfg_strength=DEFAULT_CFG_STRENGTH, device=device)
    final_wave, final_sample_rate, _ = next(tts_generator)
    if final_wave is None: raise RuntimeError("TTS generation returned no audio.")
    audio_output_filename = f"tts_output_{random.randint(0, 10000000)}.wav"
    audio_output_path = os.path.join(temp_dir, audio_output_filename)
    sf.write(audio_output_path, final_wave, final_sample_rate)
    try: remove_silence_for_generated_wav(audio_output_path)
    except Exception: pass
    return audio_output_path

def perform_blendshape_generation(audio_bytes):
    global blendshape_model_global, device
    if blendshape_model_global is None: raise RuntimeError("Neuro-Sync blendshape model is not initialized.")
    try:
        original_audio, original_sr = sf.read(io.BytesIO(audio_bytes))
        target_sr = 88200
        if original_audio.ndim > 1: original_audio = np.mean(original_audio, axis=1)
        if original_sr != target_sr:
            resampled_audio = librosa.resample(y=original_audio, orig_sr=original_sr, target_sr=target_sr)
        else:
            resampled_audio = original_audio
        resampled_audio_bytes_io = io.BytesIO()
        sf.write(resampled_audio_bytes_io, resampled_audio, target_sr, format='WAV', subtype='PCM_16')
        resampled_audio_bytes = resampled_audio_bytes_io.getvalue()
    except Exception as e:
        DEBUG_LOG(f"Audio resampling failed: {e}. Falling back to original audio bytes.")
        resampled_audio_bytes = audio_bytes
    facial_data = generate_facial_data_from_bytes(resampled_audio_bytes, blendshape_model_global, device, neurosync_config)
    return facial_data.tolist() if isinstance(facial_data, np.ndarray) else facial_data

def play_audio_from_memory(audio_bytes, start_event):
    try:
        pygame.mixer.init()
        audio_file = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_file)
        start_event.wait()
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): pygame.time.Clock().tick(10)
    except Exception as e:
        DEBUG_LOG(f"LIVELINK: Error playing audio: {e}")

def play_and_stream_animation(audio_bytes, facial_data):
    try:
        pause_default_animation()
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
    except Exception as e:
        DEBUG_LOG(f"LIVELINK: Error during stream: {e}")
    finally:
        resume_default_animation()

def start_default_livelink_animation_stream():
    try:
        py_face = initialize_py_face()
        default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
        default_animation_thread.daemon = True
        default_animation_thread.start()
        DEBUG_LOG("LIVELINK: Default idle animation thread started.")
    except Exception as e:
        DEBUG_LOG(f"LIVELINK: Failed to start default animation stream: {e}")

def producer(text_input, history, q):
    global gemini_model, SYSTEM_PROMPT, STREAMING_CHUNK_WORD_LIMIT
    try:
        _history = history or []
        gemini_history_formatted = []
        if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
            gemini_history_formatted.append({"role": "user", "parts": [SYSTEM_PROMPT.strip()]})
            gemini_history_formatted.append({"role": "model", "parts": ["Understood."]})
        for user_msg, ai_msg in _history:
            gemini_history_formatted.append({"role": "user", "parts": [user_msg or ""]})
            gemini_history_formatted.append({"role": "model", "parts": [ai_msg or ""]})
        chat_session = gemini_model.start_chat(history=gemini_history_formatted)
        response_stream = chat_session.send_message(text_input, stream=True)
        text_buffer = ""
        for chunk in response_stream:
            try: text_buffer += chunk.text
            except Exception: continue
            while True:
                words = text_buffer.split()
                if len(words) >= STREAMING_CHUNK_WORD_LIMIT:
                    words_to_process = words[:STREAMING_CHUNK_WORD_LIMIT]
                    sentence_to_process = " ".join(words_to_process)
                    text_buffer = " ".join(words[STREAMING_CHUNK_WORD_LIMIT:])
                    if sentence_to_process:
                        try:
                            audio_path = perform_tts(sentence_to_process)
                            with open(audio_path, "rb") as f: audio_bytes = f.read()
                            os.remove(audio_path)
                            blendshape_data = perform_blendshape_generation(audio_bytes)
                            q.put((sentence_to_process, audio_bytes, blendshape_data))
                        except Exception as e:
                            DEBUG_LOG(f"STREAM PRODUCER: Error processing sentence chunk: {e}")
                            text_buffer = "" 
                            break
                else:
                    break
        if text_buffer.strip():
            try:
                audio_path = perform_tts(text_buffer.strip())
                with open(audio_path, "rb") as f: audio_bytes = f.read()
                os.remove(audio_path)
                blendshape_data = perform_blendshape_generation(audio_bytes)
                q.put((text_buffer.strip(), audio_bytes, blendshape_data))
            except Exception as e:
                DEBUG_LOG(f"STREAM PRODUCER: Error processing final chunk: {e}")
    except Exception as e:
        DEBUG_LOG(f"STREAM PRODUCER: An unexpected error occurred: {e}")
    finally:
        q.put(("", None, None))
        q.put(None)

def consumer(q):
    api_url = "http://127.0.0.1:8000/update_current_subtitle"
    SUBTITLE_LEAD_TIME = 0.2
    while True:
        item = q.get()
        if item is None: break
        text_chunk, audio_bytes, facial_data = item
        try:
            response = requests.post(api_url, json={"text": text_chunk})
            if response.status_code != 200:
                DEBUG_LOG(f"STREAM CONSUMER: Subtitle API returned {response.status_code}")
        except Exception as e:
            DEBUG_LOG(f"STREAM CONSUMER: Failed to update subtitle via API: {e}")
        if text_chunk and audio_bytes: time.sleep(SUBTITLE_LEAD_TIME)
        if audio_bytes and facial_data: play_and_stream_animation(audio_bytes, facial_data)
        elif text_chunk == "": time.sleep(0.1)
        q.task_done()
    try: requests.post(api_url, json={"text": ""})
    except: pass
    q.task_done()

def generate_and_stream_response_threaded(text_input, history=None):
    data_queue = Queue()
    producer_thread = Thread(target=producer, args=(text_input, history, data_queue))
    consumer_thread = Thread(target=consumer, args=(data_queue,))
    producer_thread.start()
    consumer_thread.start()
