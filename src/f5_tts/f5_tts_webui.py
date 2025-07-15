import random
import sys
from importlib.resources import files
import gradio as gr 
import tempfile
import torchaudio
import soundfile as sf
from cached_path import cached_path
import argparse
import os

from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe,
    target_sample_rate,
)
from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything
import torch
from f5_tts.cleantext.number_tha import replace_numbers_with_thai
from f5_tts.cleantext.th_repeat import process_thai_repeat

#ถ้าอยากใช้โมเดลที่อัพเดทใหม หรือโมเดลภาษาอื่น สามารถแก้ไขโค้ด Model และ Vocab เช่น default_model_base = "hf://VIZINTZOR/F5-TTS-THAI/model_350000.pt"
default_model_base = r"D:\f5tts\F5-TTS-THAI\ckpts\model_600000.pt"
# fp16_model_base = "hf://VIZINTZOR/F5-TTS-THAI/model_600000_FP16.pt" # Removed FP16 model
vocab_base = r"D:\f5tts\F5-TTS-THAI\vocab\vocab.txt"

model_choices = ["Default", "Custom"] # Updated model choices

global f5tts_model
f5tts_model = None

def load_f5tts(ckpt_path, vocab_path_to_use=vocab_base):
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    
    if not os.path.exists(vocab_path_to_use):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path_to_use}. Ensure it is correctly specified and accessible.")

    model = load_model(DiT, F5TTS_model_cfg, ckpt_path, vocab_file=vocab_path_to_use, use_ema=True)
    print(f"Loaded model from {ckpt_path} using vocab {vocab_path_to_use}")
    return model

vocoder = load_vocoder()

# Initial model loading
try:
    print(f"Attempting to load initial model: {default_model_base} with vocab: {vocab_base}")
    if not os.path.exists(default_model_base):
        print(f"Error: Default model file not found at {default_model_base}")
        # Potentially exit or raise a more user-friendly error if used in a context where Gradio isn't up yet
        # For now, f5tts_model will remain None, and infer_tts will try to load it again.
        # This might be problematic if the path is genuinely wrong and never fixable.
    elif not os.path.exists(vocab_base):
        print(f"Error: Default vocab file not found at {vocab_base}")
    else:
        f5tts_model = load_f5tts(default_model_base, vocab_base)
        initial_model_status = f"เริ่มต้นด้วย: Default ({os.path.basename(default_model_base)})"
except Exception as e:
    print(f"Error loading initial model: {e}")
    f5tts_model = None # Ensure it's None if loading failed
    initial_model_status = "Error: ไม่สามารถโหลดโมเดลเริ่มต้นได้"


def update_custom_model(selected_model):
    return gr.update(visible=selected_model == "Custom")
    
def load_custom_model(model_choice, model_custom_path_input):
    torch.cuda.empty_cache()
    global f5tts_model
    
    actual_model_path_to_load = ""
    display_name = ""

    if model_choice == "Default":
        actual_model_path_to_load = default_model_base
        display_name = f"Default ({os.path.basename(default_model_base)})"
    elif model_choice == "Custom":
        if not model_custom_path_input or not model_custom_path_input.strip():
            gr.Warning("Custom model path is empty.")
            return "Error: Custom model path is empty."
        actual_model_path_to_load = model_custom_path_input
        display_name = f"Custom ({model_custom_path_input})" # Display full path for custom
    else:
        gr.Error("Invalid model choice selected.")
        return "Error: Invalid model choice."

    try:
        print(f"Attempting to load: {actual_model_path_to_load} with vocab: {vocab_base}")
        if actual_model_path_to_load.startswith("hf://"):
            # For Hugging Face paths, vocab_base (local) will be used as specified
            cached_model_path = str(cached_path(actual_model_path_to_load))
            f5tts_model = load_f5tts(cached_model_path, vocab_base)
        else: # Assumed local path
            if not os.path.exists(actual_model_path_to_load):
                gr.Error(f"Model file not found at {actual_model_path_to_load}")
                return f"Error: Model file not found at {actual_model_path_to_load}"
            f5tts_model = load_f5tts(actual_model_path_to_load, vocab_base)
        
        gr.Info(f"Successfully loaded model: {display_name}")
        return f"Loaded Model: {display_name}"
    except Exception as e:
        print(f"Failed to load model {display_name}: {e}")
        gr.Error(f"Failed to load model {display_name}: {e}")
        return f"Error loading model: {display_name}. Check console for details."
    
def infer_tts(
    ref_audio_orig,
    ref_text,
    gen_text,
    remove_silence=True,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    cfg_strength=2,
    max_chars=250,
    seed=-1
):
    global f5tts_model
    if f5tts_model is None:
        gr.Warning("Model not loaded. Attempting to load default model.")
        print("f5tts_model is None in infer_tts. Attempting to load default model.")
        try:
            if not os.path.exists(default_model_base) or not os.path.exists(vocab_base):
                 gr.Error(f"Default model or vocab file not found. Paths: {default_model_base}, {vocab_base}")
                 return gr.update(), gr.update(), ref_text, seed # Return early
            f5tts_model = load_f5tts(default_model_base, vocab_base)
            gr.Info("Default model loaded.")
        except Exception as e:
            print(f"Error loading default model in infer_tts: {e}")
            gr.Error(f"Failed to load default model: {e}")
            return gr.update(), gr.update(), ref_text, seed # Return early

    if seed == -1:
        seed = random.randint(0, sys.maxsize)
    seed_everything(seed)
    output_seed = seed

    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text, output_seed

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text, output_seed
    
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text)
    cross_fade_duration = float(cross_fade_duration)
    
    gen_text_cleaned = process_thai_repeat(replace_numbers_with_thai(gen_text))
    
    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text_cleaned,
        f5tts_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        progress=gr.Progress(),
        cfg_strength=cfg_strength,
        target_rms=0.1,
        sway_sampling_coef=-1,
        set_max_chars=max_chars
    )

    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)
    
    print("seed:", output_seed)
    return (final_sample_rate, final_wave), spectrogram_path, ref_text, output_seed 

def create_gradio_interface():
    global initial_model_status # Access the status set during initial load
    with gr.Blocks(title="F5-TTS ไทย",theme=gr.themes.Ocean()) as demo:
        gr.Markdown("# F5-TTS ภาษาไทย")
        gr.Markdown("สร้างคำพูดจากข้อความ ด้วย Zero-shot TTS หรือ เสียงต้นฉบับ ภาษาไทย.")

        with gr.Row():
            model_select = gr.Radio(
                label="โมเดล",
                choices=model_choices,
                value="Default", # Default selection
                interactive=True,
                # info="ถ้าใช้ FP16 จะใช้ทรัพยากรเครื่องหรือ VRAM น้อยกว่า" # Removed info as FP16 option is removed
            )
            model_custom = gr.Textbox(label="ตำแหน่งโมเดลแบบกำหนดเอง",value="hf://VIZINTZOR/F5-TTS-THAI/model_500000.pt", visible=False, interactive=True, placeholder="เช่น D:\\models\\my_model.pt หรือ hf://org/repo/model.pt")
            load_custom_btn = gr.Button("โหลด",variant="primary")
            
        with gr.Row():
            with gr.Column():
                ref_text = gr.Textbox(label="ข้อความต้นฉบับ", lines=1, info="แนะนำให้ใช้เสียงที่มีความยาวไม่เกิน 5-10 วินาที")
                ref_audio = gr.Audio(label="เสียงต้นฉบับ", type="filepath")
                gen_text = gr.Textbox(label="ข้อความที่จะสร้าง", lines=4)
                generate_btn = gr.Button("สร้าง",variant="primary")

                with gr.Accordion(label="ตั้งค่า"):
                    remove_silence = gr.Checkbox(label="Remove Silence", value=True)
                    speed = gr.Slider(label="ความเร็ว", value=1, minimum=0.3, maximum=2, step=0.1)
                    cross_fade_duration = gr.Slider(label="Cross Fade Duration", value="0.15", minimum=0, maximum=1, step=0.05)
                    nfe_step = gr.Slider(label="NFE Step", value=32, minimum=16, maximum=64, step=8, info="ยิ่งค่ามากยิ่งมีคุณภาพสูง แต่อาจจะช้าลง")
                    cfg_strength = gr.Slider(label="CFG Strength", value=2, minimum=1, maximum=4, step=0.5)
                    max_chars = gr.Number(label="ตัวอักษรสูงสุดต่อส่วน", minimum=50, maximum=1000, value=250,
                                          info="จำนวนตัวอักษรสูงสุดที่ใช้ในการแบ่งส่วน สำหรับข้อความยาวๆ")
                    seed = gr.Number(label="Seed", value=-1, precision=0, info="-1 = สุ่ม Seed")
                    
            with gr.Column():
                output_audio = gr.Audio(label="เสียงที่สร้าง", type="filepath")
                seed_output = gr.Textbox(label="Seed", interactive=False)
                model_status_textbox = gr.Textbox(label="สถานะโมเดล", value=initial_model_status, interactive=False) # Use initial_model_status
                
        gr.Examples(
            examples=[
                [
                    "./src/f5_tts/infer/examples/thai_examples/ref_gen_1.wav",
                    "ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น.",
                    "พรุ่งนี้มีประชุมสำคัญ อย่าลืมเตรียมเอกสารให้เรียบร้อย"
                ],
                [
                    "./src/f5_tts/infer/examples/thai_examples/ref_gen_2.wav",
                    "ฉันเดินทางไปเที่ยวที่จังหวัดเชียงใหม่ในช่วงฤดูหนาวเพื่อสัมผัสอากาศเย็นสบาย.",
                    "ฉันชอบฟังเพลงขณะขับรถ เพราะช่วยให้รู้สึกผ่อนคลาย"
                ],
                [
                    "./src/f5_tts/infer/examples/thai_examples/ref_gen_3.wav",
                    "กู้ดอาฟเต้อนูนไนท์ทูมีทยู.",
                    "วันนี้อากาศดีมาก เหมาะกับการไปเดินเล่นที่สวนสาธารณะ"
                ]
            ],
            inputs=[ref_audio, ref_text, gen_text],
            fn=infer_tts,
            cache_examples=False, # Consider server capacity if enabling caching with local paths
            label="ตัวอย่าง"
        )

        load_custom_btn.click(
            fn=load_custom_model,
            inputs=[
                model_select,
                model_custom
                ],
            outputs=model_status_textbox # Wire output to the textbox
        )
        
        model_select.change(
            fn=update_custom_model,
            inputs=model_select,
            outputs=model_custom
        )
        
        generate_btn.click(
            fn=infer_tts,
            inputs=[
                ref_audio,
                ref_text,
                gen_text,
                remove_silence,
                cross_fade_duration,
                nfe_step,
                speed,
                cfg_strength,
                max_chars,
                seed
            ],
            outputs=[
                output_audio,
                gr.Image(label="Spectrogram"),
                ref_text,
                seed_output
            ]
        )

    return demo

def main():
    parser = argparse.ArgumentParser(description="Share Link")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_gradio_interface()
    demo.launch(inbrowser=True, share=args.share)

if __name__ == "__main__":
    main()