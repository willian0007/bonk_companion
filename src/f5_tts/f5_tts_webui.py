import random
import sys
from importlib.resources import files
import gradio as gr 
import tempfile
import torchaudio

import soundfile as sf
#import tqdm
from cached_path import cached_path

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

model_base = "hf://VIZINTZOR/F5-TTS-THAI/model_150000.pt"
vocab_base = "./vocab/vocab.txt"

#Load Model
def load_f5tts(ckpt_path=str(cached_path(model_base)), vocab_path=vocab_base):
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = load_model(DiT, F5TTS_model_cfg, ckpt_path, vocab_file=vocab_path, use_ema=False)
    print("Model Loaded")
    return model

f5tts_model = load_f5tts()

vocoder = load_vocoder()

def infer_tts(
    ref_audio_orig,
    ref_text,
    gen_text,
    remove_silence=True,
    cross_fade_duration=0.15,  # Default value
    nfe_step=32,
    speed=1,
    cfg_strength=2,
    show_info=gr.Info,
    seed=-1,
):
    seed = -1
    if seed == -1:
        seed = random.randint(0, sys.maxsize)
    seed_everything(seed)
    output_seed = seed

    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text
    
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    # Convert cross_fade_duration to float
    cross_fade_duration = float(cross_fade_duration)

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        f5tts_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
        cfg_strength=cfg_strength
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)
    
    print("seed:", output_seed)
    return (final_sample_rate, final_wave), spectrogram_path, ref_text , output_seed 

def create_gradio_interface():
    with gr.Blocks(title="F5-TTS") as demo:
        gr.Markdown("# F5-TTS Thai Language Support")
        gr.Markdown("Generate speech from text using a reference audio sample with improve thai language.")
        
        with gr.Row():
            with gr.Column():
                # Input components
                ref_text = gr.Textbox(label="Reference Text", lines=1)
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                gen_text = gr.Textbox(label="Text to Generate", lines=4)
                generate_btn = gr.Button("Generate Speech")

                with gr.Accordion(label="Advance Settings"):
                    remove_silence = gr.Checkbox(label="Remove Silence",value=True)
                    speed = gr.Slider(label="Speed", value=1, minimum=0.1, maximum=2,step=0.1)
                    cross_fade_duration = gr.Slider(label="Cross Fade Duration", value="0.15",minimum=0, maximum=1,step=0.05)
                    nfe_step = gr.Slider(label="NFE Step", value=32 ,minimum=16, maximum=64, step=8)
                    cfg_strength = gr.Slider(label="CFG Strength", value=2,minimum=0, maximum=5,step=0.5)
                    
            with gr.Column():
                # Output components
                output_audio = gr.Audio(label="Generated Speech", type="filepath")
                seed = gr.Textbox(label="Seed")
        
        # Connect the interface
        generate_btn.click(
            fn=infer_tts,
            inputs=[
                ref_audio,    # ref_file
                ref_text,     # ref_text
                gen_text,       # gen_text
                remove_silence,     
                cross_fade_duration,
                nfe_step,
                speed,
                cfg_strength
            ],
            outputs=[output_audio, gr.Image(label="Spectrogram"), ref_text, seed]
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface ()
    demo.launch(inbrowser=True)
