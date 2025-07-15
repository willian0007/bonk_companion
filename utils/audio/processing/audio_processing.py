# audio_processing.py - FIXED VERSION

import numpy as np
import torch
from torch.cuda.amp import autocast

def decode_audio_chunk(audio_chunk, model, device, config):
    use_half_precision = config.get("use_half_precision", True)
    dtype = torch.float16 if use_half_precision else torch.float32
    src_tensor = torch.tensor(audio_chunk, dtype=dtype).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_half_precision:
            with autocast(dtype=torch.float16):
                encoder_outputs = model.encoder(src_tensor)
                output_sequence = model.decoder(encoder_outputs)
        else:
            encoder_outputs = model.encoder(src_tensor)
            output_sequence = model.decoder(encoder_outputs)

        decoded_outputs = output_sequence.squeeze(0).cpu().numpy()
    return decoded_outputs


def concatenate_outputs(all_decoded_outputs, num_frames):
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)
    final_decoded_outputs = final_decoded_outputs[:num_frames]
    return final_decoded_outputs

def ensure_2d(final_decoded_outputs):
    if final_decoded_outputs.ndim == 3:
        final_decoded_outputs = final_decoded_outputs.reshape(-1, final_decoded_outputs.shape[-1])
    return final_decoded_outputs

def pad_audio_chunk(audio_chunk, frame_length, num_features, pad_mode='replicate'):
    if audio_chunk.shape[0] < frame_length:
        pad_length = frame_length - audio_chunk.shape[0]
        
        if pad_mode == 'reflect':
            padding = np.pad(
                audio_chunk,
                pad_width=((0, pad_length), (0, 0)),
                mode='reflect'
            )
            audio_chunk = np.vstack((audio_chunk, padding[-pad_length:, :num_features]))
        
        elif pad_mode == 'replicate':
            last_frame = audio_chunk[-1:]  
            replication = np.tile(last_frame, (pad_length, 1)) 
            audio_chunk = np.vstack((audio_chunk, replication))
        
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}. Choose 'reflect' or 'replicate'.")
    
    return audio_chunk


def blend_chunks(chunk1, chunk2, overlap):
    actual_overlap = min(overlap, len(chunk1), len(chunk2))
    if actual_overlap == 0:
        return np.vstack((chunk1, chunk2))
    
    blended_chunk = np.copy(chunk1)
    for i in range(actual_overlap):
        alpha = i / actual_overlap 
        blended_chunk[-actual_overlap + i] = (1 - alpha) * chunk1[-actual_overlap + i] + alpha * chunk2[i]
        
    return np.vstack((blended_chunk, chunk2[actual_overlap:]))

def process_audio_features(audio_features, model, device, config):
    frame_length = config['frame_size'] 
    overlap = config.get('overlap', 32) 
    num_features = audio_features.shape[1]
    num_frames = audio_features.shape[0]
    all_decoded_outputs = []
    model.eval()
    
    start_idx = 0
    while start_idx < num_frames:
        end_idx = min(start_idx + frame_length, num_frames)
        audio_chunk = audio_features[start_idx:end_idx]
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)
        decoded_outputs = decode_audio_chunk(audio_chunk, model, device, config)
        decoded_outputs = decoded_outputs[:end_idx - start_idx]
        
        if all_decoded_outputs:
            last_chunk = all_decoded_outputs.pop()
            blended_chunk = blend_chunks(last_chunk, decoded_outputs, overlap)
            all_decoded_outputs.append(blended_chunk)
        else:
            all_decoded_outputs.append(decoded_outputs)
            
        start_idx += frame_length - overlap

    current_length = sum(len(chunk) for chunk in all_decoded_outputs)
    if current_length < num_frames:
        remaining_frames = num_frames - current_length
        final_chunk_start = num_frames - remaining_frames
        audio_chunk = audio_features[final_chunk_start:num_frames]
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)
        decoded_outputs = decode_audio_chunk(audio_chunk, model, device, config)
        all_decoded_outputs.append(decoded_outputs[:remaining_frames])

    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)[:num_frames]
    final_decoded_outputs = ensure_2d(final_decoded_outputs)
    
    final_decoded_outputs[:, :61] /= 100  

    ease_duration_frames = min(int(0.1 * 60), final_decoded_outputs.shape[0])
    easing_factors = np.linspace(0, 1, ease_duration_frames)[:, None]
    final_decoded_outputs[:ease_duration_frames] *= easing_factors
    
    # CRITICAL FIX: Use selective zeroing instead of aggressive zeroing
    final_decoded_outputs = selective_zero_columns(final_decoded_outputs)

    return final_decoded_outputs


def selective_zero_columns(data):
    """
    FIXED VERSION: Only zero specific eye movement columns that should be controlled 
    by Unreal Engine, but preserve mouth and eyebrow data that should come from audio.
    """
    # Only zero eye look directions and head rotation - preserve mouth and eyebrows
    columns_to_zero = [
        # Eye look directions (preserve blinks and squints)
        1, 2, 3, 4,    # Left eye look directions
        8, 9, 10, 11,  # Right eye look directions
        # Head rotation (let UE handle this)
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60
    ]
    
    modified_data = np.copy(data) 
    modified_data[:, columns_to_zero] = 0
    return modified_data


# DEPRECATED - this was too aggressive
def zero_columns(data):
    """
    ISSUE: This function was zeroing too many important facial features including 
    eyebrow movements (0, 7) and mouth features that should be preserved.
    """
    # OLD CODE THAT CAUSED ISSUES:
    # columns_to_zero = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    pass  # Disabled