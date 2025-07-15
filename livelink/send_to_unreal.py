# send_to_unreal.py
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import time
from typing import List

from livelink.connect.livelink_init import create_socket_connection, FaceBlendShape
from livelink.animations.default_animation import default_animation_data, stop_default_animation
from livelink.animations.blending_anims import (
    generate_blend_frames,
    combine_frame_streams,
    FAST_BLENDSHAPES,
    default_animation_state,
)

def pre_encode_facial_data(facial_data: list, py_face, fps: int = 60, smooth: bool = False) -> list:
    """
    Encodes the full stream:
    1. Blend-IN (idle → capture)
    2. Main captured frames
    3. Blend-OUT (capture → idle **frame 0**)

    Returns
    -------
    encoded_data : list[bytes]
        Ready-to-send UDP packets.
    """
    encoded_data = []
    
    # IMPORTANT FIX: Only apply blink data, not other facial features
    apply_selective_default_data(facial_data, default_animation_data)

    total_duration = len(facial_data) / fps
    slow_duration  = 0.3 if total_duration < 1.0 else 0.5
    if total_duration < 0.5:
        slow_duration = 0.2

    fast_duration  = 0.1                    # jaw/mouth quick ease
    slow_blend_frames = int(slow_duration * fps)

    fast_blend_in = generate_blend_frames(
        facial_data, slow_blend_frames, default_animation_data, fps,
        FAST_BLENDSHAPES, mode='in', active_duration_sec=fast_duration
    )

    slow_blend_in = generate_blend_frames(
        facial_data, slow_blend_frames, default_animation_data, fps,
        set(range(51)) - FAST_BLENDSHAPES, mode='in'
    )

    blend_in_frames = combine_frame_streams(slow_blend_in, fast_blend_in, FAST_BLENDSHAPES)

    for frame in blend_in_frames:
        for i in range(51):
            py_face.set_blendshape(FaceBlendShape(i), frame[i])
        encoded_data.append(py_face.encode())

    main_start = slow_blend_frames
    main_end   = len(facial_data) - slow_blend_frames

    for frame_data in facial_data[main_start:main_end]:
        for i in range(51):
            py_face.set_blendshape(FaceBlendShape(i), frame_data[i])
        encoded_data.append(py_face.encode())

    default_animation_state['current_index'] = 0

    fast_blend_out = generate_blend_frames(
        facial_data, slow_blend_frames, default_animation_data, fps,
        FAST_BLENDSHAPES, mode='out', active_duration_sec=fast_duration,
        default_start_index=0              
    )

    slow_blend_out = generate_blend_frames(
        facial_data, slow_blend_frames, default_animation_data, fps,
        set(range(51)) - FAST_BLENDSHAPES, mode='out',
        default_start_index=0                                      
    )

    blend_out_frames = combine_frame_streams(slow_blend_out, fast_blend_out, FAST_BLENDSHAPES)

    for frame in blend_out_frames:
        for i in range(51):
            py_face.set_blendshape(FaceBlendShape(i), frame[i])
        encoded_data.append(py_face.encode())

    return encoded_data


def apply_selective_default_data(facial_data: List, default_animation_data: List[List[float]]):
    """
    FIXED VERSION: Only apply essential default data (like blinking) without overriding 
    mouth and eyebrow movements that should come from the facial animation.
    """
    # Only apply blink indices - DO NOT override mouth and eyebrow data
    essential_indices = {
        FaceBlendShape.EyeBlinkLeft.value, 
        FaceBlendShape.EyeBlinkRight.value
        # Removed mouth and eyebrow indices to prevent overriding
    }
    
    default_len = len(default_animation_data)
    for idx, frame in enumerate(facial_data):
        default_idx = idx % default_len
        for essential_idx in essential_indices:
            if essential_idx < len(frame):
                # Only blend, don't completely override
                frame[essential_idx] = max(frame[essential_idx], default_animation_data[default_idx][essential_idx])


# DEPRECATED - replaced with apply_selective_default_data
def apply_blink_to_facial_data(facial_data: List, default_animation_data: List[List[float]]):
    """
    ISSUE: This function was overriding ALL blink data, which includes eyebrow movements
    """
    pass  # Disabled to prevent conflicts


def smooth_facial_data(facial_data: list) -> list:
    if len(facial_data) < 2:
        return facial_data.copy()  

    smoothed_data = [facial_data[0]]
    for i in range(1, len(facial_data)):
        previous_frame = facial_data[i - 1]
        current_frame = facial_data[i]
        averaged_frame = [(a + b) / 2 for a, b in zip(previous_frame, current_frame)]
        smoothed_data.append(averaged_frame)
    
    return smoothed_data


def send_pre_encoded_data_to_unreal(encoded_facial_data: List[bytes], start_event, fps: int, socket_connection=None):
    try:
        # CRITICAL FIX: Stop default animation to prevent conflicts
        stop_default_animation.set()
        
        own_socket = False
        if socket_connection is None:
            socket_connection = create_socket_connection()
            own_socket = True

        start_event.wait()  
        frame_duration = 1 / fps  
        start_time = time.time()  

        for frame_index, frame_data in enumerate(encoded_facial_data):
            current_time = time.time()
            elapsed_time = current_time - start_time
            expected_time = frame_index * frame_duration 
            if elapsed_time < expected_time:
                time.sleep(expected_time - elapsed_time)
            elif elapsed_time > expected_time + frame_duration:
                continue

            socket_connection.sendall(frame_data)  

    except KeyboardInterrupt:
        pass
    finally:
        if own_socket:
            socket_connection.close()
        # Restart default animation after facial animation completes
        stop_default_animation.clear()