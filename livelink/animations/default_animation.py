# default_animation.py - FIXED VERSION

import time
import socket
import pandas as pd
from threading import Event

from livelink.connect.livelink_init import FaceBlendShape, UDP_IP, UDP_PORT
from livelink.animations.blending_anims import blend_animation_start_end
from livelink.animations.blending_anims import default_animation_state, blend_animation_start_end

def load_animation(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop(columns=['Timecode', 'BlendshapeCount'])
    
    # CRITICAL FIX: Be more selective about what to zero
    # Only zero eye look directions, not blinks or eyebrows
    selective_zero_cols = [1, 2, 3, 4, 8, 9, 10, 11]  # Only eye look directions
    selective_zero_cols = [i for i in selective_zero_cols if i < data.shape[1]] 
    data.iloc[:, selective_zero_cols] = 0.0

    return data.values

# ==================== DEFAULT ANIMATION SETUP ====================

# Path to the default animation CSV file
ground_truth_path = r"livelink/animations/default_anim/default.csv"

# Load the default animation data
default_animation_data = load_animation(ground_truth_path)

# Create the blended default animation data
default_animation_data = blend_animation_start_end(default_animation_data, blend_frames=16)

# Event to signal stopping of the default animation loop
stop_default_animation = Event()

# ADDED: Flag to pause default animation during facial animation
facial_animation_active = Event()

def default_animation_loop(py_face):
    """
    FIXED VERSION: Loops through the default animation but pauses when facial animation is active.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((UDP_IP, UDP_PORT))
        while not stop_default_animation.is_set():
            # CRITICAL FIX: Pause default animation when facial animation is active
            if facial_animation_active.is_set():
                time.sleep(0.1)  # Wait and check again
                continue
                
            for idx, frame in enumerate(default_animation_data):
                if stop_default_animation.is_set() or facial_animation_active.is_set():
                    break
                    
                # update shared state
                default_animation_state['current_index'] = idx

                for i, value in enumerate(frame):
                    py_face.set_blendshape(FaceBlendShape(i), float(value))
                try:
                    s.sendall(py_face.encode())
                except Exception as e:
                    print(f"Error in default animation sending: {e}")

                # maintain 60fps
                total_sleep = 1 / 60
                sleep_interval = 0.005
                while total_sleep > 0 and not stop_default_animation.is_set() and not facial_animation_active.is_set():
                    time.sleep(min(sleep_interval, total_sleep))
                    total_sleep -= sleep_interval

def pause_default_animation():
    """Pause the default animation when facial animation starts"""
    facial_animation_active.set()

def resume_default_animation():
    """Resume the default animation when facial animation ends"""
    facial_animation_active.clear()