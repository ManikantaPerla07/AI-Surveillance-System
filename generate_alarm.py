"""
Generate a simple alarm sound (alarm.wav) for the surveillance system.
This creates a 1-second beeping tone.
"""

import numpy as np
from scipy.io import wavfile
import os

def generate_alarm_sound(filename="alarm.wav", duration=1.0, frequency=1000, sample_rate=44100):
    """
    Generate a simple beep alarm sound.
    
    Args:
        filename: Output filename
        duration: Duration in seconds
        frequency: Frequency of the beep in Hz
        sample_rate: Sample rate in Hz
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate sine wave at specified frequency
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add envelope to avoid clicks at start/end (fade in/out)
    fade_duration = int(0.05 * sample_rate)  # 50ms fade
    if len(audio) > 2 * fade_duration:
        fade_in = np.linspace(0, 1, fade_duration)
        fade_out = np.linspace(1, 0, fade_duration)
        audio[:fade_duration] *= fade_in
        audio[-fade_duration:] *= fade_out
    
    # Convert to 16-bit PCM
    audio_int16 = np.int16(audio * 32767)
    
    # Write to WAV file
    wavfile.write(filename, sample_rate, audio_int16)
    print(f"✓ Alarm sound generated: {filename}")

if __name__ == "__main__":
    generate_alarm_sound()
