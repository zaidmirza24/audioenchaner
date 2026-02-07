import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
import os
import sys
from scipy.signal import butter, filtfilt

def highpass_filter(audio, sr, cutoff=80):
    """Remove low rumble (AC, fan, mic handling noise)"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, audio)

def soft_compress(audio, threshold=0.6, ratio=2.0):
    """Make voice more consistent and punchy"""
    compressed = np.copy(audio)
    mask = np.abs(audio) > threshold
    compressed[mask] = np.sign(audio[mask]) * (
        threshold + (np.abs(audio[mask]) - threshold) / ratio
    )
    return compressed

def enhance_speech_v2(input_file):

    base = os.path.basename(input_file)
    output_file = "pro_enhanced_" + base

    print("\nLoading audio...")
    audio, sr = librosa.load(input_file, sr=None)

    print("Step 1 — Removing low rumble...")
    audio = highpass_filter(audio, sr)

    print("Step 2 — Mild noise reduction...")
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=0.8
    )

    print("Step 3 — Light compression for clarity...")
    audio = soft_compress(audio)

    print("Step 4 — Smart loudness balancing...")
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, -14.0)

    print("Saving professional enhanced audio...")
    sf.write(output_file, audio, sr)

    print("\n✅ Done! New file:", output_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_enhancer.py \"your_file.mp3\"")
    else:
        enhance_speech_v2(sys.argv[1])
