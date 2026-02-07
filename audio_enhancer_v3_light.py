import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
from scipy.signal import butter, filtfilt

def highpass_filter(audio, sr, cutoff=80):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, audio)

def bandpass_filter(audio, sr, low=120, high=7000):
    nyquist = 0.5 * sr
    low_norm = low / nyquist
    high_norm = high / nyquist
    b, a = butter(4, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, audio)

def soft_compress(audio, threshold=0.6, ratio=2.0):
    compressed = np.copy(audio)
    mask = np.abs(audio) > threshold
    compressed[mask] = np.sign(audio[mask]) * (
        threshold + (np.abs(audio[mask]) - threshold) / ratio
    )
    return compressed

def restore_old_lecture_light(input_mp3):
    base = os.path.basename(input_mp3)
    final_output = "v3_light_restored_" + base

    print("\n🎧 Loading audio...")
    audio, sr = librosa.load(input_mp3, sr=None)

    print("🎛️ Step 1 — Removing rumble (AC, fan, mic handling)...")
    audio = highpass_filter(audio, sr)

    print("🎛️ Step 2 — Keeping only speech frequencies...")
    audio = bandpass_filter(audio, sr)

    print("🎛️ Step 3 — Smart noise reduction for old tapes...")
    audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.9)

    print("🎛️ Step 4 — Balancing dynamics...")
    audio = soft_compress(audio)

    print("🎛️ Step 5 — Professional loudness normalization...")
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, -14.0)

    print("\n💾 Saving restored audio:", final_output)
    sf.write(final_output, audio, sr)

    print("\n✅ V3 (Light) Restoration Complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python audio_enhancer_v3_light.py \"your_file.mp3\"")
    else:
        restore_old_lecture_light(sys.argv[1])
