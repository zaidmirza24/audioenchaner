import os
import shutil
import tempfile
import uuid
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
from scipy.signal import butter, lfilter
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="Audio Enhancer API")

# Ensure static directory exists
Path("static").mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def apply_compression(y, threshold_db=-20, ratio=4):
    """Simple soft-knee compression logic"""
    # Convert to dB
    y_db = 20 * np.log10(np.abs(y) + 1e-9)
    
    # Apply compression above threshold
    mask = y_db > threshold_db
    y_db[mask] = threshold_db + (y_db[mask] - threshold_db) / ratio
    
    # Convert back to linear
    y_comp = np.sign(y) * (10**(y_db / 20))
    return y_comp

def cleanup_temp_dir(path: str):
    try:
        shutil.rmtree(path)
        print(f"Cleaned up {path}")
    except Exception as e:
        print(f"Error cleaning up {path}: {e}")

@app.post("/enhance")
async def enhance_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp3', '.wav')):
        raise HTTPException(status_code=400, detail="Only .mp3 and .wav files are supported")

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}_{file.filename}")
    output_path = os.path.join(temp_dir, f"pro_enhanced_{file.filename.split('.')[0]}.wav")

    try:
        # 1. Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Load audio
        y, sr = librosa.load(input_path, sr=None)

        # 3. Processing Pipeline
        
        # A. High-pass filter (cutoff = 80 Hz) to remove low-end rumble
        y = highpass_filter(y, 80, sr)

        # B. Noise Reduction (prop_decrease=0.8)
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)

        # C. Light Compression
        y = apply_compression(y, threshold_db=-18, ratio=3)

        # D. Loudness Normalization to -14 LUFS
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        y_normalized = pyln.normalize.loudness(y, loudness, -14.0)

        # 4. Save enhanced file as WAV
        sf.write(output_path, y_normalized, sr)

        background_tasks.add_task(cleanup_temp_dir, temp_dir)

        return FileResponse(
            output_path, 
            media_type="audio/wav", 
            filename=os.path.basename(output_path)
        )

    except Exception as e:
        # Cleanup in case of error too
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        print(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
