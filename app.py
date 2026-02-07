import os
import shutil
import tempfile
import uuid
import numpy as np
import soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
from scipy.signal import butter, lfilter, resample_poly
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import gc

app = FastAPI(title="Audio Enhancer - Nuclear RAM Mode")

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

def apply_compression(y, threshold_db=-18, ratio=3):
    eps = 1e-9
    y_db = 20 * np.log10(np.abs(y) + eps)
    mask = y_db > threshold_db
    y_db[mask] = threshold_db + (y_db[mask] - threshold_db) / ratio
    y_comp = np.sign(y) * (10**(y_db / 20))
    return y_comp

def cleanup_temp_dir(path: str):
    try:
        shutil.rmtree(path)
    except:
        pass

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

        info = sf.info(input_path)
        orig_sr = info.samplerate
        total_frames = info.frames
        channels = info.channels

        # TARGET SPECS (16kHz Mono = Nuclear RAM Safety)
        TARGET_SR = 16000
        # 20 second chunks @ 16kHz = only 320,000 samples (Extremely light!)
        CHUNK_SECONDS = 20
        CHUNK_SIZE_SAMPLES = int(CHUNK_SECONDS * orig_sr)
        
        # We'll pre-calculate the noise profile from the first chunk to save memory in future chunks
        noise_profile = None

        with sf.SoundFile(output_path, mode='w', samplerate=TARGET_SR, channels=1) as out_f:
            for start in range(0, total_frames, CHUNK_SIZE_SAMPLES):
                # Read chunk in original SR
                chunk, _ = sf.read(input_path, start=start, frames=CHUNK_SIZE_SAMPLES, dtype='float32')
                
                # A. Mono conversion
                if channels > 1:
                    chunk = np.mean(chunk, axis=1)
                
                # B. Downsample to 16kHz using scipy (memory efficient)
                if orig_sr != TARGET_SR:
                    chunk = resample_poly(chunk, TARGET_SR, orig_sr)
                    gc.collect()

                # C. High-pass filter (80 Hz)
                chunk = highpass_filter(chunk, 80, TARGET_SR)
                gc.collect()

                # D. Noise Reduction
                if noise_profile is None:
                    # Capture noise profile from first 2 seconds of the first chunk
                    profile_len = min(int(2 * TARGET_SR), len(chunk))
                    noise_profile = chunk[:profile_len]
                
                # Use stationary=True for consistent memory usage and reuse profile
                chunk = nr.reduce_noise(y=chunk, sr=TARGET_SR, y_noise=noise_profile, stationary=True)
                gc.collect()

                # E. Light Compression
                chunk = apply_compression(chunk, threshold_db=-18, ratio=3)
                gc.collect()

                # F. Loudness Normalization to -14 LUFS
                try:
                    meter = pyln.Meter(TARGET_SR)
                    loudness = meter.integrated_loudness(chunk)
                    chunk = pyln.normalize.loudness(chunk, loudness, -14.0)
                    gc.collect()
                except:
                    pass

                out_f.write(chunk)
                del chunk
                gc.collect()

        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))

    except Exception as e:
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        print(f"Nuclear Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
