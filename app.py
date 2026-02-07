import os
import shutil
import tempfile
import uuid
import numpy as np
import soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
from scipy.signal import butter, lfilter
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="Audio Enhancer API - Chunked V2")

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
    """Simple soft-knee compression logic"""
    # Safeguard against zero energy
    eps = 1e-9
    # Convert to dB
    y_db = 20 * np.log10(np.abs(y) + eps)
    
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
        # 1. Save uploaded file to temp disk
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Get Audio Info
        info = sf.info(input_path)
        sr = info.samplerate
        total_frames = info.frames
        channels = info.channels

        # Chunk Settings: 60 seconds (Reduced from 300 for 512MB RAM safety)
        CHUNK_SECONDS = 60
        CHUNK_SIZE = int(CHUNK_SECONDS * sr)
        
        import gc

        with sf.SoundFile(output_path, mode='w', samplerate=sr, channels=1) as out_f:
            # 3. Process in Chunks
            for start in range(0, total_frames, CHUNK_SIZE):
                # Read chunk
                chunk, _ = sf.read(input_path, start=start, frames=CHUNK_SIZE, dtype='float32')
                
                # Convert to Mono if Stereo immediately
                if channels > 1:
                    chunk = np.mean(chunk, axis=1)
                    gc.collect()
                
                # A. High-pass filter (80 Hz)
                chunk = highpass_filter(chunk, 80, sr)
                gc.collect()

                # B. Noise Reduction (prop_decrease=0.8)
                # Stationary=True can also save RAM if noise is constant
                chunk = nr.reduce_noise(y=chunk, sr=sr, prop_decrease=0.8)
                gc.collect()

                # C. Light Compression
                chunk = apply_compression(chunk, threshold_db=-18, ratio=3)
                gc.collect()

                # D. Loudness Normalization to -14 LUFS
                try:
                    meter = pyln.Meter(sr)
                    loudness = meter.integrated_loudness(chunk)
                    chunk = pyln.normalize.loudness(chunk, loudness, -14.0)
                    gc.collect()
                except:
                    pass

                # Write chunk to final file
                out_f.write(chunk)
                
                # Final cleanup for this chunk
                del chunk
                gc.collect()

        background_tasks.add_task(cleanup_temp_dir, temp_dir)

        return FileResponse(
            output_path, 
            media_type="audio/wav", 
            filename=os.path.basename(output_path)
        )

    except Exception as e:
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        print(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
