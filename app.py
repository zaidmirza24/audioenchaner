import os
import shutil
import tempfile
import uuid
import numpy as np
import soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
from scipy.signal import butter, lfilter, resample_poly, hilbert
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import gc

app = FastAPI(title="Audio Enhancer - Vintage Clean Mode")

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

def apply_compression(y, threshold_db=-22, ratio=4):
    eps = 1e-9
    y_db = 20 * np.log10(np.abs(y) + eps)
    mask = y_db > threshold_db
    y_db[mask] = threshold_db + (y_db[mask] - threshold_db) / ratio
    y_comp = np.sign(y) * (10**(y_db / 20))
    return y_comp

def apply_limiter(y):
    return np.clip(y, -0.99, 0.99)

def consonant_exciter(y, sr, amount=0.15):
    """
    Restores clarity to muffled recordings by adding controlled 
    harmonics to high frequencies (3.5kHz - 7.5kHz).
    """
    # 1. High-pass to isolate potential consonant region
    nyq = 0.5 * sr
    b, a = butter(4, 3500/nyq, btype='high')
    consonants = lfilter(b, a, y)
    
    # 2. Nonlinear excitation (create harmonics)
    # Simple soft-clipping/saturation for harmonics
    excited = np.sign(consonants) * (1 - np.exp(-np.abs(consonants * 1.5)))
    
    # 3. Blend back with original
    return y + (amount * excited)

def simple_dereverb(y, sr, decay=0.95, threshold=0.01):
    """
    Lightweight Envelope-based De-reverb.
    Suppresses 'room tails' by tracking the signal envelope.
    """
    # Calculate analytic signal envelope
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Smoothed envelope (leaky integrator)
    smoothed_env = np.zeros_like(amplitude_envelope)
    current_val = 0
    for i in range(len(amplitude_envelope)):
        if amplitude_envelope[i] > current_val:
            current_val = amplitude_envelope[i]
        else:
            current_val *= decay # decay factor
        smoothed_env[i] = current_val
    
    # Create attenuation mask
    # When signal drops below threshold relative to peak, attenuate
    # This targets the 'tails' of words
    gain_mask = np.ones_like(y)
    peak_val = np.max(smoothed_env) + 1e-9
    
    # If the smoothed envelope is falling fast, it's likely reverb tail
    mask = (smoothed_env / peak_val) < threshold
    gain_mask[mask] = 0.7 # Suppress tails by 30%
    
    return y * gain_mask

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

        # TARGET SPECS
        TARGET_SR = 16000
        CHUNK_SECONDS = 20
        OVERLAP_SECONDS = 1
        
        CHUNK_SAMPLES = int(CHUNK_SECONDS * orig_sr)
        OVERLAP_SAMPLES = int(OVERLAP_SECONDS * orig_sr)
        TARGET_OVERLAP_SAMPLES = int(OVERLAP_SECONDS * TARGET_SR)

        # 4. INITIAL NOISE PROFILE
        noise_chunk, _ = sf.read(input_path, start=0, frames=int(3 * orig_sr), dtype='float32')
        if channels > 1: noise_chunk = np.mean(noise_chunk, axis=1)
        if orig_sr != TARGET_SR: noise_chunk = resample_poly(noise_chunk, TARGET_SR, orig_sr)
        noise_profile = noise_chunk
        del noise_chunk
        gc.collect()

        prev_overlap = None
        current_time_s = 0

        with sf.SoundFile(output_path, mode='w', samplerate=TARGET_SR, channels=1) as out_f:
            for start in range(0, total_frames, CHUNK_SAMPLES):
                read_len = CHUNK_SAMPLES + OVERLAP_SAMPLES
                chunk, _ = sf.read(input_path, start=start, frames=read_len, dtype='float32')
                if len(chunk) == 0: break

                # A. Mono & Downsample
                if channels > 1:
                    chunk = np.mean(chunk, axis=1)
                if orig_sr != TARGET_SR:
                    chunk = resample_poly(chunk, TARGET_SR, orig_sr)
                gc.collect()

                # B. DYNAMIC NOISE PROFILING (Every 120s)
                # Helps adapt to changing crowd noise/tape hiss
                if current_time_s > 0 and current_time_s % 120 == 0:
                    profile_len = min(int(2 * TARGET_SR), len(chunk))
                    noise_profile = chunk[:profile_len]

                # C. Noise Reduction (Adaptive to profile)
                chunk = nr.reduce_noise(y=chunk, sr=TARGET_SR, y_noise=noise_profile, stationary=True, prop_decrease=0.88)
                gc.collect()

                # D. VINTAGE FIX 1: De-reverb (Echo Killer)
                chunk = simple_dereverb(chunk, TARGET_SR)
                gc.collect()

                # E. VINTAGE FIX 2: Consonant Exciter (Clarity)
                chunk = consonant_exciter(chunk, TARGET_SR)
                gc.collect()

                # F. Filtering (Voice Band-pass 80-7500Hz)
                chunk = highpass_filter(chunk, 80, TARGET_SR)
                nyq = 0.5 * TARGET_SR
                b, a = butter(4, 7500/nyq, btype='low')
                chunk = lfilter(b, a, chunk)
                gc.collect()

                # G. Compression & Normalization
                chunk = apply_compression(chunk)
                chunk = apply_limiter(chunk)
                
                try:
                    meter = pyln.Meter(TARGET_SR)
                    loudness = meter.integrated_loudness(chunk)
                    if loudness > -40.0:
                        chunk = pyln.normalize.loudness(chunk, loudness, -14.0)
                        chunk = apply_limiter(chunk)
                except:
                    pass

                # H. Overlap-Add Crossfade
                if prev_overlap is not None:
                    fade_in = np.linspace(0, 1, TARGET_OVERLAP_SAMPLES)
                    fade_out = 1.0 - fade_in
                    chunk[:TARGET_OVERLAP_SAMPLES] = (chunk[:TARGET_OVERLAP_SAMPLES] * fade_in) + (prev_overlap * fade_out)
                    
                write_size = min(len(chunk), int(CHUNK_SECONDS * TARGET_SR))
                out_f.write(chunk[:write_size])
                
                if len(chunk) > write_size:
                    prev_overlap = chunk[write_size:write_size + TARGET_OVERLAP_SAMPLES]
                else:
                    prev_overlap = None
                
                current_time_s += CHUNK_SECONDS
                del chunk
                gc.collect()

        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))

    except Exception as e:
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        print(f"Vintage Clean Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
