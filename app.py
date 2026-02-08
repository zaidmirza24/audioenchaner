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
from gradio_client import Client, handle_file

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

def apply_vocal_warmth(y, sr, amount=0.3):
    """
    Adds tube-style warmth using soft saturation and a low-mid boost.
    """
    # 1. Subtle low-mid boost (200-400Hz)
    nyq = 0.5 * sr
    b, a = butter(2, [150/nyq, 450/nyq], btype='band')
    warm_band = lfilter(b, a, y)
    
    # 2. Soft saturation
    saturated = np.tanh(y * (1 + amount)) / (1 + amount)
    
    # 3. Blend
    return saturated + (0.1 * warm_band)

def apply_vocal_presence(y, sr, amount=0.15):
    """
    Boosts the 'presence' range (3kHz-5kHz) for better articulation.
    """
    nyq = 0.5 * sr
    b, a = butter(2, [3000/nyq, 5500/nyq], btype='band')
    presence_band = lfilter(b, a, y)
    return y + (amount * presence_band)

def apply_deesser(y, sr, threshold=0.15):
    """
    Attenuates harsh high frequencies (5kHz-8kHz) typical of AI artifacts.
    """
    nyq = 0.5 * sr
    b, a = butter(2, 5500/nyq, btype='high')
    sibilance = lfilter(b, a, y)
    
    # Simple dynamic suppression
    mask = np.abs(sibilance) > threshold
    y[mask] *= 0.85 # Reduce harsh peaks
    return y

def apply_noise_guard(y, threshold=0.015, attack=0.05, release=0.2):
    """
    Advanced noise gate to ensure absolute silence between words.
    Helps eliminate AI 'air' noise.
    """
    # Simple RMS-based gate
    window_size = int(0.02 * 16000) # 20ms windows
    num_windows = len(y) // window_size
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = y[start:end]
        rms = np.sqrt(np.mean(window**2))
        
        if rms < threshold:
            y[start:end] = 0.0
            
    return y

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
async def enhance_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...), use_ai: str = "false"):
    use_ai = use_ai.lower() == "true"
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
                # If use_ai is True, we go AGGRESSIVE (1.0) because AI reconstructs the voice
                nr_strength = 1.0 if use_ai else 0.88
                chunk = nr.reduce_noise(y=chunk, sr=TARGET_SR, y_noise=noise_profile, stationary=True, prop_decrease=nr_strength)
                gc.collect()

                # D. VINTAGE FIX 1: De-reverb (Echo Killer)
                chunk = simple_dereverb(chunk, TARGET_SR)
                gc.collect()

                # E. VINTAGE FIX 2: Consonant Exciter (Clarity)
                chunk = consonant_exciter(chunk, TARGET_SR)
                gc.collect()

                # F. Filtering (Voice Band-pass 80-7500Hz)
                chunk = highpass_filter(chunk, 80, TARGET_SR)
                # Hard cut high-end hiss if using AI
                cutoff = 7000 if use_ai else 7500
                nyq = 0.5 * TARGET_SR
                b, a = butter(4, cutoff/nyq, btype='low')
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
                    actual_overlap = min(len(prev_overlap), len(chunk), TARGET_OVERLAP_SAMPLES)
                    if actual_overlap > 0:
                        fade_in = np.linspace(0, 1, actual_overlap)
                        fade_out = 1.0 - fade_in
                        chunk[:actual_overlap] = (chunk[:actual_overlap] * fade_in) + (prev_overlap[:actual_overlap] * fade_out)
                    
                write_size = min(len(chunk), int(CHUNK_SECONDS * TARGET_SR))
                out_f.write(chunk[:write_size])
                
                if len(chunk) > write_size:
                    prev_overlap = chunk[write_size:write_size + TARGET_OVERLAP_SAMPLES]
                else:
                    prev_overlap = None
                
                current_time_s += CHUNK_SECONDS
                del chunk
                gc.collect()

        # --- AI ENHANCEMENT BRANCH ---
        if use_ai:
            try:
                # Use Resemble Enhance on Hugging Face (High Quality)
                # This model performs "Super-Resolution" and "Neural Restoration"
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    raise Exception("HF_TOKEN environment variable not found. Please set it in Render dashboard.")
                
                client = Client("ResembleAI/resemble-enhance", token=hf_token)
                result = client.predict(
                    handle_file(output_path), # input_audio
                    "Midpoint",             # cfm_ode_solver
                    128,                     # cfm_nfe (MAXIMIZED for Detail)
                    0.5,                     # cfm_prior_temperature
                    True,                    # denoise_before_enhancement
                    api_name="/predict"
                )
                
                # Copy the AI result back to our output_path
                shutil.copy(result[1], output_path)
                
                # --- POST-AI SUPER POLISH ---
                print("Applying Super Polish...")
                data, sr = sf.read(output_path)
                
                # 1. Warmth & Weight
                data = apply_vocal_warmth(data, sr)
                
                # 2. Presence & Clarity
                data = apply_vocal_presence(data, sr)
                
                # 3. De-Esser (Softens AI harshness)
                data = apply_deesser(data, sr)
                
                # 4. Noise Guard (Ensures silence in pauses)
                data = apply_noise_guard(data)
                
                # 5. Final Mastering Pass
                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(data)
                data = pyln.normalize.loudness(data, loudness, -14.0)
                data = apply_limiter(data)
                
                sf.write(output_path, data, sr)
                print("Super Polish complete.")
            except Exception as ai_e:
                print(f"AI Enhancement failed, falling back to Standard: {ai_e}")
                # We continue with the existing 'output_path' which already has the standard enhancement

        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))

    except Exception as e:
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        print(f"Vintage Clean Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
