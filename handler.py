"""
RunPod Serverless Handler for svoice – Speaker Voice Separation
================================================================
Accepts a mixed-speaker audio file and returns separated speaker tracks.

Input (JSON):
    {
        "input": {
            "audio_base64": "<base64-encoded WAV bytes>",   # OR
            "audio_url":    "https://example.com/mix.wav",  # one of the two
            "sample_rate":  8000,                           # optional, default 8000
            "num_speakers": 2                               # optional (unused at inference, baked into model)
        }
    }

Output (JSON):
    {
        "separated_tracks": [
            {"speaker": 1, "audio_base64": "<base64-encoded WAV>"},
            {"speaker": 2, "audio_base64": "<base64-encoded WAV>"},
            ...
        ],
        "num_speakers": 2
    }
"""

import base64
import io
import logging
import os
import sys
import tempfile

import numpy as np
import requests
import runpod
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt

# ── Ensure svoice package is importable ──────────────────────
sys.path.insert(0, "/app")

from svoice.utils import deserialize_model

logger = logging.getLogger("svoice-handler")
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

# ─────────────────────────────────────────────────────────────
#  MODEL LOADING  (runs once on cold start)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/checkpoint.th")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None


def load_model():
    """Load the svoice checkpoint and move to GPU."""
    global model
    if not os.path.isfile(MODEL_PATH):
        logger.warning(
            "Model checkpoint not found at %s – handler will fail on requests. "
            "Set MODEL_PATH env var or mount a volume.", MODEL_PATH
        )
        return
    logger.info("Loading model from %s ...", MODEL_PATH)
    pkg = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if "model" in pkg and isinstance(pkg["model"], dict) and "class" in pkg["model"]:
        # Standard svoice solver checkpoint: serialized model + best_state
        model = deserialize_model(pkg["model"])
        # If best_state exists, prefer it (it's from the best validation epoch)
        if "best_state" in pkg and pkg["best_state"] is not None:
            model.load_state_dict(pkg["best_state"])
            logger.info("Loaded best_state from checkpoint")
    elif "best_state" in pkg and "args" in pkg:
        # Checkpoint with args + best_state but no serialized model
        from svoice.models.swave import SWave
        args = pkg["args"]
        swave_args = dict(args.swave)
        swave_args["sr"] = args.sample_rate
        swave_args["segment"] = args.segment
        model = SWave(**swave_args)
        model.load_state_dict(pkg["best_state"])
        logger.info("Reconstructed model from args + best_state")
    elif "model" in pkg:
        m = pkg["model"]
        if isinstance(m, dict):
            model = deserialize_model(m)
        else:
            model = m
    else:
        model = deserialize_model(pkg)

    model.eval()
    model.to(DEVICE)
    logger.info("Model loaded on %s (%s parameters)",
                DEVICE, f"{sum(p.numel() for p in model.parameters()):,}")


# Attempt to load at import time (cold start)
load_model()


# ─────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ─────────────────────────────────────────────────────────────
def decode_audio(audio_base64: str, target_sr: int):
    """Decode base64 audio into a numpy array at the target sample rate."""
    audio_bytes = base64.b64decode(audio_base64)
    buf = io.BytesIO(audio_bytes)
    data, sr = sf.read(buf, dtype="float32")
    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Resample if needed
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data


def download_audio(url: str, target_sr: int):
    """Download audio from a URL and return as numpy array."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    buf = io.BytesIO(resp.content)
    data, sr = sf.read(buf, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data


def encode_audio(audio_np: np.ndarray, sr: int) -> str:
    """Encode a numpy audio array to base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
# ─────────────────────────────────────────────────────────────
#  POST-PROCESSING: Quality Optimizations
# ─────────────────────────────────────────────────────────────
TARGET_LUFS_DB = -20.0     # Target loudness (dBFS) – natural speech level
PEAK_LIMIT_DB = -1.0       # Hard peak ceiling to prevent clipping
SILENCE_THRESHOLD = 1e-6   # Below this RMS, consider the track silent
HIGHPASS_FREQ = 80.0       # High-pass cutoff (Hz) – removes DC + rumble
HIGHPASS_ORDER = 4         # Butterworth filter order
WIENER_EXPONENT = 2.0      # Wiener mask exponent (2 = standard power-law)
WIENER_FLOOR = 1e-8        # Spectral floor to avoid division by zero
CHUNK_SECONDS = 4.0        # Model was trained on 4s segments
OVERLAP_RATIO = 0.5        # 50% overlap between chunks
MAIN_VOICE_THRESHOLD = 0.3 # Minimum confidence to be considered a main voice


# ── 1. Wiener Post-Filter ────────────────────────────────────
def wiener_postfilter(mixture: np.ndarray, estimates: np.ndarray,
                     n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """
    Apply Wiener filtering in the STFT domain to reduce speaker bleed.

    For each source, computes a soft time-frequency mask:
        mask_i = |S_i|^p / (sum |S_j|^p + eps)
    then applies it to the mixture STFT to get a cleaner estimate.

    Args:
        mixture:   [T] mono mixture signal
        estimates: [C, T] raw separated source estimates
    Returns:
        [C, T] Wiener-filtered estimates
    """
    num_sources, T = estimates.shape

    # STFT of mixture
    mix_stft = np.fft.rfft(np.lib.stride_tricks.sliding_window_view(
        np.pad(mixture, (0, n_fft - len(mixture) % n_fft)),
        n_fft)[::hop_length] * np.hanning(n_fft))  # [frames, freq]

    # STFT of each estimate
    est_stfts = []
    for c in range(num_sources):
        padded = np.pad(estimates[c], (0, n_fft - len(estimates[c]) % n_fft))
        frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
        est_stfts.append(np.fft.rfft(frames * np.hanning(n_fft)))

    # Compute Wiener masks and apply
    power_sum = sum(np.abs(s) ** WIENER_EXPONENT for s in est_stfts) + WIENER_FLOOR
    filtered = np.zeros_like(estimates)

    for c in range(num_sources):
        mask = np.abs(est_stfts[c]) ** WIENER_EXPONENT / power_sum  # [frames, freq]
        masked_stft = mix_stft * mask

        # Inverse STFT via overlap-add
        recon_frames = np.fft.irfft(masked_stft, n=n_fft)  # [frames, n_fft]
        output = np.zeros(len(mixture) + n_fft)
        window = np.hanning(n_fft)
        for i, frame in enumerate(recon_frames):
            start = i * hop_length
            output[start:start + n_fft] += frame * window

        # Normalize by window overlap
        norm = np.zeros_like(output)
        for i in range(len(recon_frames)):
            start = i * hop_length
            norm[start:start + n_fft] += window ** 2
        norm = np.maximum(norm, 1e-8)
        output /= norm

        filtered[c] = output[:T]

    logger.info("Wiener post-filter applied (n_fft=%d, hop=%d)", n_fft, hop_length)
    return filtered


# ── 2. High-Pass Filter ──────────────────────────────────────
def highpass_filter(audio: np.ndarray, sr: int,
                   cutoff: float = HIGHPASS_FREQ,
                   order: int = HIGHPASS_ORDER) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter to remove DC offset and
    low-frequency rumble below `cutoff` Hz.
    """
    nyquist = sr / 2.0
    if cutoff >= nyquist:
        return audio  # Can't filter above Nyquist
    sos = butter(order, cutoff / nyquist, btype='high', output='sos')
    filtered = sosfilt(sos, audio).astype(np.float32)
    return filtered


# ── 3. Overlap-Add Chunked Inference ─────────────────────────
def inference_overlap_add(model_fn, audio: np.ndarray, sr: int,
                         chunk_sec: float = CHUNK_SECONDS,
                         overlap: float = OVERLAP_RATIO) -> np.ndarray:
    """
    Run model inference on overlapping chunks with cross-fade blending.

    Avoids edge artifacts that occur when processing long files in one shot
    (the model was trained on 4s segments).

    Args:
        model_fn:  callable that takes [1, T] tensor → [1, C, T] tensor
        audio:     [T] mono audio
        sr:        sample rate
        chunk_sec: chunk duration in seconds
        overlap:   overlap ratio (0.5 = 50%)
    Returns:
        [C, T] separated sources
    """
    chunk_len = int(chunk_sec * sr)
    hop = int(chunk_len * (1 - overlap))
    T = len(audio)

    # For short files, just run directly
    if T <= chunk_len:
        mixture = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        result = model_fn(mixture)[-1]  # [1, C, T]
        return result[:, :, :T].cpu().numpy()[0]  # [C, T]

    # Build Hann cross-fade window
    window = np.hanning(chunk_len).astype(np.float32)

    # Determine number of sources from a test chunk
    test_chunk = torch.tensor(audio[:chunk_len], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    test_out = model_fn(test_chunk)[-1]
    num_sources = test_out.shape[1]

    output = np.zeros((num_sources, T), dtype=np.float32)
    norm = np.zeros(T, dtype=np.float32)

    pos = 0
    while pos < T:
        end = min(pos + chunk_len, T)
        chunk = audio[pos:end]

        # Pad short final chunk
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))

        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            est = model_fn(chunk_tensor)[-1]  # [1, C, chunk_len]
        est = est[0].cpu().numpy()  # [C, chunk_len]

        # Apply window and accumulate
        actual_len = end - pos
        w = window[:actual_len]
        for c in range(num_sources):
            output[c, pos:end] += est[c, :actual_len] * w
        norm[pos:end] += w

        pos += hop

    # Normalize by accumulated window weights
    norm = np.maximum(norm, 1e-8)
    for c in range(num_sources):
        output[c] /= norm

    logger.info("Overlap-add inference: %d chunks (%.1fs each, %.0f%% overlap)",
                (T - chunk_len) // hop + 2, chunk_sec, overlap * 100)
    return output


# ── 4. Loudness Normalization ────────────────────────────────
def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to a natural loudness level.

    1. Compute RMS loudness of the signal.
    2. If nearly silent, return as-is (avoid amplifying noise).
    3. Apply gain to reach TARGET_LUFS_DB.
    4. Apply soft peak limiting so no sample exceeds PEAK_LIMIT_DB.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < SILENCE_THRESHOLD:
        logger.info("Track is silent (RMS=%.2e), skipping normalization", rms)
        return audio

    current_db = 20 * np.log10(rms + 1e-10)
    gain_db = TARGET_LUFS_DB - current_db
    gain_linear = 10 ** (gain_db / 20.0)
    audio = audio * gain_linear

    peak = np.max(np.abs(audio))
    peak_ceiling = 10 ** (PEAK_LIMIT_DB / 20.0)
    if peak > peak_ceiling:
        audio = audio * (peak_ceiling / peak)

    logger.info("Normalized: RMS %.1f dB → %.1f dB, peak %.4f",
                current_db, TARGET_LUFS_DB, np.max(np.abs(audio)))
    return audio


# ── Full post-processing pipeline ────────────────────────────
def postprocess_track(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply high-pass filter then loudness normalization to a single track."""
    audio = highpass_filter(audio, sr)
    audio = normalize_audio(audio)
    return audio


# ─────────────────────────────────────────────────────────────
#  VOICE CONFIDENCE SCORING
# ─────────────────────────────────────────────────────────────
def score_voice_confidence(audio: np.ndarray, sr: int) -> float:
    """
    Score how likely a track contains a clear, decipherable human voice.

    Combines four acoustic metrics:
      1. RMS Energy        – is the signal loud enough to be speech?
      2. Spectral Centroid – does the frequency center match speech (1–4 kHz)?
      3. Zero-Crossing Rate – speech has moderate ZCR; noise is very high.
      4. Voiced Frame Ratio – how many short frames have periodic structure
                              (detected via autocorrelation peaks)?

    Returns a confidence score between 0.0 and 1.0.
    """
    T = len(audio)
    if T == 0:
        return 0.0

    # ── 1. RMS Energy Score ──────────────────────────────────
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-7:
        return 0.0  # Silent track
    # Map RMS: below -60dB → 0, above -20dB → 1
    rms_db = 20 * np.log10(rms + 1e-10)
    rms_score = np.clip((rms_db + 60) / 40, 0.0, 1.0)

    # ── 2. Spectral Centroid Score ───────────────────────────
    # Use STFT to compute the mean spectral centroid
    n_fft = min(1024, T)
    hop = n_fft // 4
    # Pad if needed
    padded = np.pad(audio, (0, max(0, n_fft - T)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    window = np.hanning(n_fft)
    magnitudes = np.abs(np.fft.rfft(frames * window, n=n_fft))  # [num_frames, freq_bins]
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    mag_sum = magnitudes.sum(axis=1, keepdims=True) + 1e-10
    centroids = (magnitudes * freqs[np.newaxis, :]).sum(axis=1, keepdims=True) / mag_sum
    mean_centroid = centroids.mean()

    # Speech centroid is typically 1000–4000 Hz
    # Score peaks at ~2500 Hz, falls off outside 500–6000 Hz
    centroid_ideal = 2500.0
    centroid_width = 2000.0
    centroid_score = np.exp(-0.5 * ((mean_centroid - centroid_ideal) / centroid_width) ** 2)

    # ── 3. Zero-Crossing Rate Score ──────────────────────────
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))) > 0) / T
    # Normalize by sample rate — speech ZCR is typically 0.02–0.10
    zcr_normalized = zero_crossings  # already per-sample
    # Score: peak around 0.05, drops at very low (<0.01) or high (>0.2)
    zcr_ideal = 0.05
    zcr_width = 0.06
    zcr_score = np.exp(-0.5 * ((zcr_normalized - zcr_ideal) / zcr_width) ** 2)

    # ── 4. Voiced Frame Ratio ────────────────────────────────
    # Check short frames for periodicity using autocorrelation
    frame_len = min(int(0.03 * sr), T)  # 30ms frames
    frame_hop = frame_len // 2
    voiced_count = 0
    total_frames = 0

    for start in range(0, T - frame_len, frame_hop):
        frame = audio[start:start + frame_len]
        frame_energy = np.sum(frame ** 2)
        if frame_energy < 1e-10:
            total_frames += 1
            continue

        # Autocorrelation (normalized)
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]  # Take positive lags only
        corr = corr / (corr[0] + 1e-10)  # Normalize

        # Look for peaks in the pitch range (80–400 Hz)
        min_lag = max(1, int(sr / 400))  # 400 Hz
        max_lag = min(len(corr) - 1, int(sr / 80))  # 80 Hz

        if max_lag > min_lag:
            search_region = corr[min_lag:max_lag + 1]
            peak_val = np.max(search_region)
            if peak_val > 0.3:  # Voiced threshold
                voiced_count += 1

        total_frames += 1

    voiced_ratio = voiced_count / max(total_frames, 1)

    # ── Combine Scores ───────────────────────────────────────
    # Weighted combination: voiced ratio is the strongest indicator
    weights = {
        'rms': 0.15,
        'centroid': 0.20,
        'zcr': 0.15,
        'voiced': 0.50,
    }
    confidence = (
        weights['rms'] * rms_score +
        weights['centroid'] * centroid_score +
        weights['zcr'] * zcr_score +
        weights['voiced'] * voiced_ratio
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    logger.info(
        "Voice confidence: %.3f (rms=%.2f, centroid=%.2f[%.0fHz], zcr=%.2f, voiced=%.2f)",
        confidence, rms_score, centroid_score, mean_centroid, zcr_score, voiced_ratio
    )
    return round(confidence, 3)
# ─────────────────────────────────────────────────────────────
def handler(job):
    """RunPod serverless handler – separates speakers from a mixed audio."""
    global model

    job_input = job["input"]

    # ── Validate input ───────────────────────────────────────
    audio_b64 = job_input.get("audio_base64")
    audio_url = job_input.get("audio_url")
    if not audio_b64 and not audio_url:
        return {"error": "Provide either 'audio_base64' or 'audio_url' in input."}

    sample_rate = int(job_input.get("sample_rate", 16000))

    # ── Check model is loaded ────────────────────────────────
    if model is None:
        return {"error": f"Model not loaded. Checkpoint not found at {MODEL_PATH}."}

    # ── Load audio ───────────────────────────────────────────
    try:
        if audio_b64:
            audio_np = decode_audio(audio_b64, sample_rate)
        else:
            audio_np = download_audio(audio_url, sample_rate)
    except Exception as e:
        return {"error": f"Failed to load audio: {str(e)}"}

    logger.info("Audio loaded: %.2f sec @ %d Hz", len(audio_np) / sample_rate, sample_rate)

    # ── Run inference (overlap-add for long files) ────────────
    try:
        estimate_sources = inference_overlap_add(
            model, audio_np, sample_rate
        )  # [C, T]
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    # ── Wiener post-filter to reduce speaker bleed ───────────
    try:
        estimate_sources = wiener_postfilter(audio_np, estimate_sources)
    except Exception as e:
        logger.warning("Wiener filter failed, using raw estimates: %s", e)

    # ── Post-process & score each track ───────────────────────
    num_speakers = estimate_sources.shape[0]
    tracks = []
    for c in range(num_speakers):
        track_audio = postprocess_track(estimate_sources[c], sample_rate)
        confidence = score_voice_confidence(track_audio, sample_rate)
        tracks.append({
            "speaker": c + 1,
            "confidence": confidence,
            "is_main": confidence >= MAIN_VOICE_THRESHOLD,
            "audio_base64": encode_audio(track_audio, sample_rate),
        })

    # Sort by confidence descending
    tracks.sort(key=lambda t: t["confidence"], reverse=True)

    main_count = sum(1 for t in tracks if t["is_main"])
    logger.info("Separated %d speakers (%d main voices)", num_speakers, main_count)
    return {
        "separated_tracks": tracks,
        "num_speakers": num_speakers,
        "main_voices": main_count,
    }


# ── Start RunPod serverless worker ───────────────────────────
runpod.serverless.start({"handler": handler})
