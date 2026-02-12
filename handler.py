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
#  AUDIO NORMALIZATION
# ─────────────────────────────────────────────────────────────
TARGET_LUFS_DB = -20.0   # Target loudness (dBFS) – natural speech level
PEAK_LIMIT_DB = -1.0     # Hard peak ceiling to prevent clipping
SILENCE_THRESHOLD = 1e-6 # Below this RMS, consider the track silent


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to a natural loudness level.

    1. Compute RMS loudness of the signal.
    2. If nearly silent, return as-is (avoid amplifying noise).
    3. Apply gain to reach TARGET_LUFS_DB.
    4. Apply soft peak limiting so no sample exceeds PEAK_LIMIT_DB.
    """
    # Skip silent / near-silent tracks
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < SILENCE_THRESHOLD:
        logger.info("Track is silent (RMS=%.2e), skipping normalization", rms)
        return audio

    # Current RMS in dB
    current_db = 20 * np.log10(rms + 1e-10)

    # Gain needed to reach target
    gain_db = TARGET_LUFS_DB - current_db
    gain_linear = 10 ** (gain_db / 20.0)

    audio = audio * gain_linear

    # Peak limiting: if any sample exceeds the ceiling, scale down
    peak = np.max(np.abs(audio))
    peak_ceiling = 10 ** (PEAK_LIMIT_DB / 20.0)  # ~0.891
    if peak > peak_ceiling:
        audio = audio * (peak_ceiling / peak)

    logger.info("Normalized: RMS %.1f dB → %.1f dB, peak %.4f",
                current_db, TARGET_LUFS_DB, np.max(np.abs(audio)))
    return audio
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

    # ── Run inference ────────────────────────────────────────
    try:
        with torch.no_grad():
            mixture = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            length = torch.tensor([audio_np.shape[0]], dtype=torch.long).to(DEVICE)

            # model returns list of outputs per RNN layer; take the last one
            estimate_sources = model(mixture)[-1]  # [1, C, T]

            # Trim to original length
            estimate_sources = estimate_sources[:, :, :audio_np.shape[0]]
            estimate_sources = estimate_sources.cpu().numpy()
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    # ── Build response ───────────────────────────────────────
    num_speakers = estimate_sources.shape[1]
    tracks = []
    for c in range(num_speakers):
        track_audio = estimate_sources[0, c, :]
        track_audio = normalize_audio(track_audio)
        tracks.append({
            "speaker": c + 1,
            "audio_base64": encode_audio(track_audio, sample_rate),
        })

    logger.info("Separated %d speakers", num_speakers)
    return {
        "separated_tracks": tracks,
        "num_speakers": num_speakers,
    }


# ── Start RunPod serverless worker ───────────────────────────
runpod.serverless.start({"handler": handler})
