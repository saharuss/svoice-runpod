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
import whisper
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


# ── Load Whisper tiny model for voice quality scoring ────────
whisper_model = None

def load_whisper():
    """Load Whisper tiny model for transcription-based scoring."""
    global whisper_model
    try:
        whisper_model = whisper.load_model("tiny", device=DEVICE)
        logger.info("Whisper 'tiny' model loaded on %s", DEVICE)
    except Exception as e:
        logger.warning("Failed to load Whisper model: %s", e)


# ── Load Silero VAD for speech activity detection ────────────
vad_model = None
vad_utils = None

def load_vad():
    """Load Silero VAD model for speech activity detection."""
    global vad_model, vad_utils
    try:
        _model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
        vad_model = _model
        vad_utils = utils
        logger.info("Silero VAD loaded")
    except Exception as e:
        logger.warning("Failed to load Silero VAD: %s", e)


# Attempt to load all models at import time (cold start)
load_model()
load_whisper()
load_vad()


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
#  VOICE CONFIDENCE SCORING (AI-powered)
# ─────────────────────────────────────────────────────────────

def _vad_speech_ratio(audio: np.ndarray, sr: int) -> float:
    """
    Use Silero VAD to compute the fraction of audio that contains speech.

    Returns a value between 0.0 (no speech) and 1.0 (all speech).
    Fast (~5ms per track).
    """
    if vad_model is None:
        return 0.5  # Default if VAD not loaded

    try:
        # Silero VAD expects 16kHz mono
        if sr != 16000:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        tensor = torch.tensor(audio_16k, dtype=torch.float32)

        # Process in 512-sample chunks (32ms at 16kHz)
        window_size = 512
        speech_probs = []
        for i in range(0, len(tensor) - window_size, window_size):
            chunk = tensor[i:i + window_size]
            prob = vad_model(chunk, 16000).item()
            speech_probs.append(prob)

        if not speech_probs:
            return 0.0

        # Fraction of chunks with speech probability > 0.5
        speech_ratio = sum(1 for p in speech_probs if p > 0.5) / len(speech_probs)
        return speech_ratio

    except Exception as e:
        logger.warning("VAD scoring failed: %s", e)
        return 0.5


def _whisper_score(audio: np.ndarray, sr: int) -> dict:
    """
    Use Whisper tiny model to attempt transcription and extract quality metrics.

    Returns:
        {
            'avg_logprob': float,    # Higher = clearer speech (-0.2 is great, -1.5 is noise)
            'no_speech_prob': float, # 0.0 = definitely speech, 1.0 = definitely not
            'word_count': int,       # Number of words transcribed
            'text': str,             # The transcription itself
        }
    """
    if whisper_model is None:
        return {'avg_logprob': -1.0, 'no_speech_prob': 0.5, 'word_count': 0, 'text': ''}

    try:
        # Whisper expects 16kHz audio
        if sr != 16000:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        # Whisper's transcribe function expects a file path or numpy array
        # Use fp16=False if on CPU, True on GPU
        result = whisper.transcribe(
            whisper_model,
            audio_16k,
            language='en',
            fp16=(DEVICE.type == 'cuda'),
            verbose=False,
        )

        # Extract metrics from segments
        segments = result.get('segments', [])
        if not segments:
            return {
                'avg_logprob': -2.0,
                'no_speech_prob': 1.0,
                'word_count': 0,
                'text': '',
            }

        avg_logprob = sum(s.get('avg_logprob', -1.0) for s in segments) / len(segments)
        no_speech_prob = sum(s.get('no_speech_prob', 0.5) for s in segments) / len(segments)
        text = result.get('text', '').strip()
        word_count = len(text.split()) if text else 0

        return {
            'avg_logprob': avg_logprob,
            'no_speech_prob': no_speech_prob,
            'word_count': word_count,
            'text': text,
        }

    except Exception as e:
        logger.warning("Whisper scoring failed: %s", e)
        return {'avg_logprob': -1.0, 'no_speech_prob': 0.5, 'word_count': 0, 'text': ''}


def score_voice_confidence(audio: np.ndarray, sr: int) -> float:
    """
    AI-powered voice confidence scoring.

    Combines:
      1. Silero VAD   (30%) – fast speech detection, kills obvious noise
      2. Whisper clarity (50%) – transcription confidence (avg_logprob)
      3. Whisper no-speech (20%) – inverse of no_speech_prob

    Returns a confidence score between 0.0 and 1.0.
    """
    T = len(audio)
    if T == 0:
        return 0.0

    # Quick energy check — skip near-silent tracks entirely
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-7:
        logger.info("Voice confidence: 0.000 (silent track)")
        return 0.0

    # ── 1. Silero VAD: speech activity ratio ─────────────────
    vad_ratio = _vad_speech_ratio(audio, sr)

    # Fast reject: if VAD says < 5% speech, don’t bother with Whisper
    if vad_ratio < 0.05:
        confidence = vad_ratio * 0.3  # Will be near 0
        logger.info("Voice confidence: %.3f (VAD fast-reject, ratio=%.2f)",
                    confidence, vad_ratio)
        return round(confidence, 3)

    # ── 2. Whisper transcription scoring ───────────────────
    w = _whisper_score(audio, sr)

    # Map avg_logprob to 0–1 score:
    #   -0.2 and above = excellent (1.0)
    #   -1.0 = mediocre (0.5)
    #   -2.0 and below = garbage (0.0)
    clarity_score = np.clip((w['avg_logprob'] + 2.0) / 1.8, 0.0, 1.0)

    # Invert no_speech_prob: 0.0 no_speech = 1.0 score
    speech_score = 1.0 - w['no_speech_prob']

    # ── Combine ─────────────────────────────────────────
    confidence = (
        0.30 * vad_ratio +
        0.50 * clarity_score +
        0.20 * speech_score
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    logger.info(
        "Voice confidence: %.3f (vad=%.2f, clarity=%.2f[logp=%.2f], speech=%.2f, words=%d, text='%s')",
        confidence, vad_ratio, clarity_score, w['avg_logprob'],
        speech_score, w['word_count'], w['text'][:80]
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
