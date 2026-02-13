"""
RunPod Serverless Handler for svoice – Speaker Voice Separation
================================================================
Accepts a mixed-speaker audio file and returns separated speaker tracks.

Pipeline:
  1. svoice model separates mixture into N source estimates
  2. Whisper + Silero VAD scores each track for voice confidence
  3. Main voices (confidence >= threshold) are enhanced via ElevenLabs Voice Isolation
  4. Main voices are transcribed via ElevenLabs Speech-to-Text (Scribe v2)
  5. All tracks returned with confidence scores, audio, and transcriptions

Input (JSON):
    {
        "input": {
            "audio_base64": "<base64-encoded WAV bytes>",   # OR
            "audio_url":    "https://example.com/mix.wav",  # one of the two
            "sample_rate":  16000                            # optional, default 16000
        }
    }

Output (JSON):
    {
        "separated_tracks": [
            {
                "speaker": 1,
                "confidence": 0.92,
                "is_main": true,
                "audio_base64": "<base64-encoded WAV>",
                "transcription": "Hello, how are you?"
            },
            ...
        ],
        "num_speakers": 7,
        "main_voices": 2
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

# ElevenLabs API
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "sk_a400e53ea64bc1a2c64f3976394662a7f8be4d435c5b0a18")
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

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
        model = deserialize_model(pkg["model"])
        if "best_state" in pkg and pkg["best_state"] is not None:
            model.load_state_dict(pkg["best_state"])
            logger.info("Loaded best_state from checkpoint")
    elif "best_state" in pkg and "args" in pkg:
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


# ── Load SpeechBrain ECAPA-TDNN for speaker verification ─────
ecapa_model = None

def load_ecapa():
    """Load SpeechBrain ECAPA-TDNN pretrained model for speaker embeddings."""
    global ecapa_model
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/app/models/ecapa",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        logger.info("SpeechBrain ECAPA-TDNN loaded")
    except Exception as e:
        logger.warning("Failed to load ECAPA-TDNN: %s", e)


# Attempt to load all models at import time (cold start)
load_model()
load_whisper()
load_vad()
load_ecapa()


# ─────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ─────────────────────────────────────────────────────────────
def decode_audio(audio_base64: str, target_sr: int):
    """Decode base64 audio into a numpy array at the target sample rate.
    Supports WAV, FLAC, OGG via soundfile, and M4A/AAC/MP3 via pydub+ffmpeg fallback."""
    audio_bytes = base64.b64decode(audio_base64)
    buf = io.BytesIO(audio_bytes)

    try:
        data, sr = sf.read(buf, dtype="float32")
    except Exception:
        # Fallback for formats soundfile can't handle (m4a, aac, some mp3)
        from pydub import AudioSegment
        buf.seek(0)
        seg = AudioSegment.from_file(buf)
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels)).mean(axis=1)
        data = samples / (2 ** (seg.sample_width * 8 - 1))  # normalize to [-1, 1]

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
    """Encode a numpy audio array to base64 WAV string (PCM_16 for browser compatibility)."""
    buf = io.BytesIO()
    # Clip to [-1, 1] before PCM_16 encoding to avoid clipping distortion
    audio_clipped = np.clip(audio_np, -1.0, 1.0)
    sf.write(buf, audio_clipped, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def audio_to_wav_bytes(audio_np: np.ndarray, sr: int) -> bytes:
    """Convert numpy audio to WAV bytes for API uploads."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────
#  SPEAKER EMBEDDING (ECAPA-TDNN)
# ─────────────────────────────────────────────────────────────
def extract_speaker_embedding(audio_np: np.ndarray, sr: int):
    """
    Extract a 192-dim speaker embedding using ECAPA-TDNN.
    Returns a normalised numpy vector, or None on failure.
    """
    if ecapa_model is None:
        logger.warning("ECAPA model not loaded, can't extract embedding")
        return None
    try:
        # ECAPA expects mono float32 tensor at 16 kHz
        if sr != 16000:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        embedding = ecapa_model.encode_batch(waveform)  # [1, 1, 192]
        emb = embedding.squeeze().cpu().numpy()
        # L2 normalise
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception as e:
        logger.warning("Embedding extraction failed: %s", e)
        return None


def match_speaker(target_emb: np.ndarray, track_embeddings: list):
    """
    Find the best-matching track by cosine similarity.
    Returns (best_index, best_similarity, all_similarities).
    """
    similarities = []
    for emb in track_embeddings:
        if emb is not None and target_emb is not None:
            sim = float(np.dot(target_emb, emb))  # both L2-normalised
        else:
            sim = 0.0
        similarities.append(sim)
    best_idx = int(np.argmax(similarities))
    return best_idx, similarities[best_idx], similarities


# ─────────────────────────────────────────────────────────────
#  SIMPLE INFERENCE (direct model forward pass)
# ─────────────────────────────────────────────────────────────
MAIN_VOICE_THRESHOLD = 0.3  # Minimum confidence to be considered a main voice
SILENCE_THRESHOLD = 1e-6    # Below this RMS, consider the track silent
TARGET_LUFS_DB = -20.0      # Target loudness (dBFS)
PEAK_LIMIT_DB = -1.0        # Hard peak ceiling


def run_inference(model_fn, audio: np.ndarray) -> np.ndarray:
    """
    Run svoice model on the full audio signal.

    Args:
        model_fn: the loaded svoice model
        audio: [T] mono audio numpy array
    Returns:
        [C, T] separated source estimates
    """
    mixture = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        result = model_fn(mixture)[-1]  # [1, C, T]
    return result[0].cpu().numpy()  # [C, T]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Simple loudness normalization with peak limiting."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < SILENCE_THRESHOLD:
        return audio

    current_db = 20 * np.log10(rms + 1e-10)
    gain_db = TARGET_LUFS_DB - current_db
    gain_linear = 10 ** (gain_db / 20.0)
    audio = audio * gain_linear

    peak = np.max(np.abs(audio))
    peak_ceiling = 10 ** (PEAK_LIMIT_DB / 20.0)
    if peak > peak_ceiling:
        audio = audio * (peak_ceiling / peak)

    return audio


# ─────────────────────────────────────────────────────────────
#  VOICE CONFIDENCE SCORING (AI-powered: Whisper + Silero VAD)
# ─────────────────────────────────────────────────────────────

def _vad_speech_ratio(audio: np.ndarray, sr: int) -> float:
    """
    Use Silero VAD to compute the fraction of audio that contains speech.
    Returns 0.0 (no speech) to 1.0 (all speech). Fast (~5ms).
    """
    if vad_model is None:
        return 0.5

    try:
        if sr != 16000:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        tensor = torch.tensor(audio_16k, dtype=torch.float32)
        window_size = 512
        speech_probs = []
        for i in range(0, len(tensor) - window_size, window_size):
            chunk = tensor[i:i + window_size]
            prob = vad_model(chunk, 16000).item()
            speech_probs.append(prob)

        if not speech_probs:
            return 0.0

        return sum(1 for p in speech_probs if p > 0.5) / len(speech_probs)

    except Exception as e:
        logger.warning("VAD scoring failed: %s", e)
        return 0.5


def _whisper_score(audio: np.ndarray, sr: int) -> dict:
    """
    Use Whisper tiny to attempt transcription and extract quality metrics.
    Returns avg_logprob, no_speech_prob, word_count, text.
    """
    if whisper_model is None:
        return {'avg_logprob': -1.0, 'no_speech_prob': 0.5, 'word_count': 0, 'text': ''}

    try:
        if sr != 16000:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        result = whisper.transcribe(
            whisper_model,
            audio_16k,
            language='en',
            fp16=(DEVICE.type == 'cuda'),
            verbose=False,
        )

        segments = result.get('segments', [])
        if not segments:
            return {'avg_logprob': -2.0, 'no_speech_prob': 1.0, 'word_count': 0, 'text': ''}

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
    Combines Silero VAD (30%), Whisper clarity (50%), Whisper no-speech (20%).
    """
    if len(audio) == 0:
        return 0.0

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-7:
        logger.info("Voice confidence: 0.000 (silent track)")
        return 0.0

    vad_ratio = _vad_speech_ratio(audio, sr)

    if vad_ratio < 0.05:
        confidence = vad_ratio * 0.3
        logger.info("Voice confidence: %.3f (VAD fast-reject, ratio=%.2f)",
                    confidence, vad_ratio)
        return round(confidence, 3)

    w = _whisper_score(audio, sr)
    clarity_score = np.clip((w['avg_logprob'] + 2.0) / 1.8, 0.0, 1.0)
    speech_score = 1.0 - w['no_speech_prob']

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
#  ELEVENLABS API: Voice Isolation + Transcription
# ─────────────────────────────────────────────────────────────

def elevenlabs_voice_isolate(audio_np: np.ndarray, sr: int) -> np.ndarray:
    """
    Use ElevenLabs Audio Isolation API to enhance voice clarity
    by removing background noise and artifacts.

    Returns enhanced audio as numpy array, or original on failure.
    """
    try:
        wav_bytes = audio_to_wav_bytes(audio_np, sr)

        resp = requests.post(
            f"{ELEVENLABS_BASE_URL}/audio-isolation",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            files={"audio": ("track.wav", wav_bytes, "audio/wav")},
            timeout=60,
        )

        if resp.status_code != 200:
            logger.warning("ElevenLabs voice isolation failed (HTTP %d): %s",
                          resp.status_code, resp.text[:200])
            return audio_np

        # Response is the isolated audio bytes (WAV/MP3)
        enhanced_buf = io.BytesIO(resp.content)
        enhanced_audio, enhanced_sr = sf.read(enhanced_buf, dtype="float32")

        # Convert to mono if needed
        if enhanced_audio.ndim > 1:
            enhanced_audio = enhanced_audio.mean(axis=1)

        # Resample back if needed
        if enhanced_sr != sr:
            import librosa
            enhanced_audio = librosa.resample(enhanced_audio, orig_sr=enhanced_sr, target_sr=sr)

        logger.info("ElevenLabs voice isolation applied (%.2f sec)",
                    len(enhanced_audio) / sr)
        return enhanced_audio

    except Exception as e:
        logger.warning("ElevenLabs voice isolation error: %s", e)
        return audio_np


def elevenlabs_transcribe(audio_np: np.ndarray, sr: int) -> dict:
    """
    Use ElevenLabs Speech-to-Text (Scribe v2) to transcribe audio.

    Returns dict with 'text' and 'words' (word-level timestamps).
    Each word: {text: str, start: float, end: float}
    """
    empty = {"text": "", "words": []}
    try:
        wav_bytes = audio_to_wav_bytes(audio_np, sr)

        resp = requests.post(
            f"{ELEVENLABS_BASE_URL}/speech-to-text",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            data={
                "model_id": "scribe_v2",
                "language_code": "en",
                "timestamps_granularity": "word",
            },
            files={"file": ("track.wav", wav_bytes, "audio/wav")},
            timeout=60,
        )

        if resp.status_code != 200:
            logger.warning("ElevenLabs transcription failed (HTTP %d): %s",
                          resp.status_code, resp.text[:200])
            return empty

        result = resp.json()
        text = result.get("text", "").strip()

        # Extract word-level timestamps
        raw_words = result.get("words", [])
        words = []
        for w in raw_words:
            if w.get("type") == "word" and w.get("text", "").strip():
                words.append({
                    "text": w["text"].strip(),
                    "start": round(w.get("start", 0.0), 3),
                    "end": round(w.get("end", 0.0), 3),
                })

        logger.info("ElevenLabs transcription (%d words, %d timed): '%s'",
                    len(text.split()) if text else 0, len(words), text[:100])
        return {"text": text, "words": words}

    except Exception as e:
        logger.warning("ElevenLabs transcription error: %s", e)
        return empty


# ─────────────────────────────────────────────────────────────
#  MAIN HANDLER
# ─────────────────────────────────────────────────────────────
def handler(job):
    """RunPod serverless handler – separates speakers from a mixed audio."""
    global model

    job_input = job["input"]

    # ── Validate input ───────────────────────────────────────
    audio_b64 = job_input.get("audio_base64")
    audio_url = job_input.get("audio_url")
    identities = job_input.get("identities", [])  # [{name, audio_base64}]
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

    # ── Run inference (direct forward pass) ───────────────────
    try:
        estimate_sources = run_inference(model, audio_np)  # [C, T]
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    # ── Normalize & score each track ─────────────────────────
    num_speakers = estimate_sources.shape[0]
    tracks = []
    for c in range(num_speakers):
        track_audio = normalize_audio(estimate_sources[c])
        confidence = score_voice_confidence(track_audio, sample_rate)
        tracks.append({
            "speaker": c + 1,
            "confidence": confidence,
            "is_main": confidence >= MAIN_VOICE_THRESHOLD,
            "audio_np": track_audio,  # Keep numpy for ElevenLabs processing
        })

    # Sort by confidence descending
    tracks.sort(key=lambda t: t["confidence"], reverse=True)

    # ── ElevenLabs: enhance + transcribe main voices ─────────
    for track in tracks:
        if track["is_main"]:
            # Voice isolation to enhance clarity
            enhanced = elevenlabs_voice_isolate(track["audio_np"], sample_rate)
            track["audio_np"] = enhanced

            # Transcribe the enhanced audio
            track["transcription"] = elevenlabs_transcribe(enhanced, sample_rate)
        else:
            track["transcription"] = {"text": "", "words": []}

    # ── Identity matching (optional) ──────────────────────────
    # Match each track against all provided voice identities.
    # For each track, find the best-matching identity above threshold.
    IDENTITY_MATCH_THRESHOLD = 0.25
    identity_matches = {}  # {track_index: {name, similarity}}

    if identities:
        try:
            # Extract embeddings for all tracks
            track_embs = [extract_speaker_embedding(t["audio_np"], sample_rate) for t in tracks]

            # Extract embeddings for all identities
            identity_embs = []
            for ident in identities:
                try:
                    ident_audio = decode_audio(ident["audio_base64"], sample_rate)
                    emb = extract_speaker_embedding(ident_audio, sample_rate)
                    identity_embs.append((ident["name"], emb))
                    logger.info("Identity '%s' embedding extracted", ident["name"])
                except Exception as e:
                    logger.warning("Failed to process identity '%s': %s", ident.get("name", "?"), e)
                    identity_embs.append((ident.get("name", "?"), None))

            # For each track, find best-matching identity
            for t_idx, t_emb in enumerate(track_embs):
                if t_emb is None:
                    continue
                best_name = None
                best_sim = 0.0
                for i_name, i_emb in identity_embs:
                    if i_emb is None:
                        continue
                    sim = float(np.dot(t_emb, i_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = i_name
                if best_name and best_sim >= IDENTITY_MATCH_THRESHOLD:
                    identity_matches[str(t_idx)] = {
                        "name": best_name,
                        "similarity": round(best_sim, 4),
                    }
                    logger.info("Track %d matched identity '%s' (%.4f)",
                                t_idx, best_name, best_sim)

        except Exception as e:
            logger.warning("Identity matching failed: %s", e)

    # ── Encode audio and build response ──────────────────────
    result_tracks = []
    for track in tracks:
        result_tracks.append({
            "speaker": track["speaker"],
            "confidence": track["confidence"],
            "is_main": track["is_main"],
            "audio_base64": encode_audio(track["audio_np"], sample_rate),
            "transcription": track["transcription"],
        })

    main_count = sum(1 for t in result_tracks if t["is_main"])
    logger.info("Separated %d speakers (%d main voices, ElevenLabs enhanced)",
                num_speakers, main_count)
    result = {
        "separated_tracks": result_tracks,
        "num_speakers": num_speakers,
        "main_voices": main_count,
    }
    if identity_matches:
        result["identity_matches"] = identity_matches
    return result


# ── Start RunPod serverless worker ───────────────────────────
runpod.serverless.start({"handler": handler})
