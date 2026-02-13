# ============================================================
# svoice – RunPod Serverless Docker Image
# CUDA 11.8 · Python 3.10 · PyTorch 2.0
# ============================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/model/checkpoint.th

# ── System packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    libsndfile1 ffmpeg wget curl && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Pip upgrade ──────────────────────────────────────────────
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch + torchaudio (CUDA 11.8) ────────────────────────
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ── Python dependencies ─────────────────────────────────────
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    tqdm==4.66.1 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    runpod==1.6.2 \
    requests==2.31.0 \
    pydub==0.25.1 \
    speechbrain==1.0.0 \
    "huggingface_hub<0.24"

# ── Whisper for AI-powered voice scoring ────────────────────
RUN pip install --no-cache-dir "setuptools<71" && \
    pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

# ── Pre-download Whisper tiny model (avoids network on cold start) ──
RUN python -c "import whisper; whisper.load_model('tiny')"

# ── Pre-download ECAPA-TDNN model (avoids cold-start download) ──
RUN mkdir -p /app/models/ecapa && \
    python -c "from speechbrain.inference.speaker import EncoderClassifier; EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='/app/models/ecapa')"

# ── Copy application code ───────────────────────────────────
WORKDIR /app
COPY svoice/ /app/svoice/
COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

# ── Bake in the model checkpoint ─────────────────────────────
COPY model/checkpoint.th /app/model/checkpoint.th

# ── Entrypoint ───────────────────────────────────────────────
CMD ["python", "/app/handler.py"]
