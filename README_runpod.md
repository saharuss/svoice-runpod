# svoice – RunPod Serverless Deployment

Deploy the [svoice](https://github.com/facebookresearch/svoice) speaker separation model as a RunPod serverless GPU endpoint.

## Quick Start

### 1. Get a Model Checkpoint

You need a trained svoice model. Either:
- **Train one** using the svoice training pipeline (see main README)
- **Use a pre-trained checkpoint** if available

Place the checkpoint at `model/checkpoint.th` relative to this directory:

```bash
mkdir -p model
cp /path/to/your/checkpoint.th model/checkpoint.th
```

### 2. Build the Docker Image

```bash
docker build -t svoice-runpod .
```

### 3. Push to Docker Hub

```bash
docker tag svoice-runpod your-dockerhub-user/svoice-runpod:latest
docker push your-dockerhub-user/svoice-runpod:latest
```

### 4. Create a RunPod Serverless Endpoint

1. Go to [RunPod Console → Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Set **Container Image** to `your-dockerhub-user/svoice-runpod:latest`
4. Set **GPU Type** (any NVIDIA GPU works, e.g. RTX 3090, A100)
5. Set environment variable `MODEL_PATH=/app/model/checkpoint.th` (or wherever your checkpoint is)
6. Deploy

### 5. Send Requests

#### Via base64 audio:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_base64": "'$(base64 -i mixed_audio.wav)'",
      "sample_rate": 8000
    }
  }'
```

#### Via audio URL:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/mixed_speakers.wav",
      "sample_rate": 8000
    }
  }'
```

### Response Format

```json
{
  "separated_tracks": [
    { "speaker": 1, "audio_base64": "<base64 WAV>" },
    { "speaker": 2, "audio_base64": "<base64 WAV>" }
  ],
  "num_speakers": 2
}
```

## Using a Network Volume (recommended for large models)

Instead of baking the model into the Docker image, use RunPod Network Volumes:

1. Create a Network Volume in RunPod console
2. Upload your `checkpoint.th` to the volume
3. Set `MODEL_PATH=/runpod-volume/checkpoint.th` in the endpoint env vars

## Local Testing

```bash
# Without a model (will warn but won't crash):
docker run --rm svoice-runpod

# With a model:
docker run --rm --gpus all \
  -v $(pwd)/model:/app/model \
  svoice-runpod
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/model/checkpoint.th` | Path to the svoice model checkpoint |

## Project Structure

```
svoice/
├── Dockerfile          # GPU-enabled container definition
├── handler.py          # RunPod serverless handler
├── test_input.json     # Local test payload
├── .dockerignore       # Build context exclusions
├── model/              # Place checkpoint.th here
│   └── checkpoint.th
└── svoice/             # Original svoice source code
    ├── models/
    ├── data/
    ├── separate.py
    └── ...
```
