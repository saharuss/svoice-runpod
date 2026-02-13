#!/bin/bash
set -e
LOG=~/upgrade.log

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

log "=== SVOICE TRAINING UPGRADE ==="
log "Phase 1: Download WHAM! noise dataset"
log "Phase 2: Generate 10-speaker dataset with WHAM! noise"
log "Phase 3: Download domain-specific data (podcasts/meetings)"
log ""

# ── Phase 1: WHAM! ──
cd ~
if [ ! -d ~/wham_noise/wham_noise ]; then
    log "Downloading WHAM! noise (17GB)..."
    wget -c -q --show-progress "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip" \
        -O ~/wham_noise.zip 2>&1 | tee -a $LOG
    log "Unzipping WHAM!..."
    mkdir -p ~/wham_noise
    unzip -o ~/wham_noise.zip -d ~/wham_noise >> $LOG 2>&1
    rm -f ~/wham_noise.zip
    log "WHAM! ready at ~/wham_noise/"
else
    log "WHAM! already downloaded, skipping"
fi

# Find the noise wav directory
NOISE_DIR=$(find ~/wham_noise -name "*.wav" -print -quit | xargs dirname 2>/dev/null || echo "")
if [ -z "$NOISE_DIR" ]; then
    NOISE_DIR=$(find ~/wham_noise -type d -name "tr" -print -quit 2>/dev/null || echo ~/wham_noise)
fi
log "Noise directory: $NOISE_DIR"

# ── Phase 2: Generate 10-speaker mixed dataset ──
cd ~/svoice_demo
log "Cleaning old dataset..."
rm -rf egs/dataset/tr egs/dataset/vl egs/dataset/ts
mkdir -p egs/dataset

log "Generating TRAINING set (20000 scenes, 10 speakers, WHAM! noise)..."
python3 scripts/make_dataset.py \
    --in_path ~/librimix_storage/LibriSpeech/train-clean-360 \
    --out_path egs/dataset/tr \
    --noise_path "$NOISE_DIR" \
    --num_of_speakers 10 \
    --num_of_scenes 20000 \
    --sec 4 --sr 16000 2>&1 | tee -a $LOG

log "Generating VALIDATION set (3000 scenes, 10 speakers, WHAM! noise)..."
python3 scripts/make_dataset.py \
    --in_path ~/librimix_storage/LibriSpeech/dev-clean \
    --out_path egs/dataset/vl \
    --noise_path "$NOISE_DIR" \
    --num_of_speakers 10 \
    --num_of_scenes 3000 \
    --sec 4 --sr 16000 2>&1 | tee -a $LOG

log "Generating TEST set (3000 scenes, 10 speakers, WHAM! noise)..."
python3 scripts/make_dataset.py \
    --in_path ~/librimix_storage/LibriSpeech/test-clean \
    --out_path egs/dataset/ts \
    --noise_path "$NOISE_DIR" \
    --num_of_speakers 10 \
    --num_of_scenes 3000 \
    --sec 4 --sr 16000 2>&1 | tee -a $LOG

log "Dataset generation COMPLETE!"

# ── Phase 3: Download domain-specific data ──
log "Downloading domain-specific audio (podcasts, meetings)..."
mkdir -p ~/domain_data/podcasts ~/domain_data/meetings

# AMI Meeting Corpus (free, widely used for meeting separation research)
log "Downloading AMI Meeting Corpus headset mixes..."
if [ ! -d ~/domain_data/meetings/ami ]; then
    mkdir -p ~/domain_data/meetings/ami
    # Download a curated subset of AMI headset mix recordings
    wget -c -q "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav" \
        -O ~/domain_data/meetings/ami/ES2002a.wav 2>> $LOG || log "AMI sample 1 failed (non-critical)"
    wget -c -q "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002b/audio/ES2002b.Mix-Headset.wav" \
        -O ~/domain_data/meetings/ami/ES2002b.wav 2>> $LOG || log "AMI sample 2 failed (non-critical)"
    wget -c -q "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2003a/audio/ES2003a.Mix-Headset.wav" \
        -O ~/domain_data/meetings/ami/ES2003a.wav 2>> $LOG || log "AMI sample 3 failed (non-critical)"
    wget -c -q "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2003b/audio/ES2003b.Mix-Headset.wav" \
        -O ~/domain_data/meetings/ami/ES2003b.wav 2>> $LOG || log "AMI sample 4 failed (non-critical)"
    log "AMI samples downloaded"
else
    log "AMI data exists, skipping"
fi

# LibriSpeech dev-other for harder separation scenarios
log "Downloading LibriSpeech dev-other (more challenging speakers)..."
if [ ! -d ~/librimix_storage/LibriSpeech/dev-other ]; then
    wget -c -q "https://www.openslr.org/resources/12/dev-other.tar.gz" \
        -O ~/librimix_storage/dev-other.tar.gz 2>> $LOG
    tar xzf ~/librimix_storage/dev-other.tar.gz -C ~/librimix_storage/ 2>> $LOG
    rm -f ~/librimix_storage/dev-other.tar.gz
    log "dev-other extracted"
else
    log "dev-other exists, skipping"
fi

log ""
log "=== ALL PHASES COMPLETE ==="
log "Dataset: egs/dataset/{tr,vl,ts}"
log "WHAM noise: $NOISE_DIR"
log "Domain data: ~/domain_data/"
log "Ready to start training!"
