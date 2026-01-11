#!/bin/bash

# Resume training from existing checkpoints (Single H100 GPU)
# 이 스크립트는 기존 checkpoint에서 학습을 재개합니다.
# 사전 조건: restore_checkpoints.sh로 checkpoint가 이미 복원되어 있어야 함
#
# 사용법:
#   1) 먼저 checkpoint 복원:
#      bash restore_checkpoints.sh run-20260102_165130-7whwtqoq
#   2) 학습 재개:
#      export WANDB_RUN_ID=7whwtqoq
#      bash resume_training_h100_single.sh
#
#   또는 screen 사용:
#      export WANDB_RUN_ID=7whwtqoq
#      screen -S resume bash resume_training_h100_single.sh

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/data/nanochat"

# -----------------------------------------------------------------------------
# 사전 조건 확인

if [ ! -d "$NANOCHAT_BASE_DIR" ]; then
    echo "❌ $NANOCHAT_BASE_DIR 디렉토리가 없습니다!"
    echo "먼저 restore_checkpoints.sh를 실행하세요."
    exit 1
fi

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "❌ Tokenizer가 없습니다: $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
    echo "먼저 restore_checkpoints.sh를 실행하세요."
    exit 1
fi

# WANDB_RUN_ID 확인 (resume에 필요)
if [ -z "${WANDB_RUN_ID:-}" ]; then
    echo "⚠️  WANDB_RUN_ID가 설정되지 않았습니다. 새 wandb run이 생성됩니다."
    echo "기존 run을 이어가려면: export WANDB_RUN_ID=<short_id>"
else
    echo "✅ Wandb run resume: $WANDB_RUN_ID"
fi

echo "✅ Tokenizer 확인됨"
echo "✅ Base directory: $NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
uv pip install nvidia-nvshmem-cu12
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Rust & rustbpe tokenizer

command -v cargo &> /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env
source ~/.cargo/env 2>/dev/null || true
maturin develop --manifest-path rustbpe/Cargo.toml --release

# -----------------------------------------------------------------------------
# Dataset download (skip if already exists)

DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
EXPECTED_SHARDS=240

existing_shards=$(find "$DATA_DIR" -name "shard_*.parquet" 2>/dev/null | wc -l)
if [ "$existing_shards" -ge "$EXPECTED_SHARDS" ]; then
    echo "✅ Dataset already complete: $existing_shards shards"
else
    echo "Downloading dataset (have $existing_shards/$EXPECTED_SHARDS shards)..."
    # Download first 8 shards for tokenizer (blocking)
    python -m nanochat.dataset download_fw --shards 0:8
    # Download remaining shards in background
    python -m nanochat.dataset download_fw --shards 0:240 &
    DOWNLOAD_PID=$!
fi

# -----------------------------------------------------------------------------
# Tokenizer - skip training (should already exist from restore)

TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
if [ -f "$TOKENIZER_FILE" ]; then
    echo "✅ Tokenizer already exists, skipping training..."
else
    echo "❌ Tokenizer not found! Run restore_checkpoints.sh first."
    exit 1
fi

# -----------------------------------------------------------------------------
# Base model pretraining (with auto-resume)

# Wait for dataset download if running
if [ -n "${DOWNLOAD_PID:-}" ]; then
    echo "Waiting for dataset download to complete..."
    wait $DOWNLOAD_PID
fi

# Auto-detect latest checkpoint for resume
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d20"
RESUME_ARG=""
if [ -d "$CKPT_DIR" ]; then
    LATEST_CKPT=$(ls -v "$CKPT_DIR"/model_*.pt 2>/dev/null | tail -n 1)
    if [ -n "$LATEST_CKPT" ]; then
        STEP=$(basename "$LATEST_CKPT" | sed 's/model_0*\([0-9]*\)\.pt/\1/')
        RESUME_ARG="--resume_from_step=$STEP"
        echo "✅ Resuming from checkpoint: step $STEP"
    fi
fi

if [ -z "$RESUME_ARG" ]; then
    echo "⚠️  No checkpoint found, starting from scratch"
fi

# Train base model (resume from checkpoint, model architecture loaded from checkpoint)
python -m scripts.base_train --run=$WANDB_RUN $RESUME_ARG --save_every=500
python -m scripts.base_loss
python -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining

# Download identity conversations dataset
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.mid_train --run=$WANDB_RUN
python -m scripts.chat_eval -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (SFT)

python -m scripts.chat_sft --run=$WANDB_RUN
python -m scripts.chat_eval -i sft

# -----------------------------------------------------------------------------
# Optional: CLI or Web chat (uncomment to use)
# python -m scripts.chat_cli -i sft
# python -m scripts.chat_web -i sft

# -----------------------------------------------------------------------------
# Generate final report

python -m nanochat.report generate

echo "=========================================="
echo "✅ Resume training completed!"
echo "=========================================="
