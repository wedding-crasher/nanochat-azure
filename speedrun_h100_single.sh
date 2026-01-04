#!/bin/bash

# This script is adapted for SINGLE GPU training from speedrun.sh
# It will take ~8x longer than the original 8XH100 script (approx 32 hours on 1XH100).

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/data/nanochat"

# Clean up previous run data
if [ -d "$NANOCHAT_BASE_DIR" ]; then
    echo "Cleaning up previous run data at $NANOCHAT_BASE_DIR..."
    rm -rf "$NANOCHAT_BASE_DIR"
fi

# Create dir with sudo if needed (Azure VM /data is often root-owned)
sudo mkdir -p "$NANOCHAT_BASE_DIR"
sudo chown -R "$(whoami):$(whoami)" "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# install NVSHMEM for PyTorch CUDA support
uv pip install nvidia-nvshmem-cu12
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!

# Train tokenizer only if not already present (resume case: skip to avoid overwriting)
TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
if [ -f "$TOKENIZER_FILE" ]; then
    echo "Tokenizer already exists at $TOKENIZER_FILE, skipping training..."
else
    # train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
    python -m scripts.tok_train --max_chars=2000000000
    # evaluate the tokenizer (report compression ratio etc.)
    python -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Single GPU - no torchrun needed
# The code will automatically use gradient accumulation to match the same effective batch size

# pretrain the d20 model
BASE_DEPTH=20

# Auto-detect latest checkpoint and resume if exists
RESUME_ARG=""
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d${BASE_DEPTH}"
if [ -d "$CKPT_DIR" ]; then
    LAST_MODEL=$(ls -1 "$CKPT_DIR"/model_*.pt 2>/dev/null | sort | tail -n 1 || true)
    if [ -n "$LAST_MODEL" ]; then
        BASENAME=$(basename "$LAST_MODEL")
        STEP_STR=${BASENAME#model_}
        STEP_STR=${STEP_STR%.pt}
        RESUME_FROM_STEP=$(echo "$STEP_STR" | sed 's/^0*//')
        RESUME_FROM_STEP=${RESUME_FROM_STEP:-0}
        echo "Resuming base_train from step: $RESUME_FROM_STEP ($LAST_MODEL)"
        RESUME_ARG="--resume_from_step=$RESUME_FROM_STEP"
    fi
fi

python -m scripts.base_train --depth=$BASE_DEPTH --run=$WANDB_RUN --save_every=500 $RESUME_ARG
# evaluate the model on a larger chunk of train/val data and draw some samples
python -m scripts.base_loss
# evaluate the model on CORE tasks
python -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
python -m scripts.mid_train --run=$WANDB_RUN
python -m scripts.chat_eval -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
python -m scripts.chat_sft --run=$WANDB_RUN
python -m scripts.chat_eval -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# python -m scripts.chat_rl --run=$WANDB_RUN
# eval the RL model only on GSM8K
# python -m scripts.chat_eval -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
