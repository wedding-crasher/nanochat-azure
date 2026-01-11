#!/bin/bash

# SFT from midtrain checkpoint (single GPU)
export NANOCHAT_BASE_DIR="/data/nanochat"

# Activate venv if needed
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run SFT from mid checkpoint
python -m scripts.chat_sft --source=mid --run=${WANDB_RUN_ID:-dummy}
