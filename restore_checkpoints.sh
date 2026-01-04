#!/bin/bash

# Azure Blob Storage 체크포인트 복원 스크립트
# Azure Managed Identity 사용 (자격 증명 불필요)
# Run ID별 폴더에서 체크포인트 다운로드

STORAGE_ACCOUNT_NAME="mlenvblob"
CONTAINER_NAME="nanochat"
DEST_DIR="/data/nanochat"
WORKSPACE_DIR="/home/azureuser/workspace/nanochat"
LOG_DIR="${WORKSPACE_DIR}/logs"
LOG_FILE="${LOG_DIR}/restore_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

mkdir -p "$LOG_DIR"
log "=========================================="
log "체크포인트 복원 시작"
log "=========================================="

# Run ID 결정 (우선순위: CLI 인자 > 환경변수 > 에러)
if [ -n "$1" ]; then
    FOLDER_NAME="$1"
    log "Folder name from CLI: $FOLDER_NAME"
elif [ -n "$WANDB_RUN_ID" ]; then
    FOLDER_NAME="$WANDB_RUN_ID"
    log "Folder name from env: $FOLDER_NAME"
else
    log "❌ Blob 폴더명이 필요합니다!"
    log "사용법: bash restore_checkpoints.sh <folder_name>"
    log "  예: bash restore_checkpoints.sh run-20260104_064217-ixrmbcgf_20260104_103000"
    log "  또는: export WANDB_RUN_ID=run-20260104_064217-ixrmbcgf_20260104_103000"
    exit 1
fi

# 목적지 디렉토리 생성
log "목적지 디렉토리 준비: $DEST_DIR"
sudo mkdir -p "$DEST_DIR"
sudo chown -R "$(whoami):$(whoami)" "$DEST_DIR"

# 서브디렉토리 미리 생성 (azcopy race condition 방지)
mkdir -p "$DEST_DIR"/{base_checkpoints/d20,base_data,eval_bundle,mid_checkpoints,tokenized_data,report,tokenizer}

# azcopy 확인
if ! command -v azcopy >/dev/null 2>&1; then
    log "❌ azcopy가 설치되어 있지 않습니다!"
    exit 1
fi

# Azure Managed Identity로 인증
export AZCOPY_AUTO_LOGIN_TYPE=MSI
SRC_URL="https://${STORAGE_ACCOUNT_NAME}.blob.core.windows.net/${CONTAINER_NAME}/${FOLDER_NAME}"

log "소스: ${CONTAINER_NAME}/${FOLDER_NAME}/"
log "목적지: $DEST_DIR"
log "azcopy copy 시작..."
START_TIME=$(date +%s)

azcopy copy "$SRC_URL/*" "$DEST_DIR" \
    --recursive=true \
    --overwrite=ifSourceNewer \
    --log-level=INFO \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $EXIT_CODE -eq 0 ]; then
    log "✅ 복원 성공! (소요 시간: ${DURATION}초)"
    
    # 복원된 파일 요약
    CHECKPOINT_COUNT=$(find "$DEST_DIR" -type f -name "*.pt" 2>/dev/null | wc -l)
    TOKENIZER_EXISTS=$([ -f "$DEST_DIR/tokenizer/tokenizer.pkl" ] && echo "Yes" || echo "No")
    DISK_USAGE=$(du -sh "$DEST_DIR" 2>/dev/null | cut -f1)
    
    log "복원된 .pt 파일 수: $CHECKPOINT_COUNT"
    log "Tokenizer 존재: $TOKENIZER_EXISTS"
    log "총 데이터 크기: $DISK_USAGE"
else
    log "❌ 복원 실패! (종료 코드: $EXIT_CODE)"
    exit 1
fi

log "=========================================="
log "복원 완료"
log "=========================================="
