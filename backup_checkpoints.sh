#!/bin/bash

# Azure Blob Storage 체크포인트 백업 스크립트
# Azure Managed Identity 사용 (자격 증명 불필요)
# Run ID별로 폴더를 생성하여 증분 백업

STORAGE_ACCOUNT_NAME="mlenvblob"
CONTAINER_NAME="nanochat"
SOURCE_DIR="/data/nanochat"
WORKSPACE_DIR="/home/azureuser/workspace/nanochat"
LOG_DIR="${WORKSPACE_DIR}/logs"
LOG_FILE="${LOG_DIR}/backup_$(date +%Y%m%d).log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

mkdir -p "$LOG_DIR"
log "=========================================="
log "체크포인트 백업 시작"
log "=========================================="

# Run ID 추출 (우선순위: 환경변수 > wandb latest-run > 기본값)
if [ -n "$WANDB_RUN_ID" ]; then
    RUN_ID="$WANDB_RUN_ID"
    log "Run ID from env: $RUN_ID"
elif [ -L "${WORKSPACE_DIR}/wandb/latest-run" ]; then
    LATEST_RUN=$(readlink "${WORKSPACE_DIR}/wandb/latest-run")
    RUN_ID=$(basename "$LATEST_RUN")
    log "Run ID from wandb: $RUN_ID"
else
    RUN_ID="default"
    log "Run ID not found, using: $RUN_ID"
fi

# 타임스탬프 추가하여 매번 새 폴더 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FOLDER="${RUN_ID}_${TIMESTAMP}"
log "Backup folder: $BACKUP_FOLDER"

# 소스 확인
[ ! -d "$SOURCE_DIR" ] && { log "소스 디렉토리 없음"; exit 0; }

CHECKPOINT_COUNT=$(find "$SOURCE_DIR" -type f \( -name "*.pt" -o -name "*.json" \) 2>/dev/null | wc -l)
log "백업할 파일 수: $CHECKPOINT_COUNT"
[ "$CHECKPOINT_COUNT" -eq 0 ] && { log "백업할 파일 없음"; exit 0; }

DISK_USAGE=$(du -sh "$SOURCE_DIR" 2>/dev/null | cut -f1)
log "데이터 크기: $DISK_USAGE"

log "azcopy sync 시작..."
START_TIME=$(date +%s)

# Azure Managed Identity를 사용하여 인증
# 타임스탬프 포함된 새 폴더로 백업 (덮어쓰기 방지)
DEST_URL="https://${STORAGE_ACCOUNT_NAME}.blob.core.windows.net/${CONTAINER_NAME}/${BACKUP_FOLDER}"
export AZCOPY_AUTO_LOGIN_TYPE=MSI
log "Destination: ${CONTAINER_NAME}/${BACKUP_FOLDER}/"
azcopy copy "$SOURCE_DIR/*" "$DEST_URL" \
    --recursive=true \
    --log-level=INFO \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $EXIT_CODE -eq 0 ]; then
    log "✅ 백업 성공! (소요 시간: ${DURATION}초)"
else
    log "❌ 백업 실패! (종료 코드: $EXIT_CODE)"
    exit 1
fi

find "$LOG_DIR" -name "backup_*.log" -mtime +30 -delete 2>/dev/null
log "=========================================="
log "백업 완료"
log "=========================================="
