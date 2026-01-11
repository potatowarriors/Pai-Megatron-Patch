#!/bin/bash
#
# Alpha Checkpoint Backup Wrapper Script
# =======================================
# For use with cron jobs or manual execution
#
# Usage:
#   ./checkpoint_backup.sh [OPTIONS]
#
# Options:
#   --background, -b    Run in background (nohup)
#   --status            Check if backup is running
#   --dry-run           Simulate without uploading
#   --force             Re-backup all checkpoints
#   --list              Show backup state
#
# Environment Variables (set in .env file or export):
#   MINIO_ENDPOINT
#   MINIO_ACCESS_KEY
#   MINIO_SECRET_KEY

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "$0")"
LOG_DIR="${SCRIPT_DIR}/logs"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
PID_FILE="${SCRIPT_DIR}/state/backup.pid"

# Create directories if needed
mkdir -p "${LOG_DIR}"
mkdir -p "${SCRIPT_DIR}/state"

# Parse arguments for --background flag
BACKGROUND=false
SHOW_STATUS=false
PASSTHROUGH_ARGS=()

for arg in "$@"; do
    case $arg in
        --background|-b)
            BACKGROUND=true
            ;;
        --status)
            SHOW_STATUS=true
            ;;
        *)
            PASSTHROUGH_ARGS+=("$arg")
            ;;
    esac
done

# Handle --status: check if backup is running
if [ "$SHOW_STATUS" = true ]; then
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Backup is RUNNING (PID: $PID)"
            echo ""
            echo "Log file:"
            LATEST_LOG=$(ls -t "${LOG_DIR}"/backup_*.log 2>/dev/null | head -1)
            if [ -n "$LATEST_LOG" ]; then
                echo "  $LATEST_LOG"
                echo ""
                echo "Last 10 lines:"
                tail -10 "$LATEST_LOG"
            fi
            exit 0
        else
            echo "Backup is NOT running (stale PID file found)"
            rm -f "$PID_FILE"
            exit 0
        fi
    else
        echo "Backup is NOT running"
        exit 0
    fi
fi

# Handle --background: re-exec self with nohup
if [ "$BACKGROUND" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BG_LOG_FILE="${LOG_DIR}/backup_${TIMESTAMP}.log"

    echo "Starting backup in background..."
    echo "Log file: $BG_LOG_FILE"
    echo ""

    # Re-execute without --background flag (use absolute path)
    nohup "$SCRIPT_PATH" "${PASSTHROUGH_ARGS[@]}" > "$BG_LOG_FILE" 2>&1 &
    BG_PID=$!
    echo $BG_PID > "$PID_FILE"

    echo "Background PID: $BG_PID"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f $BG_LOG_FILE"
    echo ""
    echo "To check status:"
    echo "  ./checkpoint_backup.sh --status"
    echo ""
    echo "To stop:"
    echo "  kill $BG_PID"

    exit 0
fi

# === Main execution (foreground) ===

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/backup_${TIMESTAMP}.log"

# Save PID for status checking
echo $$ > "$PID_FILE"

# Cleanup PID file on exit
cleanup() {
    rm -f "$PID_FILE"
}
trap cleanup EXIT

# Source environment variables if env file exists
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from ${ENV_FILE}" | tee -a "$LOG_FILE"
    set -a  # Auto-export all variables
    source "$ENV_FILE"
    set +a
fi

# Validate required environment variables
if [ -z "$MINIO_ENDPOINT" ] || [ -z "$MINIO_ACCESS_KEY" ] || [ -z "$MINIO_SECRET_KEY" ]; then
    echo "ERROR: Missing required environment variables" | tee -a "$LOG_FILE"
    echo "Required: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Either:" | tee -a "$LOG_FILE"
    echo "  1. Create a .env file: cp .env.example .env && edit .env" | tee -a "$LOG_FILE"
    echo "  2. Export variables: export MINIO_ENDPOINT=... MINIO_ACCESS_KEY=... MINIO_SECRET_KEY=..." | tee -a "$LOG_FILE"
    exit 1
fi

echo "=== Alpha Checkpoint Backup ===" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "MinIO Endpoint: ${MINIO_ENDPOINT}" | tee -a "$LOG_FILE"
echo "PID: $$" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the backup script
cd "${SCRIPT_DIR}"
python3 checkpoint_backup.py --config "$CONFIG_FILE" "${PASSTHROUGH_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: ${EXIT_CODE}" | tee -a "$LOG_FILE"

# Clean up old logs (keep last 30 days)
find "${LOG_DIR}" -name "backup_*.log" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE
