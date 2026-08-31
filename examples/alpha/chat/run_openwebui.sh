#!/bin/bash
# run_openwebui.sh — alpha 채팅 UI. 백엔드는 chat/serve_chat.sh 가 띄운 vLLM.
#
# 설계 결정:
#   - DATA_DIR 은 NFS. 컨테이너 재생성에도 대화 이력이 남는다 (/tmp 는 휘발).
#   - WEBUI_AUTH=false : 1인 사용 확정(사용자, 2026-08-31). 여럿이 쓰게 되면 반드시 켠다.
#   - 포트 8080 = BACKENDAI_SERVICE_PORTS 의 nniboard preopen 슬롯. Backend.AI 앱 프록시로
#     노출되기를 노린 선택. 안 되면 ssh -L 8080:localhost:8080 <세션> 로 터널.
#   - RAG/이미지생성 전부 off. 우리는 소재 모델과 대화만 한다 — 임베딩 모델 다운로드로
#     기동이 느려지거나 오프라인에서 실패하는 것을 막는다.
#
# 사용: bash chat/run_openwebui.sh [PORT] [VLLM_BASE_URL]
set -euo pipefail
PORT="${1:-8080}"
VLLM_URL="${2:-http://localhost:8001/v1}"
VENV=/home/work/vidsearch/tools/openwebui_venv

export PYTHONNOUSERSITE=1          # user-site(~/.local) 누수 차단 — venv 격리 원칙
export DATA_DIR=/home/work/vidsearch/tools/openwebui_data
mkdir -p "$DATA_DIR"

# 시크릿 키를 DATA_DIR 에 둔다. 안 주면 open-webui 가 **현재 작업 디렉토리**에
# .webui_secret_key 를 떨궈 리포를 오염시킨다 (2026-08-31 실측).
[ -f "$DATA_DIR/secret_key" ] || (umask 077; head -c 48 /dev/urandom | base64 | tr -d '\n' > "$DATA_DIR/secret_key")
export WEBUI_SECRET_KEY="$(cat "$DATA_DIR/secret_key")"

# 스키마 초기화를 분리한다 — 0.11.2 는 config 로드 중 마이그레이션을 돌리다 순환 import 로
# 실패하고 그 예외를 삼킨다. 사유·구조는 init_openwebui_db.py 상단. 멱등.
export ENABLE_DB_MIGRATIONS=false
"$VENV/bin/python" "$(dirname "$0")/init_openwebui_db.py" || { echo "[webui] 스키마 초기화 실패 — 중단"; exit 1; }

# 백엔드 연결 (vLLM 은 OpenAI 호환. 키는 검사하지 않지만 비우면 클라이언트가 거부한다)
export OPENAI_API_BASE_URL="$VLLM_URL"
export OPENAI_API_KEY=dummy
export ENABLE_OLLAMA_API=false

# 1인 사용 — 로그인 화면 없음
export WEBUI_AUTH=false
export WEBUI_NAME="Alpha v2"

# 불필요 서브시스템 off (기동 시 모델 다운로드 방지)
export RAG_EMBEDDING_ENGINE=""
export ENABLE_RAG_WEB_SEARCH=false
export ENABLE_IMAGE_GENERATION=false
export AUDIO_STT_ENGINE=""
export ENABLE_OPENAI_API=true
export HF_HUB_OFFLINE=1
export SCARF_NO_ANALYTICS=true
export DO_NOT_TRACK=true
export ANONYMIZED_TELEMETRY=false

echo "[webui] backend=$VLLM_URL  port=$PORT  data=$DATA_DIR"
exec $VENV/bin/open-webui serve --host 0.0.0.0 --port "$PORT"
