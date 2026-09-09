#!/bin/bash
# run_librechat.sh — alpha 채팅 UI (LibreChat). 백엔드는 chat/serve_chat.sh 가 띄운 vLLM.
#
# OpenWebUI 를 대체한 이유 (2026-09-09, docs/KNOWN_ISSUES.md 2026-09-09):
#   ① 라이선스 — Open WebUI License 는 사용자 50명 초과 시 브랜딩 변경을 금지한다. LibreChat 은 MIT.
#   ② OpenWebUI 0.11 은 UI 발 요청마다 내장 도구 25종을 `tools` 로 주입한다. alpha 템플릿이 tool
#      시나리오로 분기해 "안녕?" 의 프롬프트가 17 → 5,440 토큰이 되고 모델이 도구를 부르거나 영어로
#      답했다. LibreChat 은 model / stream / messages 만 보낸다 (PoC 요청 로그로 확인).
#
# 설계 결정:
#   - 설치본 LC_HOME: NFS. 소스 LibreChat @8da4ae7 (v0.8.8-rc2, 2026-09-09) + 빌드 산출물. 재빌드는 README §7.
#   - 데이터 LC_DATA: NFS. MongoDB dbpath · 로그 · 시크릿. 컨테이너 재생성에도 대화가 남는다.
#   - MongoDB 8.0.16 공식 tarball 바이너리. docker 없이 --fork 데몬, 127.0.0.1 전용. 이미 떠 있으면 재사용.
#   - 설정 정본은 이 스크립트 + 같은 디렉토리의 librechat.yaml. .env 는 매 기동마다 다시 쓴다.
#   - 시크릿(JWT/CREDS)은 첫 실행에 만들어 LC_DATA/secrets 에 둔다. 바꾸면 기존 로그인 세션이 무효.
#   - 로그인 필수 — LibreChat 에는 auth-off 가 없다. 첫 사용자는 UI 회원가입 (ALLOW_REGISTRATION=true).
#     포트에 닿는 누구나 가입할 수 있다는 점은 OpenWebUI 의 WEBUI_AUTH=false 와 같은 노출 수준이다.
#   - 포트 8080 = BACKENDAI_SERVICE_PORTS 의 nniboard preopen 슬롯 (OpenWebUI 에서 이관).
#     Backend.AI 앱 프록시로 안 열리면 ssh -N -L 8080:localhost:8080 main1.
#
# 사용: bash chat/run_librechat.sh [PORT] [VLLM_BASE_URL]
#   로그는 stdout. 데몬화는 호출 측에서: nohup bash chat/run_librechat.sh > /home/work/vidsearch/tools/chat_logs/librechat_8080.log 2>&1 &
set -euo pipefail
PORT="${1:-8080}"
VLLM_URL="${2:-http://localhost:8001/v1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LC_HOME="${LC_HOME:-/home/work/vidsearch/tools/librechat}"
LC_DATA="${LC_DATA:-/home/work/vidsearch/tools/librechat_data}"
MONGO_BIN="${MONGO_BIN:-/home/work/vidsearch/tools/mongodb-8.0.16/bin/mongod}"
MONGO_PORT="${MONGO_PORT:-27017}"
mkdir -p "$LC_DATA/mongo" "$LC_DATA/logs"

[ -f "$LC_HOME/api/server/index.js" ] || { echo "[librechat] 설치본 없음: $LC_HOME (README §7 설치 절차)"; exit 1; }
[ -x "$MONGO_BIN" ] || { echo "[librechat] mongod 없음: $MONGO_BIN (README §7)"; exit 1; }
[ -f "$HERE/librechat.yaml" ] || { echo "[librechat] 설정 없음: $HERE/librechat.yaml"; exit 1; }

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# 1. MongoDB — 떠 있으면 재사용, 아니면 데몬으로 기동
if port_open "$MONGO_PORT"; then
  echo "[librechat] mongod 재사용 (:$MONGO_PORT)"
else
  "$MONGO_BIN" --dbpath "$LC_DATA/mongo" --logpath "$LC_DATA/logs/mongod.log" --logappend \
    --bind_ip 127.0.0.1 --port "$MONGO_PORT" --fork >/dev/null
  for _ in $(seq 1 30); do port_open "$MONGO_PORT" && break; sleep 1; done
  port_open "$MONGO_PORT" || { echo "[librechat] mongod 기동 실패 — $LC_DATA/logs/mongod.log"; exit 1; }
  echo "[librechat] mongod 기동 (:$MONGO_PORT, dbpath=$LC_DATA/mongo)"
fi

# 2. 시크릿 — 첫 실행에 생성, 이후 재사용
SEC="$LC_DATA/secrets"
if [ ! -f "$SEC" ]; then
  hex() { head -c "$1" /dev/urandom | od -An -tx1 | tr -d ' \n'; }
  (umask 077; printf 'JWT_SECRET=%s\nJWT_REFRESH_SECRET=%s\nCREDS_KEY=%s\nCREDS_IV=%s\n' \
     "$(hex 32)" "$(hex 32)" "$(hex 32)" "$(hex 16)" > "$SEC")
  echo "[librechat] 시크릿 생성: $SEC"
fi

# 3. .env — 정본은 이 스크립트. 손으로 고치지 말 것.
{
  cat <<ENV
HOST=0.0.0.0
PORT=$PORT
MONGO_URI=mongodb://127.0.0.1:$MONGO_PORT/LibreChat
DOMAIN_CLIENT=http://localhost:$PORT
DOMAIN_SERVER=http://localhost:$PORT
CONFIG_PATH=$HERE/librechat.yaml
ALPHA_VLLM_URL=$VLLM_URL
APP_TITLE=Alpha v2
NO_INDEX=true
TRUST_PROXY=1
ENDPOINTS=custom
SEARCH=false
MEILI_NO_ANALYTICS=true
CHECK_BALANCE=false
ALLOW_EMAIL_LOGIN=true
ALLOW_REGISTRATION=true
ALLOW_SOCIAL_LOGIN=false
ALLOW_SOCIAL_REGISTRATION=false
ALLOW_PASSWORD_RESET=false
DEBUG_LOGGING=false
DEBUG_CONSOLE=false
ENV
  cat "$SEC"
} > "$LC_HOME/.env"

# 4. 웹검색 키 — examples/alpha/.env(gitignored) 의 TAVILY_API_KEY 만 뽑아 넘긴다 (source 하지 않음: 다른 줄의 형식 오류에 안 걸리게)
TAVILY_API_KEY="$(grep -m1 '^TAVILY_API_KEY=' "$HERE/../.env" 2>/dev/null | cut -d= -f2- | tr -d '"'"'"' \r')"
if [ -n "$TAVILY_API_KEY" ]; then
  echo "TAVILY_API_KEY=$TAVILY_API_KEY" >> "$LC_HOME/.env"
else
  echo "[librechat] 경고: TAVILY_API_KEY 없음 — 웹검색 토글이 비활성 (examples/alpha/.env 확인)"
fi

cd "$LC_HOME"
echo "[librechat] port=$PORT vllm=$VLLM_URL home=$LC_HOME data=$LC_DATA websearch=$([ -n "$TAVILY_API_KEY" ] && echo tavily || echo off)"
exec env NODE_ENV=production node api/server/index.js
