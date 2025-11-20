#!/bin/bash
#==============================================================================
# WANDB 환경 설정 스크립트
#==============================================================================
#
# 사용법:
#   source ./scripts/setup_wandb.sh
#
# 주의: 반드시 source 명령으로 실행해야 환경 변수가 현재 쉘에 적용됩니다.
#==============================================================================

# WANDB API 키 설정
export WANDB_API_KEY="ffbe195db0033ed7b92ba142cfd01ca70b29974d"

# WANDB 모드 설정 (online/offline/disabled)
export WANDB_MODE="online"

# WANDB 디렉토리 설정 (선택사항)
# export WANDB_DIR="/path/to/wandb/logs"

# WANDB 캐시 디렉토리 설정 (선택사항)
# export WANDB_CACHE_DIR="/path/to/wandb/cache"

# 설정 확인
echo "✅ WANDB 환경 변수 설정 완료"
echo "  - WANDB_API_KEY: ${WANDB_API_KEY:0:8}... (마스킹됨)"
echo "  - WANDB_MODE: ${WANDB_MODE}"

# WANDB 설치 확인
if python -c "import wandb" 2>/dev/null; then
    WANDB_VERSION=$(python -c "import wandb; print(wandb.__version__)")
    echo "  - WANDB 버전: ${WANDB_VERSION}"
else
    echo "  ⚠️  경고: wandb 패키지가 설치되지 않았습니다."
    echo "     설치 방법: pip install wandb"
fi

# WANDB 로그인 테스트 (선택사항)
# wandb login --relogin
