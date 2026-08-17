#!/bin/bash
#==============================================================================
# WANDB 환경 설정 스크립트 — env/키파일 참조 방식 (키 하드코딩 금지)
#
# 사용법:
#   source ./scripts/setup_wandb.sh     (train.sh / run_benchmarks*.sh가 자동 소싱)
#
# WANDB_API_KEY 해석 순서 (첫 매치 사용):
#   1. 이미 export된 WANDB_API_KEY              (그대로 존중)
#   2. $WANDB_KEY_FILE 이 가리키는 파일
#   3. <이 스크립트 디렉토리>/.wandb_key        (gitignored; chmod 600)
#   4. ~/.wandb_key
# 아무것도 없으면 경고만 남기고 반환 — train.sh 배너에 "off (no API key)"로 표시.
#
# 키 교체(rotate): 새 키를 .wandb_key 파일 한 줄로 덮어쓰면 끝. 코드 무변경.
#   printf '%s' '<NEW_KEY>' > examples/alpha/scripts/.wandb_key
#   chmod 600 examples/alpha/scripts/.wandb_key
#
# 이 파일(또는 다른 트래킹 파일)에 키를 하드코딩하지 말 것 — 2026-08-18 유출
# 정리의 원인이 하드코딩이었다 (docs/WANDB.md + git 히스토리 855cad1~; 해당 구
# 키는 rotate 대상). 이 스크립트는 train.sh의 `set -euo pipefail` 아래에서
# 소싱되므로 모든 변수 참조는 nounset-safe 해야 한다.
#==============================================================================

_WANDB_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
_WANDB_KEY_SRC=""

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    _WANDB_KEY_SRC="environment"
else
    for _kf in "${WANDB_KEY_FILE:-}" "$_WANDB_SETUP_DIR/.wandb_key" "${HOME:-}/.wandb_key"; do
        if [[ -n "$_kf" && -r "$_kf" ]]; then
            WANDB_API_KEY="$(tr -d '[:space:]' < "$_kf")"
            if [[ -n "$WANDB_API_KEY" ]]; then
                export WANDB_API_KEY
                _WANDB_KEY_SRC="$_kf"
                break
            fi
        fi
    done
    unset -v _kf
fi

# WANDB 모드 (online/offline/disabled) — 기존 값 존중
export WANDB_MODE="${WANDB_MODE:-online}"

# HuggingFace Hub 토큰 (데이터셋 다운로드 rate limit 방지) — env 참조만
export HF_TOKEN="${HF_TOKEN:-}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "✅ WANDB 환경 변수 설정 완료"
    echo "  - WANDB_API_KEY: ${WANDB_API_KEY:0:8}... (마스킹됨; 출처: ${_WANDB_KEY_SRC})"
    echo "  - WANDB_MODE: ${WANDB_MODE}"
else
    echo "⚠️  WANDB_API_KEY 미설정 — wandb 로깅 비활성 (train.sh 배너: off)."
    echo "   키 파일 생성: printf '%s' '<KEY>' > ${_WANDB_SETUP_DIR}/.wandb_key && chmod 600 ${_WANDB_SETUP_DIR}/.wandb_key"
fi

unset -v _WANDB_SETUP_DIR _WANDB_KEY_SRC
