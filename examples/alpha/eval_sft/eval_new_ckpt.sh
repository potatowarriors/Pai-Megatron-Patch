#!/bin/bash
# eval_new_ckpt.sh — 새 체크포인트 하나를 변환부터 전 티어 벤치까지 자동으로 돌린다.
#
#   MG ckpt → (G1 내장 변환) → run_suite.sh (T1·T3·에이전틱·T2) → 판정·집계
#
# 변환은 GPU 를 쓰므로 **sub1 fleet 가 내려간 상태**에서 시작해야 한다. run_suite.sh 가
# 각 단계마다 fleet 를 갈아끼우므로, 이 스크립트는 변환 → run_suite 순서만 보장한다.
#
# 사용: bash eval_sft/eval_new_ckpt.sh <RUN_DIR> <ITER> [STAGES]
#   RUN_DIR : outputs/<sft_run>            (checkpoints/ 를 품은 디렉토리)
#   ITER    : 600                          (checkpoints/iter_0000600)
#   STAGES  : t1,t3,agentic,t2 (기본 전부)
# 환경변수: GPUS (기본 0~7; 변환은 그중 num_experts 약수 개수만 사용), SWE_W, TERM_W
set -uo pipefail
RUN_DIR="${1:?run dir (outputs/<sft_run>)}"; ITER="${2:?iteration (예: 600)}"
# 절대경로로 고정한다 — run_convert.sh 는 내부에서 cd 하므로 상대경로가 깨진다(2026-09-23 iter2300, KNOWN_ISSUES).
RUN_DIR="$(cd "$RUN_DIR" 2>/dev/null && pwd)" || { echo "[new-ckpt] ❌ run dir 없음: $1"; exit 1; }
STAGES="${3:-t1,t3,agentic,t2}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ALPHA="$(dirname "$HERE")"
REPO="$(dirname "$(dirname "$ALPHA")")"
ITERPAD=$(printf "%07d" "$ITER")
HFDIR="$RUN_DIR/hfmodel_$ITERPAD"
RUN_TAG="$(basename "$RUN_DIR")_iter$ITERPAD"

echo "[new-ckpt] run=$RUN_DIR iter=$ITER tag=$RUN_TAG stages=$STAGES"

# ---- 1) 변환 (이미 있으면 재사용) ----
# 재사용 판정은 **가중치 인덱스까지** 본다. config·generation_config 는 변환 초입에 먼저 쓰이므로 실패한 변환도
# 남긴다 — 2026-09-23 그 둘만 보고 재사용해 가중치 없는 디렉토리로 fleet 를 띄울 뻔했다.
if [ -f "$HFDIR/generation_config.json" ] && [ -f "$HFDIR/config.json" ] && [ -f "$HFDIR/model.safetensors.index.json" ]; then
  echo "[new-ckpt] HF 변환본 재사용: $HFDIR"
else
  echo "[new-ckpt] MG→HF 변환 시작 (G1 게이트 내장)"
  # fleet 가 GPU 를 물고 있으면 변환이 OOM 난다 — 먼저 내린다.
  bash "$HERE/stop_fleet.sh" "${GPUS:-0,1,2,3,4,5,6,7}" >/dev/null 2>&1 || true
  sleep 5
  # 변환 GPU 수(=EP)는 num_experts 의 약수여야 한다(192: 1·2·3·4·6·8). GPUS 가 8장이 아닌 목록이면
  # (2026-09-16 main1 GPU 7 제외 → 0~6 7장) 앞에서부터 가장 큰 약수 개수만 쓴다. run_convert.sh 의 GPUS 는
  # **개수**이고 그 nvidia-smi 자동검출은 CUDA_VISIBLE_DEVICES 를 무시하므로 여기서 둘 다 명시한다.
  IFS=',' read -ra GL <<< "${GPUS:-0,1,2,3,4,5,6,7}"
  NEXP=$(python3 "$ALPHA/tools/alpha_config.py" emit-megatron-flags --from-checkpoint "$RUN_DIR/checkpoints/iter_$ITERPAD" 2>/dev/null \
         | grep -A1 -x -e '--num-experts' | tail -1)
  NCONV=${#GL[@]}
  while [ "$NCONV" -gt 1 ] && [ $(( ${NEXP:-192} % NCONV )) -ne 0 ]; do NCONV=$((NCONV-1)); done
  CONV_CVD=$(IFS=','; echo "${GL[*]:0:$NCONV}")
  echo "[new-ckpt] 변환 GPU $NCONV 장 (CUDA_VISIBLE_DEVICES=$CONV_CVD, num_experts=${NEXP:-?})"
  if ! CUDA_VISIBLE_DEVICES="$CONV_CVD" GPUS="$NCONV" \
       bash "$REPO/toolkits/distributed_checkpoints_convertor/scripts/alpha/run_convert.sh" \
        baseline_48L "$RUN_DIR" "auto:$ITER" true true bf16; then
    # sub1 은 compat libcuda 570→595 스왑 이후 변환 **teardown** 에서 SIGSEGV 를 낸다
    # (`examples/alpha/CLAUDE.md` 함정 표 09-04). 두 선례(iter1200 .partial, iter1500·1800
    # exitcode -11) 모두 8랭크 전부가 프로그램 끝까지 도달했고 산출물은 정상이었다.
    # "종료코드 실패면 폐기" 도 "무시하고 진행" 도 틀렸다 — **산출물을 직접 잰다**.
    echo "[new-ckpt] ⚠️ 변환 종료코드 실패 — 산출물 검증으로 판정한다"
    if [ -d "$HFDIR" ] && python3 "$ALPHA/tools/verify_hf_export.py" "$HFDIR"; then
      echo "[new-ckpt] ✅ 검증 통과 — 진행 (G1·G2·G3 가 서빙에서 한 번 더 확인한다)"
    else
      echo "[new-ckpt] ❌ 변환 실패 — 중단"; exit 1
    fi
  fi
fi

# ---- 2) G1 재확인 (변환 경로를 안 탔을 수도 있으므로) ----
python3 "$ALPHA/tools/emit_generation_config.py" "$HFDIR" --check || {
  echo "[new-ckpt] ❌ G1 실패 — 벤치 중단"; exit 1; }

# ---- 3) 전 티어 ----
bash "$HERE/run_suite.sh" "$HFDIR" "$RUN_TAG" "$STAGES"
rc=$?
echo "[new-ckpt] 완료 (rc=$rc): $HERE/results/$RUN_TAG"
exit $rc
