#!/usr/bin/env bash
# alpha-SFT-KoChat-v1 (ko_chat 합성 1차 트랜치) 64k 변환 — convert_sft_64k.sh 의 run() 미러.
#
# 3세트 (플래그가 파일 단위라 trackA 를 method 별로 분리해 둠, cut_tranche.py 산출):
#   kochat_if_fanout_me : trackA_if.jsonl   — chat_v3 IF 번역분. 원본이 GPT-OSS medium-effort
#                         생성분이고 61% multi-True → chat_v3_if_fanout_me 와 동일 플래그
#   kochat_chat         : trackA_chat.jsonl — chat_v3 chat 재생성분(last-only, 자체 reasoning)
#   kochat_b            : trackB.jsonl      — 네이티브 생성(전 assistant 턴 reasoning)
# NFS 주의: 순차 1개. 멱등: idx 존재 시 스킵.
# 실행 (sub1): NCORES=96 nohup bash convert_kochat_64k.sh > /path/log 2>&1 &
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# SFT_DIR: 컷 산출 디렉토리 (1차 alpha-SFT-KoChat-v1, 2차 alpha-SFT-KoChat-v2 …)
# SUFFIX: 산출 세트명 접미사 (2차는 "_t2" — 1차 세트 보존, 블렌드에서 교체)
SFT=${SFT_DIR:-/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-KoChat-v1}
SUFFIX=${SUFFIX:-}
OUT=${OUT:-/home/work/Datasets/LL_preprocessed/v5/sft_packed_64k_pad16}
NCORES=${NCORES:-96}
TOK=$REPO/examples/alpha/tokenizer_v5
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO:$REPO/backends/megatron/Megatron-LM-251125${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"

run() {
  local name=$1 input=$2
  shift 2
  if [ -f "$OUT/$name/data_text_document.idx" ]; then
    echo "== $name: SKIP (idx 존재)"; return
  fi
  echo "== $name: $input  ($(date +%H:%M:%S))"
  mkdir -p "$OUT/$name"
  nice -n 10 python "$REPO/toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py" \
    --input "$input" --tokenizer "$TOK" \
    --output-prefix "$OUT/$name/data" \
    --seq-length 65536 --pad-doc-multiple 16 --workers "$NCORES" "$@" \
    || echo "!! $name FAILED (exit $?)"
}

# trackB: 전 assistant 턴 학습(멀티턴 43.7%가 multi-True) → fan-out 필수
# (INTERLEAVED_THINKING §7 규칙 1). medium-effort 는 GPT-OSS 산출이 아니라 미적용.
run "kochat_b_fanout$SUFFIX"     "$SFT/trackB.jsonl" --fanout-train-turns
run "kochat_if_fanout_me$SUFFIX" "$SFT/trackA_if.jsonl" --fanout-train-turns --medium-effort
run "kochat_chat$SUFFIX"         "$SFT/trackA_chat.jsonl"

echo "== ALL DONE ($(date +%H:%M:%S)) -> $OUT"
