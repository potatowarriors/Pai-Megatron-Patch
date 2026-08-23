#!/usr/bin/env bash
# alpha SFT 128k 장문 버킷 변환 — 64k 버킷의 여집합(>64k 샘플)만 수용.
#
# --min-tokens 65536 이 중복을 원천 차단: 64k 버킷은 >64k 를 too_long 드롭,
# 여기는 <=64k 를 below_min 드롭. >64k 잔여 풀(토큰 기준 ~24B): cp 12.7B /
# science 3.9B / proofs 2.8B / swe 2.2B / arc 1.4B / math 1.1B + 소량 셋.
# 128k 런 예산·블렌드는 별도 yaml (SWE 꼬리 1-pass 앵커 동형) — bins 완성 후.
# 학습 구성은 128K@CP8 + chunked offload (MUON_OFFLOAD_BACKPORT GO 실증).
#
# 실행 (유휴 노드): NCORES=96 bash convert_sft_128k.sh
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SFT=/home/work/Datasets/LL_datasets/posttraining/SFT
OUT=${OUT:-/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_pad16}
NCORES=${NCORES:-96}
TOK=$REPO/examples/alpha/tokenizer_v5
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO:$REPO/backends/megatron/Megatron-LM-251125${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"

run() {
  local name=$1 input=$2
  if [ -f "$OUT/$name/data_text_document.idx" ]; then
    echo "== $name: SKIP (idx 존재)"; return
  fi
  echo "== $name: $input  ($(date +%H:%M:%S))"
  mkdir -p "$OUT/$name"
  nice -n 10 python "$REPO/toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py" \
    --input "$input" --tokenizer "$TOK" \
    --output-prefix "$OUT/$name/data" \
    --seq-length 131072 --pad-doc-multiple 16 --min-tokens 65536 \
    --workers "$NCORES" \
    || echo "!! $name FAILED (exit $?)"
}

# >64k 꼬리 보유 세트만 (측정 too_long > 0), 작은 것부터
run math_proofs_v2   "$SFT/Nemotron-Math-Proofs-v2"
run swe_v3           "$SFT/Nemotron-SFT-SWE-v3/data"
run arc_agi_v1       "$SFT/Nemotron-SFT-ARC-AGI-v1"
run math_v4          "$SFT/Nemotron-SFT-Math-v4"
run science_v2       "$SFT/Nemotron-SFT-Science-v2"
run cp_v2            "$SFT/Nemotron-SFT-Competitive-Programming-v2"

echo "== ALL DONE ($(date +%H:%M:%S)) -> $OUT"
