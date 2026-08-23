#!/usr/bin/env bash
# alpha SFT 64k 버킷 일괄 변환 — build_alpha_sft_idxmap.py, 세트당 순차 실행.
#
# 블렌드 설계 (2026-08-23 확정, SWE 1-pass 앵커 / 총 40B bin-tokens):
#   swe_v3 24.5%(1.0ep) · chat 21.0%(if+chat[복원후]) · cp_v2 14.5%(0.22ep) ·
#   math_v4 11.0%(0.66ep) · science_v2 10.0%(0.27ep) · multilingual 6.5%(0.88ep) ·
#   agentic(opencode+cuda) 5.0%(0.28ep) · arc_agi 3.0%(0.15ep) ·
#   proofs≤64k 1.5% · longblocks_sft 1.5%(변환기 별도 — 후속) ·
#   safety 1.0% · identity 0.5%(결정 #9 상한 내)
#   근거: Ultra3 함의 epoch 역산(SWE-v1 1.1ep) + 카테고리 비율 + chat E_max 4~5.
#
# NFS 주의: emit(쓰기)이 지배 — 동시 1개(순차)로 고정. LC-A 학습(main1)이
# 같은 NFS를 읽는 중 (러너북 함정 4: 동시 쓰기 2개에서 포화).
# 멱등: <name>/data_text_document.idx 존재 시 스킵.
#
# 실행 (유휴 노드): NCORES=96 bash convert_sft_64k.sh
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SFT=/home/work/Datasets/LL_datasets/posttraining/SFT
OUT=${OUT:-/home/work/Datasets/LL_preprocessed/v5/sft_packed_64k_pad16}
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
    --seq-length 65536 --pad-doc-multiple 16 --workers "$NCORES" \
    || echo "!! $name FAILED (exit $?)"
}

# 작은 셋 먼저 (이른 신호), 대형 셋 후반
run identity_v1      "$SFT/alpha-SFT-Identity-v1/data/train.jsonl"
run cuda_v1          "$SFT/Nemotron-SFT-CUDA-v1"
run safety_v2        "$SFT/Nemotron-SFT-Safety-v2"
run math_proofs_v2   "$SFT/Nemotron-Math-Proofs-v2"
run swe_v3           "$SFT/Nemotron-SFT-SWE-v3/data"
run arc_agi_v1       "$SFT/Nemotron-SFT-ARC-AGI-v1"
run math_v4          "$SFT/Nemotron-SFT-Math-v4"
run opencode_v1      "$SFT/Nemotron-SFT-OpenCode-v1"
run science_v2       "$SFT/Nemotron-SFT-Science-v2"
run cp_v2            "$SFT/Nemotron-SFT-Competitive-Programming-v2"

# Multilingual-v2: hi 제외, ko/ja/pt 파일별
for f in "$SFT"/Nemotron-SFT-Multilingual-v2/*_ko_* \
         "$SFT"/Nemotron-SFT-Multilingual-v2/*_ja_* \
         "$SFT"/Nemotron-SFT-Multilingual-v2/*_pt_*; do
  [ -f "$f" ] || continue
  run "ml_$(basename "$f" .jsonl)" "$f"
done

# chat_v3: if 는 스모크 때 완료(스킵됨), chat 은 복원 완료 후 별도 실행:
#   run chat_v3_chat "$SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data/chat.with_prompts.jsonl"
run chat_v3_if "$SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data/instruction_following.jsonl"

echo "== ALL DONE ($(date +%H:%M:%S)) -> $OUT"
