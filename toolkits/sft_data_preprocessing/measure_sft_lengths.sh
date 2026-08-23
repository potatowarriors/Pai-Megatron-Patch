#!/usr/bin/env bash
# ultra_v3 우선 SFT 세트 전수 길이 재측정 — tokenizer_v5 + chat template 렌더 기준.
#
# 기존 수치(파일 앞 3k행 샘플 + chars/4 근사, SFT_RL_DATASETS.md §2.4)를 대체하는
# 정밀 측정. 부수 산출: 드롭율(null/injection/span_mismatch) 전수 실측 —
# Chat-v3 chat split 은 전 행 null_content 로 떨어져 복원 필요 규모가 확정된다.
#
# 실행 (유휴 노드): NCORES=96 bash measure_sft_lengths.sh
# 산출: $OUT_DIR/<name>.stats.json (+ <name>.dropped.jsonl)
# 멱등: stats.json 존재 시 스킵 — 중단 후 재실행 안전.
set -u  # -e 금지: 한 셋 실패가 다음 셋을 막지 않게 하고, 실패는 로그에 남긴다
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SFT=/home/work/Datasets/LL_datasets/posttraining/SFT
OUT_DIR=${OUT_DIR:-/home/work/Datasets/LL_preprocessed/v5/sft_measure}
NCORES=${NCORES:-96}
TOK=$REPO/examples/alpha/tokenizer_v5
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT_DIR"

run() {
  local name=$1 input=$2
  if [ -f "$OUT_DIR/$name.stats.json" ]; then
    echo "== $name: SKIP (stats 존재)"; return
  fi
  echo "== $name: $input  ($(date +%H:%M:%S))"
  nice -n 10 python "$REPO/toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py" \
    --input "$input" --tokenizer "$TOK" \
    --output-prefix "$OUT_DIR/$name" \
    --measure-only --workers "$NCORES" \
    || echo "!! $name FAILED (exit $?)"
}

# 작은 셋부터 (이른 신호), 대형 셋은 후반
run identity_v1      "$SFT/alpha-SFT-Identity-v1/data/train.jsonl"
run chat_v3_if       "$SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data/instruction_following.jsonl"
run chat_v3_chat     "$SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data/chat.jsonl"
run cuda_v1          "$SFT/Nemotron-SFT-CUDA-v1"
run safety_v2        "$SFT/Nemotron-SFT-Safety-v2"
run swe_v3           "$SFT/Nemotron-SFT-SWE-v3/data"
run math_proofs_v2   "$SFT/Nemotron-Math-Proofs-v2"
run arc_agi_v1       "$SFT/Nemotron-SFT-ARC-AGI-v1"
run math_v4          "$SFT/Nemotron-SFT-Math-v4"
run opencode_v1      "$SFT/Nemotron-SFT-OpenCode-v1"
run science_v2       "$SFT/Nemotron-SFT-Science-v2"
run cp_v2            "$SFT/Nemotron-SFT-Competitive-Programming-v2"

# Multilingual-v2: hi 제외(alpha 미지원 언어), ko/ja/pt 파일별 측정
for f in "$SFT"/Nemotron-SFT-Multilingual-v2/*_ko_* \
         "$SFT"/Nemotron-SFT-Multilingual-v2/*_ja_* \
         "$SFT"/Nemotron-SFT-Multilingual-v2/*_pt_*; do
  [ -f "$f" ] || continue
  run "multilingual_$(basename "$f" .jsonl)" "$f"
done

echo "== ALL DONE ($(date +%H:%M:%S)) -> $OUT_DIR"
