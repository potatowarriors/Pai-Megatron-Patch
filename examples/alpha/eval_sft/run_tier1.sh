#!/bin/bash
# run_tier1.sh — T1 코어 (R1 표준 세팅). thinking 모델 → temp 0.6/top_p 0.95/max 32K.
#   IFEval 만 greedy(temp 0). 수학(AIME/HMMT)은 avg@16(별도 태스크에 내장).
# 사용: bash eval_sft/run_tier1.sh <BASE_URL> <RUN_NAME> [LIMIT]
set -euo pipefail
BASE_URL="${1:?base_url}"; RUN_NAME="${2:?run name}"; LIMIT="${3:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"; LM=/home/work/vidsearch/tools/lmeval0412
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"
export HF_HOME=/home/work/Datasets/benchmarks HF_DATASETS_CACHE=/home/work/Datasets/benchmarks
export PYTHONPATH=$LM NUMEXPR_MAX_THREADS=64 TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$HERE/../.env" 2>/dev/null | cut -d= -f2)
LIM=""; [ -n "$LIMIT" ] && LIM="--limit $LIMIT"
MA="base_url=${BASE_URL}/chat/completions,model=alpha,num_concurrent=64,timeout=2400,max_retries=2,tokenized_requests=False,max_length=32768"

run() { # tasks, gen_kwargs, desc
  echo "[T1] $3"
  python3 "$HERE/../scripts/eval_wrapper.py" --model local-chat-completions \
    --model_args "$MA" --tasks "$1" --apply_chat_template --include_path "$HERE/tasks" \
    --output_path "$OUT" --log_samples --seed 1234 $LIM --gen_kwargs "$2"
}
# 지식·객관식 (R1: temp 0.6, top_p 0.95, max 32K). CoT 필요.
run "mmlu_pro,gpqa_diamond_generative_n_shot" \
    "max_gen_toks=32768,temperature=0.6,top_p=0.95,do_sample=true" "지식/추론 (temp 0.6)"
# IFEval — 지시이행은 greedy 관례
run "ifeval" "max_gen_toks=2048,temperature=0.0,do_sample=false" "IFEval (greedy)"
# 수학 avg@16 — 세팅은 태스크 yaml(temp 0.6/top_p 0.95/repeats 16)에 내장. gen_kwargs override 안 함.
echo "[T1] 수학 avg@16 (AIME25, HMMT25 — temp 0.6, 16 samples)"
python3 "$HERE/../scripts/eval_wrapper.py" --model local-chat-completions \
  --model_args "$MA" --tasks "aime25_avg16,hmmt_feb_2025_avg16" --apply_chat_template \
  --include_path "$HERE/tasks" --output_path "$OUT" --log_samples --seed 1234 $LIM
echo "== T1 완료: $OUT =="
