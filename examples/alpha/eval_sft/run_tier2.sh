#!/bin/bash
# run_tier2.sh — T2 롱컨텍스트 RULER-NIAH (64K/128K/256K). 롱컨텍스트 fleet(≥262144) 필요.
# 사용: bash eval_sft/run_tier2.sh <BASE_URL> <RUN_NAME> [TASKS] [LIMIT]
set -euo pipefail
BASE_URL="${1:?base_url}"; RUN_NAME="${2:?run name}"
TASKS="${3:-ruler_niah_single_1,ruler_niah_single_2,ruler_niah_multikey_1,ruler_niah_multivalue}"
LIMIT="${4:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"; LM=/home/work/vidsearch/tools/lmeval0412
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"
export HF_HOME=/home/work/Datasets/benchmarks HF_DATASETS_CACHE=/home/work/Datasets/benchmarks
export PYTHONPATH=$LM:$HERE/tasks NUMEXPR_MAX_THREADS=64 TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$HERE/../.env" 2>/dev/null | cut -d= -f2)
LIM=""; [ -n "$LIMIT" ] && LIM="--limit $LIMIT"
# RULER 는 긴 프롬프트 → num_concurrent 낮게(각 요청이 256K prefill). max_length 262144.
python3 "$HERE/../scripts/eval_wrapper.py" \
  --model local-chat-completions \
  --model_args "base_url=${BASE_URL}/chat/completions,model=alpha,num_concurrent=8,timeout=2400,max_retries=2,tokenized_requests=False,max_length=262144" \
  --apply_chat_template --tasks "$TASKS" --include_path "$HERE/tasks" \
  --output_path "$OUT" --log_samples --seed 1234 $LIM
echo "== T2(RULER) 완료: $OUT =="
