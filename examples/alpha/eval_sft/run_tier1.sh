#!/bin/bash
# run_tier1.sh — T1 코어 벤치 (chat/thinking 모드, vLLM OpenAI 엔드포인트 경유)
# 사용: bash eval_sft/run_tier1.sh <BASE_URL> <RUN_NAME> [TASKS] [LIMIT]
#   BASE_URL 예: http://localhost:8000/v1  (served-model-name=alpha)
#   TASKS 기본: mmlu_pro,gpqa_diamond_generative_n_shot,ifeval,aime25,hmmt_feb_2025
#   LIMIT: 태스크당 샘플 상한(스모크용). 비우면 전량.
set -euo pipefail
BASE_URL="${1:?base_url required, e.g. http://localhost:8000/v1}"
RUN_NAME="${2:?run name required}"
TASKS="${3:-mmlu_pro,gpqa_diamond_generative_n_shot,ifeval,aime25,hmmt_feb_2025}"
LIMIT="${4:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LM=/home/work/vidsearch/tools/lmeval0412
OUT="$HERE/results/$RUN_NAME"
mkdir -p "$OUT"
export HF_HOME=/home/work/Datasets/benchmarks
export HF_DATASETS_CACHE=/home/work/Datasets/benchmarks
export PYTHONPATH=$LM
export NUMEXPR_MAX_THREADS=64
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
LIM_ARG=""; [ -n "$LIMIT" ] && LIM_ARG="--limit $LIMIT"
# 생성 태스크는 thinking 여유로 max_gen_toks 크게 (태스크 yaml 기본 32768).
# gen_kwargs override 로 서버가 <|im_end|> 에서 멈추도록 stop 명시.
python3 -m lm_eval \
  --model local-chat-completions \
  --model_args "base_url=${BASE_URL}/chat/completions,model=alpha,num_concurrent=64,timeout=1200,max_retries=2,tokenized_requests=False" \
  --tasks "$TASKS" \
  --apply_chat_template \
  --include_path "$HERE/tasks" \
  --output_path "$OUT" \
  --log_samples \
  --seed 1234 \
  $LIM_ARG \
  --gen_kwargs "max_gen_toks=32768,temperature=0.0"
echo "== T1 완료: $OUT =="
