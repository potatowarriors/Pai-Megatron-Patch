#!/bin/bash
# run_tier2.sh — T2 롱컨텍스트 RULER-NIAH (AA/Nemotron 규약, 2026-08-30 재작성).
#
# **Reasoning-Off 로 돈다** — Nemotron Nano 카드가 RULER 만 예외로 추론을 끈다.
# 설정은 태스크 yaml 이 정본(`tasks/ruler_niah_*_aa.yaml`): temp 0.00001 / top_p 0.99 /
# max_gen 512 / chat_template_kwargs.enable_thinking=false. 러너는 덮어쓰지 않는다.
#
# **롱 fleet 전제**: 131072 구간 프롬프트가 들어가므로 서빙 --max-model-len ≥ 131072+여유.
#   bash eval_sft/serve_fleet.sh <ckpt> 139264 <N> <PROXY_PORT>
# 표준 fleet(40960)로 돌리면 프롬프트가 창을 넘어 전량 실패한다.
#
# 사용: bash eval_sft/run_tier2.sh <BASE_URL> <RUN_NAME> [LIMIT]
set -euo pipefail
BASE_URL="${1:?base_url}"; RUN_NAME="${2:?run name}"; LIMIT="${3:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"; LM=/home/work/vidsearch/tools/lmeval0412
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"
export HF_HOME=/home/work/Datasets/benchmarks HF_DATASETS_CACHE=/home/work/Datasets/benchmarks
export PYTHONPATH=$LM NUMEXPR_MAX_THREADS=64 TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$HERE/../.env" 2>/dev/null | cut -d= -f2 || true)

TASKS="${TASKS:-ruler_niah_single_1_aa,ruler_niah_single_2_aa,ruler_niah_multikey_1_aa,ruler_niah_multivalue_aa}"
CONC="${CONC:-8}"          # 요청당 131K prefill — 동시성을 낮게
MAXLEN="${MAXLEN:-139264}"
LIM=""; [ -n "$LIMIT" ] && LIM="--limit $LIMIT"

echo "[T2] tasks=$TASKS (Reasoning-Off)"
echo "[T2] out=$OUT"
python3 "$HERE/../scripts/eval_wrapper.py" \
  --model local-chat-completions \
  --model_args "base_url=${BASE_URL}/chat/completions,model=alpha,num_concurrent=${CONC},timeout=3600,max_retries=10,tokenized_requests=False,max_length=${MAXLEN}" \
  --apply_chat_template --tasks "$TASKS" --include_path "$HERE/tasks" \
  --output_path "$OUT" --log_samples --seed 1234 $LIM
echo "== T2 완료: $OUT =="
