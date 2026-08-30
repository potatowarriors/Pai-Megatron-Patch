#!/bin/bash
# run_tier1.sh — T1 코어 (AA/Nemotron 3 Ultra 규약).
#
# 생성 파라미터를 여기서 정하지 않는다. **태스크 yaml 이 단일 정본**이다
# (`eval_sft/tasks/*_aa.yaml`: temp 1.0 / top_p 0.95 / max_gen 32768 / seed null /
#  skip_special_tokens false / 반복 횟수). 러너가 --gen_kwargs 로 덮어쓰면 어느 설정이
# 적용됐는지 사후에 알 수 없어진다 — 2026-08-30 사고에서 원인 분리를 막은 요인.
#
# 선행 조건: G1~G3 게이트 통과 (`eval_sft/check_gates.py`). docs/SFT_BENCHMARKS.md §7.
#
# 사용: bash eval_sft/run_tier1.sh <BASE_URL> <RUN_NAME> [LIMIT]
#   BASE_URL : http://host:port/v1
#   LIMIT    : 스모크용 문항 수 제한 (비우면 전량)
# 환경변수:
#   TASKS    : 실행 태스크 (기본 5종 전부)
#   CONC     : 동시 요청 수 (기본 64)
#   MAXLEN   : 서빙 컨텍스트 상한 (기본 40960 = 32768 생성 + 프롬프트 여유)
set -euo pipefail
BASE_URL="${1:?base_url (예: http://localhost:8100/v1)}"
RUN_NAME="${2:?run name}"
LIMIT="${3:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
LM=/home/work/vidsearch/tools/lmeval0412
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"

export HF_HOME=/home/work/Datasets/benchmarks HF_DATASETS_CACHE=/home/work/Datasets/benchmarks
export PYTHONPATH=$LM NUMEXPR_MAX_THREADS=64 TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$HERE/../.env" 2>/dev/null | cut -d= -f2 || true)

TASKS="${TASKS:-mmlu_pro_aa,gpqa_diamond_aa,ifeval_aa,aime25_aa,hmmt_feb_2025_aa}"
CONC="${CONC:-64}"
MAXLEN="${MAXLEN:-40960}"
LIM=""; [ -n "$LIMIT" ] && LIM="--limit $LIMIT"

MA="base_url=${BASE_URL}/chat/completions,model=alpha,num_concurrent=${CONC}"
MA="${MA},timeout=3600,max_retries=10,tokenized_requests=False,max_length=${MAXLEN}"

echo "[T1] tasks=$TASKS"
echo "[T1] 생성 파라미터는 태스크 yaml 정본 — 러너는 덮어쓰지 않는다"
echo "[T1] out=$OUT"

python3 "$HERE/../scripts/eval_wrapper.py" \
  --model local-chat-completions \
  --model_args "$MA" \
  --tasks "$TASKS" \
  --apply_chat_template \
  --include_path "$HERE/tasks" \
  --output_path "$OUT" \
  --log_samples \
  --seed 1234 \
  $LIM

echo "== T1 완료: $OUT =="
echo "== 판정: python3 $HERE/summarize.py $OUT =="
