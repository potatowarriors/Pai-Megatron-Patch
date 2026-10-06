#!/bin/bash
# run_base4_vllm.sh — `scripts/run_benchmarks.sh`(HF accelerate, base 프로토콜)의 **vLLM 판** (2026-10-06, 사용자 지시).
#
#   같은 lm_eval 내장 태스크·같은 few-shot 을, HF `--model hf` 대신 **fleet(vLLM 0.25.1 단일 GPU ×8 + lb_proxy)** 의
#   `/v1/completions` 로 돈다(`--model local-completions`). 프로토콜은 run_benchmarks.sh 와 동일:
#     mmlu 5-shot(loglikelihood, 4지선다 acc) · gsm8k 4-shot(greedy, exact_match) · humaneval 0-shot(pass@1) · mbpp 3-shot(pass@1)
#   채팅 템플릿·추론 없음(raw completion) — 사전학습 ckpt 수치와 같은 조건이다. SFT 정본 스위트(run_tier1.sh, *_aa.yaml, 추론 ON)와는
#   다른 프로토콜이므로 TRACKING.md 에 섞지 않는다. 결과는 `<HF_CKPT>/eval_results_vllm/`(HF 판은 `eval_results/`).
#
#   loglikelihood 는 lm_eval 이 `max_tokens=1, logprobs=1, echo=True` 로 보내고 vLLM 이 prompt logprobs 를 되돌린다
#   (`lm_eval/models/openai_completions.py` `_create_payload`). V1 엔진은 prompt_logprobs 요청에 prefix cache 를 안 쓴다.
#   completions 요청은 messages 가 없어 lb_proxy 세션 키가 없다 → 라운드로빈(8 GPU 데이터 병렬).
#
# 사용: bash eval_sft/run_base4_vllm.sh <HF_CKPT> [TASKS] [LIMIT]
#   TASKS : 쉼표 목록, 기본 mmlu,gsm8k,humaneval,mbpp
#   LIMIT : >0 이면 태스크당 문항 수 제한(스모크) → 결과는 eval_results_vllm_smoke/ 에, 기록 금지
# 환경변수: GPUS(기본 0~7) · MAX_LEN(기본 16384) · CONC(동시 요청, 기본 16) · BS(요청당 프롬프트 수, 기본 4) · PREFIX_CACHE(기본 1)
#   GPU_MEM_UTIL(기본 0.85) · MAX_BATCHED_TOKENS(기본 2048) — **prompt logprobs 는 prefill 스텝의 토큰 수 × vocab(163,968) fp32 logits 를
#   통째로 만든다.** 코어 fleet 기본(0.95 · 8192)으로는 8192 × 164k × 4 B ≈ 5.4 GB + log_softmax 사본이 KV 가 점유한 뒤의 여유(≈1.6 GB)를
#   넘어 8 백엔드가 동시에 `compute_logprobs` OOM 으로 죽었다(2026-10-06 10:01 첫 전량 실행, 스모크 8문항은 통과). 2048 이면 ≈1.3 GB.
set -uo pipefail
CKPT="${1:?HF checkpoint dir}"; TASKS="${2:-mmlu,gsm8k,humaneval,mbpp}"; LIMIT="${3:-0}"
CKPT="$(cd "$CKPT" && pwd)" || { echo "[base4] ❌ ckpt 없음: $1"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MAX_LEN="${MAX_LEN:-16384}"; PROXY=8100; BURL="http://localhost:$PROXY/v1/completions"
LOGD=/home/work/vidsearch/tools/bench_logs; mkdir -p "$LOGD"
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then OUT="$CKPT/eval_results_vllm_smoke"; LIM="--limit $LIMIT"; else OUT="$CKPT/eval_results_vllm"; LIM=""; fi
mkdir -p "$OUT"
TAG="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")"
# run_benchmarks.sh 와 같은 캐시·코드 실행 허용
export HF_HOME=/home/work/Datasets/benchmarks HF_DATASETS_CACHE=/home/work/Datasets/benchmarks HF_ALLOW_CODE_EVAL=1
export WANDB_MODE=disabled
log() { echo "[base4 $(date '+%F %T')] $*"; }
has() { case ",$TASKS," in *",$1,"*) return 0;; *) return 1;; esac; }

log "ckpt=$CKPT tasks=$TASKS limit=$LIMIT out=$OUT"
[ -f "$CKPT/model.safetensors.index.json" ] || { log "❌ 가중치 인덱스 없음(변환 미완)"; exit 1; }

# ── fleet: 코어 스위트와 같은 플래그(TOOLS=0 · reasoning 파서 없음), 창만 짧게 ──
log "fleet 기동 (max_len=$MAX_LEN GPUS=$GPUS prefix_cache=${PREFIX_CACHE:-1} gpu_mem=${GPU_MEM_UTIL:-0.85} batched_tokens=${MAX_BATCHED_TOKENS:-2048})"
bash "$HERE/stop_fleet.sh" "$GPUS" >/dev/null 2>&1 || true; sleep 5
( export PIP_CONSTRAINT= TOOLS=0 GPUS="$GPUS" REASONING_PARSER="" PREFIX_CACHE="${PREFIX_CACHE:-1}" \
         GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}" MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
  setsid bash "$HERE/serve_fleet.sh" "$CKPT" "$MAX_LEN" "$NGPU" "$PROXY" > "$LOGD/fleet_base4_${TAG}.log" 2>&1 < /dev/null & )
ok=0
for i in $(seq 1 60); do
  c=0; for g in $(echo "$GPUS" | tr ',' ' '); do [ "$(curl -s -o /dev/null -w %{http_code} --max-time 3 http://localhost:$((8000+g))/v1/models)" = "200" ] && c=$((c+1)); done
  [ "$c" -eq "$NGPU" ] && { ok=1; break; }; sleep 20
done
[ "$ok" -eq 1 ] || { log "❌ fleet 준비 실패"; bash "$HERE/stop_fleet.sh" "$GPUS" >/dev/null 2>&1; exit 1; }
log "fleet 준비 $NGPU/$NGPU"
# echo+logprobs 경로 1건 확인 — 프롬프트 logprobs 가 돌아와야 loglikelihood 태스크가 성립한다
python3 - "$BURL" <<'PY' || { log "❌ completions echo/logprobs 스모크 실패"; bash "$HERE/stop_fleet.sh" "$GPUS" >/dev/null 2>&1; exit 1; }
import json, sys, urllib.request
r = urllib.request.Request(sys.argv[1], data=json.dumps({"model": "alpha", "prompt": "The capital of France is", "max_tokens": 1,
    "temperature": 0, "logprobs": 1, "echo": True}).encode(), headers={"Content-Type": "application/json"})
o = json.load(urllib.request.urlopen(r, timeout=120))["choices"][0]
lp = o["logprobs"]; n = len(lp["token_logprobs"])
# 프롬프트 5토큰 + 생성 1토큰 ≥ 6 이어야 echo 가 프롬프트 logprob 를 돌려준 것이다(첫 토큰은 None 일 수 있다)
assert n >= 6 and lp.get("top_logprobs") is not None, lp
print(f"[base4] echo OK — 프롬프트+1 토큰 {n}개 logprob, 생성 '{o['text'].strip()}'")
PY

MA="model=alpha,base_url=$BURL,tokenizer=$CKPT,tokenized_requests=True,num_concurrent=${CONC:-16},max_retries=5,timeout=1800,max_length=$MAX_LEN"
rc=0
run_one() {  # $1=tasks $2=num_fewshot
  log "=== $1 ($2-shot) ==="
  python3 "$HERE/../scripts/eval_wrapper.py" --model local-completions --model_args "$MA" \
    --tasks "$1" --num_fewshot "$2" --batch_size "${BS:-4}" --output_path "$OUT" --log_samples --confirm_run_unsafe_code $LIM \
    > "$LOGD/base4_${TAG}_$1.log" 2>&1 || { log "❌ $1 실패 (rc=$?) — $LOGD/base4_${TAG}_$1.log"; rc=1; }
  grep -a -A12 "^|Tasks" "$LOGD/base4_${TAG}_$1.log" | head -14
}
has mmlu      && run_one mmlu 5
has gsm8k     && run_one gsm8k 4
has humaneval && run_one humaneval 0
has mbpp      && run_one mbpp 3

log "fleet 종료"; bash "$HERE/stop_fleet.sh" "$GPUS" >/dev/null 2>&1 || true
log "완료 (rc=$rc): $OUT"
exit $rc
