#!/bin/bash
# ruler_cap_run.sh — RULER 능력 스위트(11태스크, n=50) 실행 (study/lc_512k_eval.md §6).
#
#   sub1 에서:  setsid nohup bash scripts/ruler_cap_run.sh [CELL] \
#                 > outputs/ruler_cap_eval/run.log 2>&1 < /dev/null &
#   CELL 기본 sft:yarn2 (능력 판정 대상). 중단: touch outputs/ruler_cap_eval/STOP
#
# lc512k_grid.sh 와 같은 골격(대기 → 프로파일 → fleet 524288 → 게이트 → lm_eval → 요약).
# 차이: ① lc512k 그리드가 살아 있으면 그 DONE 까지 추가로 기다린다(그리드가 fleet 를
# 갈아끼우므로 동시 불가) ② 태스크 = ruler_cap_* 11종 (qa_squad·cwe 구조적 제외 — utils docstring) ③ 결과 = outputs/ruler_cap_eval/.
# 판정(구간 평균 ≥85 = 그 길이 "지원")은 scripts/ruler_cap_summarize.py 가 계산한다.
set -uo pipefail
ALPHA=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
EVAL=$ALPHA/eval_sft
LC512K=$ALPHA/outputs/lc512k_eval
OUT=${OUT:-$ALPHA/outputs/ruler_cap_eval}
LM=/home/work/vidsearch/tools/lmeval0412
MAXLEN=524288
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PROXY=${PROXY:-8100}; BURL="http://localhost:$PROXY/v1"
CONC="${CONC:-8}"
CELL="${1:-sft:yarn2}"; MODEL="${CELL%%:*}"; PROF="${CELL##*:}"
TAG="${MODEL}_${PROF}"
SFT_RUN=$ALPHA/outputs/alpha_baseline_48L_sft_128k_full_20260828_081911
BASE="${BASE:-$ALPHA/outputs/alpha_baseline_48L_lc_b_resume_20260826_223836/hfmodel_0000320}"
SFT="${SFT:-$(ls -d "$SFT_RUN"/hfmodel_* 2>/dev/null | while read -r d; do [ -f "$d/generation_config.json" ] && echo "$d"; done | tail -1)}"
TASKS="ruler_cap_niah_single_1,ruler_cap_niah_single_2,ruler_cap_niah_single_3,ruler_cap_niah_multikey_1,ruler_cap_niah_multikey_2,ruler_cap_niah_multikey_3,ruler_cap_niah_multiquery,ruler_cap_niah_multivalue,ruler_cap_vt,ruler_cap_fwe,ruler_cap_qa_hotpot"

mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }
stopped() { [ -f "$OUT/STOP" ]; }

export HF_HOME=/home/work/Datasets/benchmarks HF_DATASETS_CACHE=/home/work/Datasets/benchmarks
export PYTHONPATH=$LM NUMEXPR_MAX_THREADS=64 TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ALPHA/.env" 2>/dev/null | cut -d= -f2 || true)

log "cell=$CELL tag=$TAG tasks=11종 n=50"

# ── 대기: lc512k 그리드 → 일반 벤치/fleet → GPU 유휴 ──────────────────────
while pgrep -f "[l]c512k_grid.sh" >/dev/null 2>&1 && [ ! -f "$LC512K/DONE" ]; do
    stopped && { log "STOP — 종료"; exit 0; }
    log "대기: lc512k 그리드 진행 중 (DONE 대기)"; sleep 180
done
while :; do
    stopped && { log "STOP — 종료"; exit 0; }
    procs=$(pgrep -f "[r]un_suite.sh|[r]un_tier[12].sh|[r]un_swe.sh|[r]un_terminal.sh|[e]val_wrapper.py|[e]val_new_ckpt.sh|[l]c512k_grid.sh|alpha_serve_venv/bin/vllm" | wc -l)
    maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$procs" -eq 0 ] && [ "${maxmem:-99999}" -lt 2000 ] && { log "sub1 유휴 (max GPU mem ${maxmem}MiB)"; break; }
    log "대기: 프로세스 $procs 개, max GPU mem ${maxmem}MiB"; sleep 120
done

# ── 프로파일 (lc512k 것 재사용; 없으면 생성) ──────────────────────────────
[ "$MODEL" = sft ] && SRC=$SFT || SRC=$BASE
DIR=$LC512K/profiles/$TAG
if [ ! -f "$DIR/.lc_profile.json" ] || ! python3 "$ALPHA/tools/set_long_context_config.py" --check "$DIR" >/dev/null 2>&1; then
    rm -rf "$DIR"
    case $PROF in
        ext)   python3 "$ALPHA/tools/set_long_context_config.py" "$SRC" --out "$DIR" --max-pos $MAXLEN ;;
        yarn2) python3 "$ALPHA/tools/set_long_context_config.py" "$SRC" --out "$DIR" --max-pos $MAXLEN --yarn-factor 2 ;;
        yarn4) python3 "$ALPHA/tools/set_long_context_config.py" "$SRC" --out "$DIR" --max-pos $MAXLEN --yarn-factor 4 --yarn-original 131072 ;;
        *) log "❌ 알 수 없는 프로파일 $PROF"; exit 1 ;;
    esac || { log "❌ 프로파일 실패"; exit 1; }
fi
log "프로파일 $DIR (src=$SRC)"

# ── fleet ─────────────────────────────────────────────────────────────────
fleet_down() { bash "$EVAL/stop_fleet.sh" "$GPUS" >/dev/null 2>&1 || true; }
fleet_down; sleep 5
( export PIP_CONSTRAINT= TOOLS=0 GPUS="$GPUS"
  setsid bash "$EVAL/serve_fleet.sh" "$DIR" "$MAXLEN" "$NGPU" "$PROXY" > "$OUT/fleet_$TAG.log" 2>&1 < /dev/null & )
ok=0
for i in $(seq 1 90); do
    c=0
    for g in $(echo "$GPUS" | tr ',' ' '); do
        [ "$(curl -s -o /dev/null -w %{http_code} --max-time 3 http://localhost:$((8000+g))/v1/models)" = "200" ] && c=$((c+1))
    done
    [ "$c" -eq "$NGPU" ] && { ok=1; log "fleet 준비 $c/$NGPU"; break; }
    [ "$(pgrep -f 'alpha_serve_venv/bin/vllm' | wc -l)" -eq 0 ] && [ $i -gt 3 ] && break
    sleep 20
done
[ "$ok" = 1 ] || { log "❌ fleet 준비 실패 — $OUT/fleet_$TAG.log"; fleet_down; exit 1; }

CDIR="$OUT/$TAG"; mkdir -p "$CDIR"
python3 "$EVAL/check_gates.py" --hf-dir "$DIR" --base-url "$BURL" \
    --max-tokens $([ "$MODEL" = sft ] && echo 32768 || echo 2048) > "$CDIR/gates.log" 2>&1
if [ $? -ne 0 ] && [ "$MODEL" = sft ]; then log "❌ 게이트 실패 — $CDIR/gates.log"; fleet_down; exit 1; fi
log "게이트 확인 (로그 $CDIR/gates.log)"

log "RULER 능력 스위트 시작 (11태스크 × 4구간 × 50 — 데이터 생성 ~1h + 추론 ~1.5h 추정)"
python3 "$ALPHA/scripts/eval_wrapper.py" \
    --model local-chat-completions \
    --model_args "base_url=${BURL}/chat/completions,model=alpha,num_concurrent=${CONC},timeout=7200,max_retries=10,tokenized_requests=False,max_length=$MAXLEN" \
    --apply_chat_template --tasks "$TASKS" --include_path "$EVAL/tasks" \
    --output_path "$CDIR" --log_samples --seed 1234 > "$CDIR/lm_eval.log" 2>&1
rc=$?
fleet_down
[ $rc -eq 0 ] || { log "❌ lm_eval 실패 (rc=$rc) — $CDIR/lm_eval.log"; exit 1; }

python3 "$ALPHA/scripts/ruler_cap_summarize.py" "$OUT" | tee "$OUT/summary.log"
date > "$OUT/DONE"
log "완료 → $OUT/SUMMARY.md"
