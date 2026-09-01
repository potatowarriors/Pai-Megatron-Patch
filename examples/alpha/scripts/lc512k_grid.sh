#!/bin/bash
# lc512k_grid.sh — 512K 추론-only 판정 그리드 (study/lc_512k_eval.md Phase 0-2 + Phase 1).
#
#   sub1 에서:  mkdir -p outputs/lc512k_eval && setsid nohup bash scripts/lc512k_grid.sh \
#                 > outputs/lc512k_eval/grid.log 2>&1 < /dev/null &
#   중단:       touch outputs/lc512k_eval/STOP   (현재 셀이 끝나면 멈춘다; fleet 는 내린다)
#
# 단계:
#   ① sub1 벤치 suite(run_suite/run_tier*/run_swe/run_terminal/eval_wrapper) 와 fleet(vLLM) 종료 대기
#      + 전 GPU 유휴(<2GB). suite 가 fleet 를 갈아끼우며 stop_fleet.sh 로 **모든** vLLM 을 죽이므로
#      suite 와 동시에 띄울 수 없다 — 그래서 기다린다 (2026-09-01 iter900 suite 와 경합).
#   ② 프로파일 생성 (tools/set_long_context_config.py — 가중치 symlink, config.json 만 교체)
#   ③ (모델 × 프로파일) 셀마다: fleet 524288 기동 → 게이트(G1~G3) → RULER 512k 4태스크
#      (Reasoning-Off) → [SFT 모델만] 단문 회귀 표본(ifeval·gpqa 각 100문항) → fleet 종료
#   ④ 요약: scripts/lc512k_summarize.py → outputs/lc512k_eval/SUMMARY.md
#
# 셀 순서는 정보량 순 (중단돼도 앞 결과가 쓸모 있게): sft:yarn2 → sft:ext → sft:yarn4 → base:*.
#   yarn2 = factor 2 / original 262144 (실측 창 기준, 왜곡 최소)   ← 1순위 후보
#   ext   = 순수 외삽 (rope_scaling null, max_pos 만 524288)       ← 393K 절벽 대조군
#   yarn4 = factor 4 / original 131072 (학습 길이 기준, YaRN 정석)
#
# 결과는 eval_sft/results/ **밖**(outputs/lc512k_eval/)에 둔다 — aggregate_results.py 가 results/
# 아래 모든 디렉토리를 TRACKING 행으로 읽으므로 연구 그리드가 ckpt 추이표를 오염시키면 안 된다.
set -uo pipefail
ALPHA=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
EVAL=$ALPHA/eval_sft
OUT=${OUT:-$ALPHA/outputs/lc512k_eval}
LM=/home/work/vidsearch/tools/lmeval0412
MAXLEN=524288
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PROXY=${PROXY:-8100}; BURL="http://localhost:$PROXY/v1"
CONC_LONG="${CONC_LONG:-8}"     # 512K prefill ~30s/요청(H100) — 레플리카당 1 이면 충분
CONC_SHORT="${CONC_SHORT:-64}"
T1_LIMIT="${T1_LIMIT:-100}"
SFT_RUN=$ALPHA/outputs/alpha_baseline_48L_sft_128k_full_20260828_081911
BASE="${BASE:-$ALPHA/outputs/alpha_baseline_48L_lc_b_resume_20260826_223836/hfmodel_0000320}"
# SFT 최신 = generation_config.json 까지 갖춘(변환 완료) 마지막 hfmodel_*
SFT="${SFT:-$(ls -d "$SFT_RUN"/hfmodel_* 2>/dev/null | while read -r d; do [ -f "$d/generation_config.json" ] && echo "$d"; done | tail -1)}"
GRID="${GRID:-sft:yarn2 sft:ext sft:yarn4 base:yarn2 base:ext base:yarn4}"

mkdir -p "$OUT/profiles"
log() { echo "[$(date '+%F %T')] $*"; }
stopped() { [ -f "$OUT/STOP" ]; }

export HF_HOME=/home/work/Datasets/benchmarks HF_DATASETS_CACHE=/home/work/Datasets/benchmarks
export PYTHONPATH=$LM NUMEXPR_MAX_THREADS=64 TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ALPHA/.env" 2>/dev/null | cut -d= -f2 || true)

log "grid=$GRID"; log "base=$BASE"; log "sft=$SFT"
[ -d "$BASE" ] || { log "❌ base 없음: $BASE"; exit 1; }
[ -d "$SFT" ] || { log "❌ sft 없음: $SFT"; exit 1; }

# ── ① 대기 ────────────────────────────────────────────────────────────────
wait_idle() {
    while :; do
        stopped && { log "STOP 감지 — 대기 중 종료"; exit 0; }
        procs=$(pgrep -f "[r]un_suite.sh|[r]un_tier[12].sh|[r]un_swe.sh|[r]un_terminal.sh|[e]val_wrapper.py|[e]val_new_ckpt.sh|alpha_serve_venv/bin/vllm" | wc -l)
        maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        if [ "$procs" -eq 0 ] && [ "${maxmem:-99999}" -lt 2000 ]; then
            log "sub1 유휴 확인 (procs=0, max GPU mem ${maxmem}MiB)"; return 0
        fi
        log "대기: 벤치/fleet 프로세스 $procs 개, max GPU mem ${maxmem}MiB"
        sleep 120
    done
}

# ── ② 프로파일 ────────────────────────────────────────────────────────────
profile_dir() {  # $1=model(sft|base) $2=prof(ext|yarn2|yarn4) → 경로 출력
    echo "$OUT/profiles/${1}_${2}"
}
make_profile() {
    local model=$1 prof=$2 src dir
    [ "$model" = sft ] && src=$SFT || src=$BASE
    dir=$(profile_dir "$model" "$prof")
    if [ -f "$dir/.lc_profile.json" ]; then
        python3 "$ALPHA/tools/set_long_context_config.py" --check "$dir" >/dev/null && { echo "$dir"; return 0; }
    fi
    rm -rf "$dir"
    case $prof in
        ext)   python3 "$ALPHA/tools/set_long_context_config.py" "$src" --out "$dir" --max-pos $MAXLEN ;;
        yarn2) python3 "$ALPHA/tools/set_long_context_config.py" "$src" --out "$dir" --max-pos $MAXLEN --yarn-factor 2 ;;
        yarn4) python3 "$ALPHA/tools/set_long_context_config.py" "$src" --out "$dir" --max-pos $MAXLEN --yarn-factor 4 --yarn-original 131072 ;;
        *) log "❌ 알 수 없는 프로파일 $prof"; return 1 ;;
    esac >&2 || return 1
    echo "$dir"
}

# ── ③ fleet ───────────────────────────────────────────────────────────────
fleet_up() {  # $1=ckpt dir
    bash "$EVAL/stop_fleet.sh" "$GPUS" >/dev/null 2>&1 || true
    sleep 5
    ( export PIP_CONSTRAINT= TOOLS=0 GPUS="$GPUS"
      setsid bash "$EVAL/serve_fleet.sh" "$1" "$MAXLEN" "$NGPU" "$PROXY" > "$OUT/fleet_$(basename "$1").log" 2>&1 < /dev/null & )
    for i in $(seq 1 90); do   # 최대 30분 — 524288 KV 프로파일링·cudagraph 캡처는 262144 보다 느리다
        c=0
        for g in $(echo "$GPUS" | tr ',' ' '); do
            [ "$(curl -s -o /dev/null -w %{http_code} --max-time 3 http://localhost:$((8000+g))/v1/models)" = "200" ] && c=$((c+1))
        done
        [ "$c" -eq "$NGPU" ] && { log "fleet 준비 $c/$NGPU (max_len=$MAXLEN)"; return 0; }
        # 서버가 죽었으면 일찍 포기
        [ "$(pgrep -f 'alpha_serve_venv/bin/vllm' | wc -l)" -eq 0 ] && [ $i -gt 3 ] && { log "❌ vLLM 프로세스 소멸 — $OUT/fleet_$(basename "$1").log 확인"; return 1; }
        sleep 20
    done
    log "❌ fleet 준비 실패 (30분)"; return 1
}
fleet_down() { bash "$EVAL/stop_fleet.sh" "$GPUS" >/dev/null 2>&1 || true; }

run_lm_eval() {  # $1=tasks $2=maxlen $3=conc $4=outdir [$5=extra args]
    python3 "$ALPHA/scripts/eval_wrapper.py" \
        --model local-chat-completions \
        --model_args "base_url=${BURL}/chat/completions,model=alpha,num_concurrent=$3,timeout=7200,max_retries=10,tokenized_requests=False,max_length=$2" \
        --apply_chat_template --tasks "$1" --include_path "$EVAL/tasks" \
        --output_path "$4" --log_samples --seed 1234 ${5:-}
}

run_cell() {  # $1=model $2=prof
    local model=$1 prof=$2 tag dir cell
    tag="${model}_${prof}"; cell="$OUT/$tag"; mkdir -p "$cell"
    if [ -f "$cell/.done" ]; then log "[$tag] 완료본 있음 — 건너뜀"; return 0; fi
    dir=$(make_profile "$model" "$prof") || { log "[$tag] ❌ 프로파일 실패"; return 1; }
    log "[$tag] 프로파일 $dir"
    fleet_up "$dir" || { fleet_down; return 1; }

    # 게이트. SFT 는 G1~G3 필수. base(사전학습 변환본)는 chat 학습이 없어 G2/G3 가 설계상
    # 흔들릴 수 있으므로 기록만 한다 — RULER 채점은 부분문자열 일치라 장황한 답도 채점된다.
    python3 "$EVAL/check_gates.py" --hf-dir "$dir" --base-url "$BURL" \
        --max-tokens $([ "$model" = sft ] && echo 32768 || echo 2048) > "$cell/gates.log" 2>&1
    grc=$?
    if [ $grc -ne 0 ]; then
        if [ "$model" = sft ]; then log "[$tag] ❌ 게이트 실패 — $cell/gates.log"; fleet_down; return 1; fi
        log "[$tag] ⚠️ base 게이트 비통과(기록만): $(grep -E 'FAIL|실패' "$cell/gates.log" | head -2 | tr '\n' ' ')"
    else
        log "[$tag] 게이트 통과"
    fi

    log "[$tag] RULER 512k 시작 (4태스크 × 4구간 × 20)"
    run_lm_eval "ruler_niah_single_1_512k,ruler_niah_single_2_512k,ruler_niah_multikey_1_512k,ruler_niah_multivalue_512k" \
        "$MAXLEN" "$CONC_LONG" "$cell" > "$cell/ruler.log" 2>&1 || { log "[$tag] ❌ RULER 실패 — $cell/ruler.log"; fleet_down; return 1; }
    log "[$tag] RULER 완료"

    if [ "$model" = sft ]; then
        log "[$tag] 단문 회귀 표본 (ifeval·gpqa × $T1_LIMIT, 프로파일 fleet 그대로)"
        run_lm_eval "ifeval_aa,gpqa_diamond_aa" 40960 "$CONC_SHORT" "$cell" "--limit $T1_LIMIT" > "$cell/t1_subset.log" 2>&1 \
            || log "[$tag] ⚠️ 단문 표본 실패 — $cell/t1_subset.log (RULER 결과는 유효)"
    fi
    fleet_down
    date > "$cell/.done"
    log "[$tag] 셀 완료"
}

wait_idle
for c in $GRID; do
    stopped && { log "STOP 감지 — 그리드 중단"; break; }
    run_cell "${c%%:*}" "${c##*:}" || log "셀 $c 실패 — 다음 셀로"
done
fleet_down
python3 "$ALPHA/scripts/lc512k_summarize.py" "$OUT" | tee "$OUT/summary.log"
date > "$OUT/DONE"
log "그리드 종료 → $OUT/SUMMARY.md"
