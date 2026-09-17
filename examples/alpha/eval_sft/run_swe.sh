#!/bin/bash
# run_swe.sh — SWE-bench Verified (에이전틱). mini-swe-agent(gpu06 컨테이너) → swebench 채점.
#
# 규약 (docs/SFT_BENCHMARKS.md §3.4·§7):
#   - 생성 temp 1.0 / top_p 0.95 (Nemotron 3 Ultra 동일). skip_special_tokens false.
#   - **반복 1회 (pass@1 단일 시도)** — SWE-bench 리더보드 규약이다:
#     "Your system submits 1 prediction per task instance" (submission checklist).
#     여러 번 시도해 고르려면 SWE-bench 테스트를 쓰지 않는 **독립 선택기**가 있어야 하고
#     best@k 로 따로 표기해야 한다. 우리는 그런 선택기가 없으므로 1회가 맞다.
#     Nemotron 3 Ultra 의 num_repeats=3 은 분산 추정을 위한 **런 평균**이지 리더보드
#     제출 형식이 아니다. 분산이 필요하면 RUN_TAG 를 바꿔 전체를 다시 돌린다.
#   - **전량 실행이 기본** — 부분 표본 결과는 프론티어 규약상 보고 대상이 아니다
#     (NVIDIA 재현 문서: "Never report sub-sampled / limited runs").
#     N 을 지정해 줄이면 결과 JSON 에 subsampled=true 가 박혀 집계에서 무효 처리된다.
#   - 투입 전 A1~A3 게이트 통과 필수 (check_agentic_gates.py).
#
# 전제: **TOOLS=1 로 서빙된 fleet** (mini-swe-agent litellm 이 tool_choice=auto 전송) +
#       sub1→컨테이너 역터널(컨테이너:8199 → sub1:8100).
#
# 사용: bash eval_sft/run_swe.sh <RUN_NAME> [N_INSTANCES] [WORKERS]   (SWE_RESUME=1 이면 완료분 건너뜀)
#   N_INSTANCES: 0 또는 미지정 = 전량(500). 양수면 부분 표본(무효 표시).
set -uo pipefail
RUN_NAME="${1:?run name}"; N="${2:-0}"; W="${3:-12}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SSHC="/home/work/vidsearch/.ssh-keys/config"
BASE_URL="${BASE_URL:-http://localhost:8100/v1}"
RID="alpha_$(echo "$RUN_NAME" | md5sum | cut -c1-10)"
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"
HFTOK=$(grep -E '^HF_TOKEN=' "$HERE/../.env" 2>/dev/null | cut -d= -f2 || true)

# ---- 게이트 (SKIP_GATES=1 은 이미 통과한 실행의 재개 전용) ----
if [ "${SKIP_GATES:-0}" != "1" ]; then
  python3 "$HERE/check_agentic_gates.py" --base-url "$BASE_URL" --min-disk-gb "${MIN_DISK_GB:-300}" || {
    echo "[swe] ❌ 게이트 실패 — 중단. 이 상태의 0점은 모델 실패와 구분되지 않는다."; exit 1; }
fi

SLICE=""; SUBSAMPLED=false
if [ "$N" -gt 0 ] 2>/dev/null; then
  SLICE="--shuffle --slice 0:$N"; SUBSAMPLED=true
  echo "[swe] ⚠️ 부분 표본 $N 건 — 결과에 subsampled=true 가 박힌다(집계 무효)"
else
  echo "[swe] 전량(SWE-bench Verified 500)"
fi

# ── 추론 분리·복원 (2026-09-14, KNOWN_ISSUES 09-14) ────────────────────────
# 에이전틱 fleet 가 reasoning 파서 없이 뜨면 vLLM 0.25.1 qwen3_xml 이 도구호출 턴의 </think> 를
# 소비한다 — 보존된 궤적 전수에서 0/148,836턴. 이력이 <think></think>+추론문으로 재렌더됐다.
# 수정은 둘이 함께 필요하다: (1) fleet 를 REASONING_PARSER=nemotron_v3 로 띄워 추론을 필드로 분리
# (run_suite 에이전틱 단계, 게이트 A5 가 확인), (2) mini-swe-agent 는 reasoning 필드를 이력에
# 재전송하지 않으므로 tau_proxy 가 캐시했다가 다음 요청 이력에 되돌린다(τ³ 와 같은 방식).
#
# 프록시는 **컨테이너 안, 하니스 옆**에 둔다: mini-swe-agent → :8110 프록시 → :8199 역터널 → sub1.
# 터널은 건드리지 않는다. TB-2 도 같은 방식으로 자기 프록시(:8111)를 둔다 — 학습 데이터(NTC)가 restore 라
# 이력 추론을 되돌려야 하고, harbor 는 vLLM 의 `reasoning` 키를 못 읽는다(run_terminal_tb2.sh 헤더).
#
# SWE_THINK: restore(기본) = 이력에 추론 복원(학습 형식) · strip = 분리만, 복원 안 함(--no-reattach).
# 복원이 낫다고 가정하지 말 것 — τ airline 5과제에서 ON 0/5 vs OFF 3/5 였다(n=5). ON/OFF 를 같이 잰다.
# 복원은 매 턴 프롬프트를 추론만큼 키우므로 긴 궤적에서 262144 창 초과가 늘 수 있다.
SWE_THINK="${SWE_THINK:-restore}"
case "$SWE_THINK" in
  restore) REATTACH="" ;;
  strip)   REATTACH="--no-reattach" ;;
  *) echo "[swe] ❌ SWE_THINK=$SWE_THINK — restore 또는 strip"; exit 1 ;;
esac
export SWE_THINK   # 결과 파서(파이썬)가 모드를 기록하려면 환경변수여야 한다
RAW_C="/opt/swebench/preds_${RUN_NAME}/proxy_raw"
# 스모크 전용 스텝 한도. 정식 실행에서는 비워 둔다(기본 250) — 한도를 줄인 결과는 비교 대상이 아니며,
# 스모크는 N>0 이라 이미 subsampled=true → 무효로 집계된다.
STEP_ARG=""; [ -n "${SWE_STEP_LIMIT:-}" ] && STEP_ARG="-c agent.step_limit=$SWE_STEP_LIMIT"
# SWE_RESUME=1: 같은 RUN_NAME 의 preds.json 에 있는 인스턴스를 건너뛰고 나머지만 돈다(mini-swe-agent 기본 동작; 우리는 평소
# --redo-existing 으로 매번 전량). fleet 설정 변경(2026-09-17 prefix caching·W=96)으로 중간 재기동할 때 쓴다 — 인스턴스는
# 서로 독립이고 생성 파라미터가 같으므로 완료분을 버릴 이유가 없다. 진행 중이던 인스턴스는 preds.json 에 없어 다시 돈다.
REDO="--redo-existing"; [ "${SWE_RESUME:-0}" = "1" ] && REDO=""
# 컨테이너의 프록시는 항상 리포 버전으로 덮는다 (표준 라이브러리만 쓴다).
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "cat > /opt/swebench/tau_proxy.py" < "$HERE/tau_proxy.py" || {
  echo "[swe] ❌ tau_proxy.py 복사 실패"; exit 1; }

echo "[swe] 예측 생성 (mini-swe-agent, W=$W workers, temp 1.0, think=$SWE_THINK)"
# 원격 스크립트는 stdin 으로 넘긴다 — 인라인 ssh "..." 는 따옴표가 중첩돼 원격 변수가 로컬에서 먹힌다.
ssh -F "$SSHC" -o BatchMode=yes alpha-eval 'bash -s' <<EOF
  export HF_TOKEN=$HFTOK OPENAI_API_KEY=dummy OPENAI_API_BASE=http://localhost:8110/v1
  # litellm 은 미등록 모델의 비용을 계산하다 RuntimeError 로 죽는다 (2026-08-30 실측).
  export LITELLM_MODEL_REGISTRY_PATH=/opt/swebench/alpha_model_registry.json
  export MSWEA_COST_TRACKING=ignore_errors
  cd /opt/swebench
  mkdir -p $RAW_C
  pkill -f '[t]au_proxy.py --port 8110' 2>/dev/null; sleep 1
  # --no-greeting: mini-swe-agent 는 합성 인사가 없어 첫 턴 miss 를 miss 로 세야 miss_rate 가 정직하다.
  # --max-entries 100000: SWE 는 총 턴 ~7.5만 — 기본 2만이면 살아 있는 궤적 초기 턴이 밀려나 miss.
  setsid python3 /opt/swebench/tau_proxy.py --port 8110 --upstream http://127.0.0.1:8199 \
    --no-greeting --max-entries 100000 $REATTACH \
    --stats-file $RAW_C/proxy_stats.json --dump-dir $RAW_C > $RAW_C/proxy.log 2>&1 < /dev/null &
  PX=\$!
  for i in \$(seq 1 30); do curl -s -m 2 -o /dev/null http://localhost:8110/stats && break; sleep 1; done
  curl -s -m 2 -o /dev/null http://localhost:8110/stats || { echo "[swe] ❌ tau_proxy 기동 실패"; tail -5 $RAW_C/proxy.log; exit 1; }
  ./venv/bin/mini-extra swebench --subset SWE-bench/SWE-bench_Verified --split test \
    $SLICE --workers $W $REDO \
    -m openai/alpha -c swebench.yaml \
    -c model.model_kwargs.api_base=http://localhost:8110/v1 \
    -c model.model_kwargs.temperature=1.0 \
    -c model.model_kwargs.top_p=0.95 \
    -c model.model_kwargs.max_tokens=${SWE_MAX_TOKENS:-32768} $STEP_ARG \
    -o /opt/swebench/preds_${RUN_NAME} 2>&1 | tail -8
  kill \$PX 2>/dev/null; sleep 2   # SIGTERM → 프록시가 stats 를 flush 하고 종료
EOF
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "cat $RAW_C/proxy_stats.json 2>/dev/null" \
  > "$OUT/swe_proxy_stats.json" 2>/dev/null || true

echo "[swe] 채점 (swebench eval)"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
  export HF_TOKEN=$HFTOK; cd /opt/swebench
  # mini-swe-agent 는 preds.json (단수, dict) 을 쓴다 — .jsonl 이 아니다 (2026-08-30 실측).
  PREDS=\$(ls -t preds_${RUN_NAME}/preds.json preds_${RUN_NAME}/preds.jsonl 2>/dev/null | head -1)
  [ -z "\$PREDS" ] && { echo "[swe] ❌ 예측 파일 없음 — 채점 생략"; exit 0; }
  ./venv/bin/swebench eval SWE-bench/SWE-bench_Verified -p \"\$PREDS\" --run-id $RID -j $W 2>&1 | tail -10
"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval \
  "cat /opt/swebench/*$RID*.json 2>/dev/null || cat /opt/swebench/logs/run_evaluation/$RID/*/report.json 2>/dev/null" \
  > "$OUT/swe_report_raw.json" 2>/dev/null || true

python3 - "$OUT" "$SUBSAMPLED" <<'PY'
import json, sys, os
outd, sub = sys.argv[1], sys.argv[2] == "true"
raw = os.path.join(outd, "swe_report_raw.json")
resolved = total = 0
d = {}   # 리포트 로드가 실패해도 아래 게이트 집계가 NameError 로 죽지 않게 (2026-09-14 수정)
try:
    d = json.load(open(raw))
    resolved = d.get("resolved_instances", d.get("resolved", 0)) or 0
    total = d.get("total_instances", d.get("submitted_instances", 0)) or 0
except Exception:
    pass
acc = resolved / total if total else 0.0
res = {"resolved,none": acc}
# 부분 표본은 no_answer 를 1.0 으로 박아 집계기가 무효로 판정하게 한다 —
# 프론티어 규약상 부분 표본 결과는 보고 대상이 아니다.
if sub or total == 0:
    res["no_answer,none"] = 1.0
# phase-3 데이터 교정의 before/after 게이트 (KNOWN_ISSUES 2026-09-09 판정문).
# 결함 ①②는 **선언된 tools 블록을 보고 호출하는** 조건부를 못 배우게 했다 → 교정 후
# 빈 패치율과 형식 오류가 줄어야 한다. resolved 만 보면 그 변화가 안 보인다.
# 이 값들은 결과 JSON·문서에만 남긴다 — wandb 는 평가 결과만 올린다(사용자, 2026-09-01).
empty = int(d.get("empty_patch_instances", 0) or 0)
errs = int(d.get("error_instances", 0) or 0)
comp = int(d.get("completed_instances", 0) or 0)
# ── 추론 분리·복원 프록시 (2026-09-14) — τ³ 의 tau_combine 과 같은 무효 규칙 ──
px = {}
try:
    px = json.load(open(os.path.join(outd, "swe_proxy_stats.json")))
except Exception:
    pass
PX_KEYS = ("requests", "reinlined", "miss", "miss_first_assistant", "think_stripped",
           "think_from_field", "think_absent", "think_unclosed", "tool_calls", "reattach", "miss_rate", "miss_rate_cache_path", "miss_turns", "restored_turns", "miss_turn_rate", "miss_samples",
           "reasoning_field_inlined", "reasoning_field_dropped", "restored")
proxy = {k: px[k] for k in PX_KEYS if k in px}
invalid = []
if not px:
    invalid.append("프록시 통계 없음 — 추론 분리 경로를 거쳤는지 확인 불가")
else:
    if px.get("reattach") and px.get("miss_rate", 0) > 0.05:
        invalid.append(f"복원 켠 채 miss_rate {px['miss_rate']:.3f} > 0.05 — 이력 복원이 새고 있다")
    if px.get("requests", 0) and (px.get("think_stripped", 0) + px.get("think_from_field", 0)) == 0:
        invalid.append("추론 미관측 — fleet 가 reasoning 을 분리하지 않는다(REASONING_PARSER 확인)")
    # 복원 판정은 tau_proxy 의 restored(= 이력 content 에 실제로 넣은 총수, a834e48)로 한다.
    # 캐시 복원과 reasoning_content 필드 인라인을 둘 다 센다. SWE 는 mini-swe-agent(litellm)가 이력에
    # reasoning_content 를 **다시 실어 보내므로** 복원이 전부 필드 경로로 일어나고 reinlined·miss 는 0/0 —
    # 위 miss_rate 규칙은 SWE 에서 아무것도 못 잡는다. 이 두 규칙이 그 자리를 메운다.
    if "restored" in px:
        applied = px["restored"]
    elif px.get("reattach"):
        # a834e48 이전 통계: restore 에서는 reinlined 가 곧 적용 수다.
        applied = px.get("reinlined", 0) + px.get("reasoning_field_inlined", 0)
    else:
        # a834e48 이전 strip 통계: reinlined 는 **캐시 적중일 뿐 적용되지 않는다**. 합산하면 정상 strip 을
        # 누수로 오판한다(2026-09-14 재검증 strip: reinlined 66 · 이력 0/10).
        applied = px.get("reasoning_field_inlined", 0)
    multi = px.get("requests", 0) > 1
    if px.get("reattach") and multi and applied == 0:
        invalid.append("복원 켰는데 다회차 요청에서 복원 0건 — 이력에 추론이 들어가지 않았다")
    # 77040d3 이전 tau_proxy 는 필드 인라인 경로가 --no-reattach 를 무시했다 → strip 에서도 추론이 이력에
    # 들어가 restore 와 구분되지 않았다. 재발 시 ON/OFF 비교가 오염되므로 무효.
    if px.get("reattach") is False and applied > 0:
        invalid.append(f"strip 모드인데 추론이 이력에 {applied}회 삽입됨 — restore 와 구분되지 않는다")
if invalid:
    res["no_answer,none"] = 1.0
json.dump({"results": {"swe_bench_verified": res},
           "swe_detail": {"resolved": resolved, "total": total, "subsampled": sub,
                          "empty_patch": empty, "errors": errs, "completed": comp,
                          "empty_patch_rate": (empty / total if total else 0.0),
                          "resolved_given_completed": (resolved / comp if comp else 0.0),
                          "think_mode": os.environ.get("SWE_THINK", "restore"),
                          "proxy": proxy, "invalid": invalid}},
          open(os.path.join(outd, "results_swe.json"), "w"), indent=2)
mark = "  [부분표본 → 무효]" if sub else ("  [리포트 없음 → 무효]" if total == 0 else "")
if invalid:
    mark += "  [무효: " + " · ".join(invalid) + "]"
print(f"[swe] resolved {resolved}/{total} = {acc*100:.1f}%{mark}")
print(f"[swe] 게이트 — 빈패치 {empty}/{total} = {empty/max(total,1)*100:.1f}% · "
      f"평가오류 {errs} · 채점기준 적중 {resolved}/{comp} = {resolved/max(comp,1)*100:.1f}%")
if px:
    print(f"[swe] 프록시 — 모드 {os.environ.get('SWE_THINK','restore')} · 요청 {px.get('requests',0)} · "
          f"추론(필드 {px.get('think_from_field',0)} / 인라인 {px.get('think_stripped',0)} / 없음 {px.get('think_absent',0)}) · "
          f"복원 {px.get('restored','—')} (캐시적중 {px.get('reinlined',0)} · 필드 인라인 {px.get('reasoning_field_inlined',0)} / 버림 "
          f"{px.get('reasoning_field_dropped',0)}) · miss {px.get('miss',0)} ({px.get('miss_rate',0)*100:.1f}%)")
PY
echo "== SWE 완료: $OUT =="
