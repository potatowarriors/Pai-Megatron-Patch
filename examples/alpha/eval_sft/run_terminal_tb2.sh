#!/bin/bash
# run_terminal_tb2.sh — Terminal-Bench **2.1** (기본) / 2.0 (Harbor + Terminus-2).
#
# TB 2.0 → 2.1 (사용자 결정 2026-10-01: iter1200 체인은 2.0 으로 끝내고 **다음 체인부터 2.1**):
#   2.1 = harbor-framework/terminal-bench-2-1 @7131e43 의 로컬 클론 /opt/harbor/terminal-bench-2-1 (gpu06). 과제 89 동일, 28 수정
#   (이미지 재고정 10 · 지시문 11 · 테스트 10 · 에이전트 한도 2 · 검증기 한도 2; 한도 중앙값 900 s 그대로 — SFT_BENCHMARKS §3.11).
#   harbor 0.22 는 Hub id(`terminal-bench/terminal-bench-2-1`)를 모른다 → `-p <클론>/tasks` 로 준다(`--install-only` 로 schema 1.1 파싱 확인).
#   결과 키·job 접두를 버전별로 나눈다(terminal_bench_2 / terminal_bench_2_1 · tb2 / tb21) → TRACKING 열이 섞이지 않는다.
#   2.0 계열(general 600·2300·2862 · agentic 200·1200)과 2.1 수치는 비교하지 않는다. TB_VERSION=2.0 이면 종전 그대로.
# 부분 통과율 (2026-10-01): 검증기는 타임아웃 트라이얼에도 돌아 verifier/ctrf.json 에 테스트별 통과를 남긴다. 전부-통과(공식)는
#   1/712 수준이라 추이가 안 보여 트라이얼 평균 테스트 통과율·≥1개 통과 비율을 보조 키로 함께 낸다(TB_VERSION 무관).
#
# 왜 TB-1 을 대체하는가 (사용자 결정 2026-09-07, `SFT_RL_DATASETS.md` §2.9):
#   학습 데이터의 Terminus 행은 **Terminus-2 스키마**
#   (`analysis/plan/commands{keystrokes,duration}/task_complete`)인데, 구 하니스는
#   TB-1 core 0.1.1 + terminus **v1**(4필드 `state_analysis/explanation/commands/
#   is_task_complete`)이었다. 스키마는 하니스 에이전트가 정하므로 v1 하니스에 v2 형식을
#   끼워 넣을 수 없다. Ultra 공개 수치도 TB 2.0/2.1 기준이라 비교 조건이 맞지 않았다.
#   TB-1 수치는 참고치로 남기고(유효 계열이 짧아 연속성 손실 없음) 이쪽을 정본으로 쓴다.
#
# 실행 규약은 TB-1 에서 **승계**한다: 반복 8 · temp 1.0 / top_p 0.95 ·
# max_tokens 65,536 · A1~A5 게이트 · 전량 실행(부분 표본은 무효 표시).
#
# TB-1 → TB-2 플래그 대응 (2026-09-07 `harbor run --help` 실측):
#   tb run                → harbor run
#   --agent terminus      → -a terminus-2
#   -k api_base=…         → --ak api_base=…      (--ak 값은 JSON 리터럴로 파싱된다)
#   --dataset core==0.1.1 → -d terminal-bench@2.0   (80 tasks → **89 tasks**)
#   --n-attempts K        → -k K                  ⚠️ TB-1 의 -k(agent kwarg)와 의미가 다르다
#   --n-concurrent W      → -n W
#   --n-tasks N           → -l N
#   --run-id / --output-path → --job-name / -o
#
# parser_name 은 **json** 이다 — 학습 데이터가 Terminus-2 JSON 스키마다.
# (terminus-2 는 json/xml 두 파서를 갖고 각각 프롬프트 템플릿이 다르다.)
#
# 사용: bash eval_sft/run_terminal_tb2.sh <RUN_NAME> [N_TASKS] [W]
#   N_TASKS: 0/미지정 = 전량(89). 양수면 부분 표본(무효 표시).
#   env TB2_THINK=restore(기본)|strip — 이전 턴 추론 보존 여부(아래 절) · TB2_INCLUDE=<glob> — 스모크용 과제 필터(부분 표본)
#   env TB_VERSION=2.1(기본)|2.0 — 과제 세트 버전(위 헤더)
set -uo pipefail
RUN_NAME="${1:?run name}"; N="${2:-0}"; W="${3:-8}"
HERE="$(cd "$(dirname "$0")" && pwd)"; SSHC="/home/work/vidsearch/.ssh-keys/config"
BASE_URL="${BASE_URL:-http://localhost:8100/v1}"
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"
# ── 과제 세트 버전 (헤더 참조) ──
TB_VERSION="${TB_VERSION:-2.1}"
case "$TB_VERSION" in
  2.1) HARBOR_DS="-p /opt/harbor/terminal-bench-2-1/tasks"; DS_LABEL="terminal-bench-2.1 (harbor-framework/terminal-bench-2-1@7131e43 local clone)"
       RIDP="tb21"; RES_KEY="terminal_bench_2_1"
       ssh -F "$SSHC" -o BatchMode=yes alpha-eval "test -d /opt/harbor/terminal-bench-2-1/tasks && [ \$(ls -d /opt/harbor/terminal-bench-2-1/tasks/*/ | wc -l) -eq 89 ]" || {
         echo "[term2] ❌ gpu06 에 2.1 클론(/opt/harbor/terminal-bench-2-1/tasks, 89 과제)이 없다 — EVAL_DOCKER_NODE.md 재구축 절 참조"; exit 1; } ;;
  2.0) HARBOR_DS="-d terminal-bench@2.0"; DS_LABEL="terminal-bench@2.0"; RIDP="tb2"; RES_KEY="terminal_bench_2" ;;
  *) echo "[term2] ❌ TB_VERSION=$TB_VERSION — 2.0 또는 2.1"; exit 1 ;;
esac
# compose 프로젝트명 길이 제한 때문에 job 이름은 짧게 (2026-08-30 사고, 71a1d84)
RID="${RIDP}$(echo "$RUN_NAME" | md5sum | cut -c1-8)"

if [ "${SKIP_GATES:-0}" != "1" ]; then
  # A5 는 required 다. terminus-2 는 요청에 tools 를 안 보내지만, **TOOLS=1 fleet 에서 reasoning 파서가
  # 없으면 도구 미선언 요청도 </think> 를 잃는다** — 2026-09-14 실측 6/6(추론문 975자가 마커 없이 JSON 앞에
  # 붙음). 파서 엔진은 요청의 tools 가 아니라 서버 플래그로 켜진다. A5 PASS = fleet 가 추론을 분리한다.
  python3 "$HERE/check_agentic_gates.py" --base-url "$BASE_URL" --min-disk-gb "${MIN_DISK_GB:-150}" || {
    echo "[term2] ❌ 게이트 실패 — 중단."; exit 1; }
fi

NTASKS=""; SUBSAMPLED=false
if [ -n "${TB2_INCLUDE:-}" ]; then
  # 스모크용 과제 이름 필터(harbor -i, glob). 전량이 아니므로 부분 표본으로 표시한다.
  NTASKS="-i ${TB2_INCLUDE}"; SUBSAMPLED=true
  echo "[term2] ⚠️ 과제 필터 '${TB2_INCLUDE}' — 결과에 subsampled=true (집계 무효)"
elif [ "$N" -gt 0 ] 2>/dev/null; then
  NTASKS="-l $N"; SUBSAMPLED=true
  echo "[term2] ⚠️ 부분 표본 $N 태스크 — 결과에 subsampled=true (집계 무효)"
else
  echo "[term2] 전량 ($DS_LABEL, 89 tasks)"
fi

# ── 이전 턴 추론 보존 (2026-09-14) ─────────────────────────────────────────────
# 학습 데이터 nvidia/Nemotron-Terminal-Corpus(ntc_v1_*)는 `--keep-history-think` 로 구웠다(bins
# data.stats.json keep_history_think=True, 디코드 실측 마지막 user 이전 assistant 턴 56/56 think 보존).
# terminus-2 는 도구 선언 없이 JSON 으로 명령을 주고받고 **터미널 출력을 user 메시지로** 보낸다 → 템플릿(28행)이
# 비도구 시나리오로 보고 마지막 user 이전 think 를 자른다(strip). 학습은 restore, 평가는 strip 이었다.
#
# restore 에는 **세 가지가 모두** 필요하다 — 하나라도 빠지면 조용히 strip 으로 돈다(2026-09-14 실측):
#   tau_proxy                        vLLM 0.25.1 은 추론을 `reasoning` 키로 돌려주는데 harbor(lite_llm.py 428행)는
#                                    `reasoning_content` 만 읽는다 → 추론을 못 받는다. 프록시가 응답에서
#                                    `reasoning` 을 `reasoning_content` 로 옮겨 적는다(SWE 도 이 덕에 받았다).
#   interleaved_thinking=true        terminus-2 가 받은 reasoning_content 를 이력 assistant 메시지에 붙인다
#   truncate_history_thinking=false  템플릿이 비도구 시나리오에서 자르지 않게 한다(llm_call_kwargs.extra_body)
# 프록시 없이 앞의 둘만 켠 스모크: 이력 assistant 2턴 중 reasoning_content 0, 재전송 prompt_tokens 차이 0.
# ⚠️ 렌더 검증은 /tokenize 로 하지 말 것 — /tokenize 는 reasoning_content 를 버린다(실측). /v1/chat/completions 의
#    usage.prompt_tokens 로 잰다.
#
# TB2_THINK: restore(기본, NTC 학습 조건) · strip(이전 추론 없음 — 비교용·구 체크포인트용)
TB2_THINK="${TB2_THINK:-restore}"
case "$TB2_THINK" in
  restore) THINK_AK="--ak interleaved_thinking=true"; REATTACH=""
           EXTRA_BODY=',"extra_body":{"chat_template_kwargs":{"truncate_history_thinking":false}}' ;;
  strip)   THINK_AK=""; EXTRA_BODY=""; REATTACH="--no-reattach" ;;
  *) echo "[term2] ❌ TB2_THINK=$TB2_THINK — restore 또는 strip"; exit 1 ;;
esac
export TB2_THINK

echo "[term2] harbor run ($DS_LABEL, terminus-2, parser=json, W=$W, k=${TERM_REPEATS:-8}, temp 1.0, think=$TB2_THINK)"
RAW_C="/opt/harbor/proxy_raw/$RID"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "rm -rf /opt/harbor/jobs/$RID $RAW_C" 2>/dev/null || true
# 컨테이너의 프록시는 항상 리포 버전으로 덮는다 (표준 라이브러리만 쓴다).
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "mkdir -p /opt/harbor && cat > /opt/harbor/tau_proxy.py" < "$HERE/tau_proxy.py" || {
  echo "[term2] ❌ tau_proxy.py 복사 실패"; exit 1; }
# 원격 스크립트는 stdin 으로 넘긴다 — 인라인 ssh "..." 는 따옴표가 중첩돼 JSON·원격 변수가 깨진다.
ssh -F "$SSHC" -o BatchMode=yes alpha-eval 'bash -s' <<EOF
  export HOME=/opt/harbor
  export OPENAI_API_KEY=dummy OPENAI_API_BASE=http://localhost:8111/v1
  # terminus-2 도 litellm 을 쓴다 — 미등록 모델 비용 계산 실패 방지 (run_swe.sh 와 동일 사유)
  export LITELLM_MODEL_REGISTRY_PATH=/opt/terminalbench/alpha_model_registry.json
  cd /opt/harbor && mkdir -p $RAW_C
  # 포트 8111: run_swe.sh 의 프록시(8110)와 같은 컨테이너에서 겹쳐도 서로 죽이지 않게 분리한다.
  pkill -f '[t]au_proxy.py --port 8111' 2>/dev/null; sleep 1
  # --no-greeting: 합성 인사가 없는 하니스. --max-entries: 89 과제 × 8 시도의 긴 궤적.
  # --timeout 7200: 업스트림 대기 상한. 65K 토큰 생성이 기본 1800초를 넘으면 프록시가 502 로 끊는다 —
  #   프록시가 없던 시절에는 없던 실패이므로 과제 타임아웃보다 먼저 걸리지 않게 둔다.
  setsid python3 /opt/harbor/tau_proxy.py --port 8111 --upstream http://127.0.0.1:8199 \
    --no-greeting --max-entries 100000 --timeout 7200 $REATTACH \
    --stats-file $RAW_C/proxy_stats.json --dump-dir $RAW_C > $RAW_C/proxy.log 2>&1 < /dev/null &
  PX=\$!
  for i in \$(seq 1 30); do curl -s -m 2 -o /dev/null http://localhost:8111/stats && break; sleep 1; done
  curl -s -m 2 -o /dev/null http://localhost:8111/stats || { echo "[term2] ❌ tau_proxy 기동 실패"; tail -5 $RAW_C/proxy.log; exit 1; }
  ./venv/bin/harbor run $HARBOR_DS -a terminus-2 -m openai/alpha \
    --ak api_base=http://localhost:8111/v1 \
    --ak temperature=1.0 \
    --ak parser_name=json \
    --ak 'llm_call_kwargs={"top_p":0.95,"max_tokens":${TERM_MAX_TOKENS:-65536}${EXTRA_BODY}}' $THINK_AK \
    $NTASKS -n $W -k ${TERM_REPEATS:-8} \
    -o /opt/harbor/jobs --job-name $RID -y -q 2>&1 | tail -20
  kill \$PX 2>/dev/null; sleep 2   # SIGTERM → 프록시가 stats 를 flush 하고 종료
EOF
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "cat $RAW_C/proxy_stats.json 2>/dev/null" \
  > "$OUT/terminal_proxy_stats.json" 2>/dev/null || true
ssh -F "$SSHC" -o BatchMode=yes alpha-eval \
  "cat /opt/harbor/jobs/$RID/result.json 2>/dev/null" > "$OUT/terminal_raw.json" 2>/dev/null || true

# ── 명령 추출률 (phase-3 before/after 게이트) ────────────────────────────
# result.json 에는 보상만 있다. "모델이 Terminus-2 형식을 지켜 명령이 뽑혔는가" 는
# 궤적에서만 나온다 — 2026-09-07 스모크에서 에이전트 응답 8개 중 4개만 추출됐다.
# KNOWN_ISSUES 09-09: 도구 학습 토큰의 43.6%가 <think></think> 타깃인데 평가는
# thinking ON 이다. 교정 후 이 비율이 오르는지가 phase-3b 판단 근거가 된다.
# 원격 스크립트는 **stdin 으로 넘긴다** — ssh "..." 인라인은 따옴표가 중첩돼 깨진다.
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "JOB=/opt/harbor/jobs/$RID bash -s" > "$OUT/terminal_extract.json" 2>/dev/null <<'REMOTE' || true
python3 - "$JOB" <<'PY2'
import json, os, sys, glob
job = sys.argv[1]
steps = withcmd = withreason = reasononly = 0
for f in glob.glob(os.path.join(job, "*", "agent", "trajectory.json")):
    try:
        t = json.load(open(f))
    except Exception:
        continue
    for st in (t.get("steps") or []):
        if st.get("source") != "agent":
            continue
        steps += 1
        if st.get("tool_calls"):
            withcmd += 1
        # restore 의 전제: fleet 가 추론을 필드로 분리해야 terminus-2 가 이력에 되돌려 보낼 수 있다.
        if str(st.get("reasoning_content") or "").strip():
            withreason += 1
            # 추론만 있고 답변이 빈 스텝 = </think> 를 안 닫고 JSON 을 쓴 턴(파서가 전부 추론으로 분류).
            # 2026-09-14 스모크에서 restore·strip 모두 관측 — 하니스는 "No valid JSON found" 로 되받는다.
            if not str(st.get("message") or "").strip():
                reasononly += 1
out = {"agent_steps": steps, "steps_with_commands": withcmd,
       "extraction_rate": (withcmd / steps if steps else 0.0),
       "steps_with_reasoning": withreason, "steps_reasoning_only": reasononly}
# ── 부분 통과율 (2026-10-01) — 검증기는 AgentTimeout 트라이얼에도 돈다. verifier/ctrf.json 의 summary.passed/tests 를 센다.
#    ctrf 가 없는 트라이얼(RuntimeError·VerifierTimeout 등)은 0 으로 센다(보수적). 수치는 트라이얼 평균(과제 가중 아님).
n_tr = n_to = n_any = n_full = n_noctrf = 0
fr, fr_ok, fr_to = [], [], []
for f in glob.glob(os.path.join(job, "*", "result.json")):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    n_tr += 1
    to = ((r.get("exception_info") or {}).get("exception_type") == "AgentTimeoutError")
    n_to += to
    try:
        sm = json.load(open(os.path.join(os.path.dirname(f), "verifier", "ctrf.json")))["results"]["summary"]
        p, t = int(sm.get("passed") or 0), int(sm.get("tests") or 0)
    except Exception:
        n_noctrf += 1
        p, t = 0, 0
    x = (p / t) if t else 0.0
    fr.append(x)
    (fr_to if to else fr_ok).append(x)
    n_any += (p > 0)
    n_full += (t > 0 and p == t)
mean = lambda v: (sum(v) / len(v)) if v else 0.0  # noqa: E731
out.update({"trials_scored": n_tr, "timeout_rate": (n_to / n_tr if n_tr else 0.0),
            "partial_pass_mean": mean(fr), "any_pass_rate": (n_any / n_tr if n_tr else 0.0),
            "full_pass_trials": n_full, "partial_pass_mean_completed": mean(fr_ok), "n_completed": len(fr_ok),
            "partial_pass_mean_timeout": mean(fr_to), "no_ctrf": n_noctrf})
print(json.dumps(out))
PY2
REMOTE

python3 - "$OUT" "$SUBSAMPLED" "$RES_KEY" "$DS_LABEL" <<'PY'
import json, sys, os
outd, sub = sys.argv[1], sys.argv[2] == "true"
res_key, ds_label = sys.argv[3], sys.argv[4]
raw = os.path.join(outd, "terminal_raw.json")
acc = 0.0; ntr = nerr = 0; rewards = {}; exc = {}
try:
    d = json.load(open(raw))
    ev = (d.get("stats") or {}).get("evals") or {}
    # 키는 "<agent>__<dataset>" — 하나뿐이다.
    for _, v in ev.items():
        ntr = int(v.get("n_trials") or 0)
        nerr = int(v.get("n_errors") or 0)
        m = v.get("metrics") or [{}]
        acc = float(m[0].get("mean") or 0.0)
        rewards = {k: len(x) for k, x in ((v.get("reward_stats") or {}).get("reward") or {}).items()}
        exc = {k: (len(x) if hasattr(x, "__len__") else x) for k, x in (v.get("exception_stats") or {}).items()}
        break
except Exception as e:  # noqa: BLE001
    print(f"[term2] ⚠️ 결과 파싱 실패: {e}")

ext = {}
try:
    ext = json.load(open(os.path.join(outd, "terminal_extract.json")))
except Exception:  # noqa: BLE001
    pass
mode = os.environ.get("TB2_THINK", "restore")
# ── 추론 분리·복원 프록시 (2026-09-14) — run_swe.sh 와 같은 무효 규칙 ──
px = {}
try:
    px = json.load(open(os.path.join(outd, "terminal_proxy_stats.json")))
except Exception:  # noqa: BLE001
    pass
# think_unclosed_stop(b87b258) = </think> 미종결로 답변·도구호출 없이 끝난 응답 — 궤적 쪽 steps_reasoning_only 와 같은 사건을
# 프록시에서 센다(finish=length 는 think_unclosed 로 따로). 둘이 크게 어긋나면 한쪽 집계가 틀린 것이다.
PX_KEYS = ("requests", "reinlined", "restored", "miss", "think_stripped", "think_from_field", "think_absent",
           "think_unclosed", "think_unclosed_stop", "reasoning_field_inlined", "reasoning_field_dropped", "reattach",
           "miss_rate", "miss_rate_cache_path", "miss_turns", "restored_turns", "miss_turn_rate", "miss_samples", "finish_length", "upstream_errors")
proxy = {k: px[k] for k in PX_KEYS if k in px}
invalid = []
if not px:
    invalid.append("프록시 통계 없음 — 추론 분리·복원 경로를 거쳤는지 확인 불가")
else:
    if px.get("requests", 0) and (px.get("think_stripped", 0) + px.get("think_from_field", 0)) == 0:
        invalid.append("추론 미관측 — fleet 가 reasoning 을 분리하지 않는다(REASONING_PARSER 확인)")
    if px.get("reattach") and px.get("requests", 0) > 1 and px.get("restored", 0) == 0:
        invalid.append("restore 인데 다회차 요청에서 복원 0건 — 이력에 추론이 들어가지 않았다")
    if px.get("reattach") and px.get("miss_rate", 0) > 0.05:
        invalid.append(f"restore 인데 miss_rate {px['miss_rate']:.3f} > 0.05 — 이력 복원이 새고 있다")
    if px.get("reattach") is False and px.get("restored", 0) > 0:
        invalid.append(f"strip 인데 추론이 이력에 {px['restored']}회 삽입됨 — restore 와 구분되지 않는다")
if mode == "restore" and ext.get("agent_steps", 0) > 1 and ext.get("steps_with_reasoning", 0) == 0:
    invalid.append("restore 인데 하니스가 받은 추론 0 스텝 — harbor 는 reasoning_content 만 읽는다(프록시 경로 확인)")
pp = ext.get("partial_pass_mean"); ap = ext.get("any_pass_rate"); tr = ext.get("timeout_rate")
main_res = {"resolved,none": acc}
if pp is not None:
    main_res.update({"partial_pass,none": pp, "any_pass,none": ap, "timeout_rate,none": tr})
res = {"results": {res_key: main_res},
       "terminal_detail": {"harness": f"{ds_label} + harbor 0.22 + terminus-2", "tb_version": os.environ.get("TB_VERSION", "2.1"),
                           "n_trials": ntr, "n_errors": nerr,
                           "reward_counts": rewards, "exception_stats": exc,
                           **{k: ext[k] for k in ext},
                           "think_mode": mode, "proxy": proxy, "invalid": invalid,
                           "subsampled": sub}}
# 부분 통과율은 TRACKING 열을 위해 보조 키로도 낸다(bench_registry: *_partial, *_anypass). wandb 는 main 키의 지표로 올라간다.
if pp is not None:
    res["results"][res_key + "_partial"] = {"partial_pass,none": pp}
    res["results"][res_key + "_anypass"] = {"any_pass,none": ap}
if invalid:
    for k in list(res["results"]):
        res["results"][k]["no_answer,none"] = 1.0
json.dump(res, open(os.path.join(outd, "results_terminal.json"), "w"), ensure_ascii=False, indent=2)
print(f"[term2] accuracy {acc*100:.1f}%  trials={ntr} errors={nerr}  ({ds_label})")
if pp is not None:
    print(f"[term2] 부분 통과 — 트라이얼 평균 테스트 통과율 {pp*100:.1f}% · ≥1개 통과 {ap*100:.1f}% · 타임아웃 {tr*100:.1f}% · "
          f"전부 통과 {ext.get('full_pass_trials')} · 시간 내 완료 {ext.get('n_completed')}(평균 {ext.get('partial_pass_mean_completed',0)*100:.1f}%) · ctrf 없음 {ext.get('no_ctrf')}")
if ext.get("agent_steps"):
    print(f"[term2] 게이트 — 명령 추출 {ext['steps_with_commands']}/{ext['agent_steps']} "
          f"= {ext['extraction_rate']*100:.1f}%")
    print(f"[term2] 추론 — 모드 {mode} · 하니스가 받은 추론 {ext.get('steps_with_reasoning', 0)}/{ext['agent_steps']} 스텝 · "
          f"추론만 있고 답변 빈 스텝 {ext.get('steps_reasoning_only', 0)}")
if px:
    print(f"[term2] 프록시 — 요청 {px.get('requests',0)} · 추론(필드 {px.get('think_from_field',0)} / 인라인 "
          f"{px.get('think_stripped',0)} / 없음 {px.get('think_absent',0)}) · 복원 {px.get('restored',0)} "
          f"(필드 인라인 {px.get('reasoning_field_inlined',0)} · 캐시 {px.get('reinlined',0)} / 버림 "
          f"{px.get('reasoning_field_dropped',0)}) · miss {px.get('miss',0)} ({px.get('miss_rate',0)*100:.1f}%) · "
          f"미종결 stop {px.get('think_unclosed_stop', '—')} / length {px.get('think_unclosed', 0)}")
if invalid:
    print("[term2] ❌ 무효: " + " · ".join(invalid))
if exc: print(f"[term2] 예외 분포: {exc}")
print(f"[term2] → {outd}/results_terminal.json")
PY
