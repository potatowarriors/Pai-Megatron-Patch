#!/bin/bash
# run_collect.sh — P3 수집 배치: 유효 과제 디렉토리 → gpu06 Harbor(terminus-2, 교사 GLM-5.3-Flash) → 트라젝토리 회수 → 변환.
#
# 사용: bash run_collect.sh <valid_task_dir> <tag> [W=64] [EFFORT=high] [MAX_TURNS=30]
#   env: TERM_MAX_TOKENS(기본 32768 — 스텝당 출력 상한; reasoning 포함), KEEP_REMOTE=1 이면 컨테이너 job 디렉토리 보존
# 산출: /home/work/vidsearch/tools/glm53/collect/<tag>/{job/, summary.txt, summary.json, rows/<tag>.jsonl, rows/MANIFEST.json}
#
# 설계 근거 (README §5 P1b): TP8 서버 + 동시 64, effort high(70%)/low(30%) 는 배치 단위로 나눠 실행,
# max_turns 30 으로 요약(summarization) 발동 전에 멈춘다, k=1 (재시도는 실패 과제만 나중에).
set -uo pipefail
TASKS="${1:?valid task dir}"; TAG="${2:?tag}"; W="${3:-64}"; EFFORT="${4:-high}"; MAX_TURNS="${5:-30}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SSHC=/home/work/vidsearch/.ssh-keys/config
OUT=/home/work/vidsearch/tools/glm53/collect/$TAG; mkdir -p "$OUT"
REMOTE=/opt/harbor/synth/$TAG
RID="col-$TAG"
N=$(find "$TASKS" -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "[collect] $TAG: tasks=$N W=$W effort=$EFFORT max_turns=$MAX_TURNS start=$(date +%m-%d\ %H:%M:%S)"

ssh -F "$SSHC" -o BatchMode=yes alpha-eval "curl -s -o /dev/null -w '%{http_code}' localhost:8299/v1/models" | grep -q 200 \
  || { echo "[collect] ❌ 컨테이너에서 GLM 역터널(8299) 응답 없음"; exit 1; }
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "rm -rf $REMOTE /opt/harbor/jobs/$RID; mkdir -p /opt/harbor/synth"
tar -C "$TASKS" --exclude='./GEN_MANIFEST.jsonl' --exclude='./VALID.txt' --exclude='./INVALID.txt' --exclude='./PRECHECK_FAIL.txt' --exclude='./.*.txt' -czf - . | ssh -F "$SSHC" -o BatchMode=yes alpha-eval "mkdir -p $REMOTE && tar -C $REMOTE -xzf -"

T0=$(date +%s)
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
  export HOME=/opt/harbor
  export OPENAI_API_KEY=dummy OPENAI_API_BASE=http://localhost:8299/v1
  export LITELLM_MODEL_REGISTRY_PATH=/opt/harbor/glm53_model_registry.json
  cd /opt/harbor
  ./venv/bin/harbor run -p $REMOTE -a terminus-2 -m openai/glm53-flash \
    --ak api_base=http://localhost:8299/v1 \
    --ak temperature=1.0 \
    --ak parser_name=json \
    --ak max_turns=$MAX_TURNS \
    --ak reasoning_effort=$EFFORT \
    --ak 'trajectory_config={\"raw_content\":true}' \
    --ak store_all_messages=true \
    --ak interleaved_thinking=true \
    --ak 'llm_call_kwargs={\"top_p\":0.95,\"max_tokens\":${TERM_MAX_TOKENS:-32768}}' \
    -n $W -k 1 \
    -o /opt/harbor/jobs --job-name $RID -y -q 2>&1 | tail -5
"
T1=$(date +%s)
echo "[collect] harbor done in $(( (T1-T0)/60 )) min → 회수"
rm -rf "$OUT/job"; scp -q -r -F "$SSHC" "alpha-eval:/opt/harbor/jobs/$RID" "$OUT/job"
python3 "$HERE/../pilot/summarize_pilot.py" "$OUT/job" --json "$OUT/summary.json" > "$OUT/summary.txt"; head -16 "$OUT/summary.txt"
mkdir -p "$OUT/rows"
python3 "$HERE/../convert/traj_to_terminus.py" "$OUT/job" --out "$OUT/rows" --tag "$TAG" --min-reward 1.0
NT=$(find "$OUT/job" -maxdepth 1 -mindepth 1 -type d -name '*__*' | wc -l)
echo "[collect] $TAG: trials=$NT wall=$(( (T1-T0)/60 ))min → $(( NT*60 / ((T1-T0)/60 + 1) )) trials/hour · rows → $OUT/rows"
[ "${KEEP_REMOTE:-0}" = "1" ] || ssh -F "$SSHC" -o BatchMode=yes alpha-eval "rm -rf /opt/harbor/jobs/$RID $REMOTE"
