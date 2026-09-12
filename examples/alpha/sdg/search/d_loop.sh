#!/bin/bash
# D1/D2 본생성 루프: 20분마다 (1) 새 연쇄 → 질문 작성 (2) 새 질문 → D1 트라젝토리 (3) D2 요청 → 트라젝토리 → 심판. 모두 재개형. 목표 도달 시 종료.
cd /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/sdg/search; export NUMEXPR_MAX_THREADS=64
say(){ echo "[$(date +%H:%M:%S)] $*"; }
D1_TARGET=${D1_TARGET:-4000}; D2_TARGET=${D2_TARGET:-1000}
cat out/d2/requests.jsonl out/d2/requests2.jsonl 2>/dev/null > out/d2/requests_all.jsonl
for round in $(seq 1 40); do
  GEN_THOROUGH=0 python3 wd_question.py --chains out/d1/chains.jsonl --out out/d1/questions.jsonl --workers 64 2>&1 | grep -E "^DONE" | cut -c1-120
  AGENT_TEACHER=dsv4-flash python3 agent_run.py --questions out/d1/questions.jsonl --out out/d1/traj.jsonl --search http://localhost:8600 --workers ${D1_WORKERS:-120} --mode d1 2>&1 | grep -E "^DONE|^questions=" | cut -c1-160
  cat out/d2/requests.jsonl out/d2/requests2.jsonl 2>/dev/null > out/d2/requests_all.jsonl
  AGENT_TEACHER=dsv4-flash python3 agent_run.py --questions out/d2/requests_all.jsonl --out out/d2/traj.jsonl --search http://localhost:8600 --workers ${D2_WORKERS:-60} --mode d2 --max-calls 15 --max-tokens 12000 2>&1 | grep -E "^DONE|^questions=" | cut -c1-160
  python3 judge_d2.py --inp out/d2/traj.jsonl --out out/d2/traj.judged.jsonl --workers 64 --min-supported 0.8 2>&1 | grep -E "^DONE" | cut -c1-160
  d1=$(wc -l < out/d1/traj.jsonl 2>/dev/null || echo 0); d2=$(wc -l < out/d2/traj.judged.jsonl 2>/dev/null || echo 0)
  say "round $round: D1 accepted $d1 / D2 judged-ok $d2 (chains $(wc -l < out/d1/chains.jsonl), questions $(wc -l < out/d1/questions.jsonl))"
  [ "$d1" -ge "$D1_TARGET" ] && [ "$d2" -ge "$D2_TARGET" ] && { say "TARGETS REACHED"; break; }
  grep -q "^DONE" out/d1/chains.log && [ "$(wc -l < out/d1/questions.jsonl)" -le "$(wc -l < out/d1/traj.jsonl)" ] && [ "$d2" -ge "$D2_TARGET" ] && { say "inputs exhausted"; break; }
  sleep 600
done
say "D LOOP DONE"
