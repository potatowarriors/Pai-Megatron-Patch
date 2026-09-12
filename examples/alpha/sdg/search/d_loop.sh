#!/bin/bash
# D 본생성: 세 루프를 독립 실행(GPU 유휴 방지) — A) 질문 작성(GLM) B) D1 트라젝토리(DSV4) C) D2 트라젝토리(DSV4)+심판(GLM). 모두 재개형.
# 사용: nohup bash d_loop.sh questions|d1|d2 > out/d_loop_<mode>.log &
cd /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/sdg/search; export NUMEXPR_MAX_THREADS=64
say(){ echo "[$(date +%H:%M:%S)] $*"; }
case "$1" in
  questions)
    for r in $(seq 1 60); do
      GEN_THOROUGH=0 python3 wd_question.py --chains out/d1/chains.jsonl --out out/d1/questions.jsonl --workers ${Q_WORKERS:-24} 2>&1 | grep -E "^DONE" | cut -c1-120
      say "questions=$(wc -l < out/d1/questions.jsonl) chains=$(wc -l < out/d1/chains.jsonl)"
      grep -q "^DONE" out/d1/chains.log && [ $r -gt 2 ] && break
      sleep 300
    done; say "QUESTIONS LOOP DONE" ;;
  d1)
    for r in $(seq 1 60); do
      AGENT_TEACHER=dsv4-flash python3 agent_run.py --questions out/d1/questions.jsonl --out out/d1/traj.jsonl --search http://127.0.0.1:8600 --workers ${D1_WORKERS:-120} --mode d1 2>&1 | grep -E "^DONE" | cut -c1-140
      n=$(wc -l < out/d1/traj.jsonl 2>/dev/null || echo 0); say "D1 accepted $n (questions $(wc -l < out/d1/questions.jsonl))"
      [ "$n" -ge "${D1_TARGET:-4000}" ] && { say "D1 TARGET"; break; }
      sleep 120
    done; say "D1 LOOP DONE" ;;
  d2)
    for r in $(seq 1 30); do
      cat out/d2/requests.jsonl out/d2/requests2.jsonl 2>/dev/null > out/d2/requests_all.jsonl
      AGENT_TEACHER=dsv4-flash python3 agent_run.py --questions out/d2/requests_all.jsonl --out out/d2/traj.jsonl --search http://127.0.0.1:8600 --workers ${D2_WORKERS:-60} --mode d2 --max-calls 15 --max-tokens 12000 2>&1 | grep -E "^DONE" | cut -c1-140
      python3 judge_d2.py --inp out/d2/traj.jsonl --out out/d2/traj.judged.jsonl --workers 48 --min-supported 0.8 2>&1 | grep -E "^DONE" | cut -c1-140
      n=$(wc -l < out/d2/traj.judged.jsonl 2>/dev/null || echo 0); say "D2 judged-ok $n (traj $(wc -l < out/d2/traj.jsonl 2>/dev/null || echo 0))"
      [ "$n" -ge "${D2_TARGET:-1000}" ] && { say "D2 TARGET"; break; }
      t=$(wc -l < out/d2/traj.jsonl 2>/dev/null || echo 0); q=$(wc -l < out/d2/requests_all.jsonl)
      [ "$t" -ge "$q" ] && grep -q "^DONE" out/d2/seeds2.log 2>/dev/null && { say "D2 requests exhausted"; break; }
      sleep 120
    done; say "D2 LOOP DONE" ;;
esac
