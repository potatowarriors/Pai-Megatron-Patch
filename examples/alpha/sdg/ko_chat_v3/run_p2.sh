#!/bin/bash
# run_p2.sh A|B [workers] — P2 생성 기동(재개 가능). 먼저 시드 재병합(해시 배정이라 안정), 리젝 구제 반영.
#   A = GLM 생성·DSV4 심판 (main1 GLM DP8, 랭크당 96 → 기본 256)   B = DSV4 생성·GLM 심판 (sub1 seqs 256, KV 903k → 기본 224)
set -u; cd /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/sdg/ko_chat_v3; export NUMEXPR_MAX_THREADS=64
side=$1; w=${2:-}
python3 rescue_ctx.py --rejects out/p1/seeds_ctx_glm.rejects.jsonl,out/p1/seeds_ctx_dsv4.rejects.jsonl --out out/p1/seeds_ctx_rescued.jsonl
python3 merge_seeds.py
if [ "$side" = A ]; then export GEN_TEACHERS=glm53-flash JUDGE=dsv4-flash; w=${w:-256}
else export GEN_TEACHERS=dsv4-flash JUDGE=glm53-flash; w=${w:-224}; fi
echo "[run_p2] side=$side GEN_TEACHERS=$GEN_TEACHERS JUDGE=$JUDGE workers=$w $(date '+%F %T')" | tee -a out/p1/gen_${side}.log
nohup python3 generate_v3.py --seeds out/p1/seeds_p1_${side}.jsonl --out out/p1/gen_${side}.jsonl --workers "$w" ${SKIP_REJECTED:---skip-rejected} >> out/p1/gen_${side}.log 2>&1 < /dev/null &
echo "pid $!"
