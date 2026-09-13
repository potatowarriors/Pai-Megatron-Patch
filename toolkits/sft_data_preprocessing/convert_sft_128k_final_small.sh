#!/bin/bash
# 최종 SFT 트리의 **100 bins 미만 멤버 해소** (2026-09-13). split 99/1 에서 bins<100 이면 valid 문서 0 → GPTDataset 빌드 무한대기
# (rules/sft-data.md "bins<100인 셋은 valid 0-doc 무한대기 — 입력을 늘려 회피"; 09-13 02:03 최종 스모크가 usab_v1_nothink(35 bins) 에서 정지한 것을 실측).
#   usab_v1          : usab_v1_think(273 bins) + usab_v1_nothink(35) 를 한 멤버로 (둘 다 4ep 고정이라 블렌드 결과 불변)
#   when2call_v1_x2  : when2call_v1(66 bins) 행을 2벌(두 번째 벌은 uuid 에 :x2) → 스펙 base_ep 3 = 고유 데이터 6ep
#   kotool_v1_x2     : kotool_v1(81 bins) 행을 2벌 → 스펙 base_ep 1 = 고유 데이터 2ep
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd); A=$REPO/examples/alpha/sdg
OUT=${OUT:-/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_final_pad16}; SRC=$OUT/.src_small; mkdir -p "$SRC"
TOK=$REPO/examples/alpha/tokenizer_v5
export TOKENIZERS_PARALLELISM=false NUMEXPR_MAX_THREADS=64 PYTHONPATH=$REPO:$REPO/backends/megatron/Megatron-LM-251125${PYTHONPATH:+:$PYTHONPATH}
dup2() { python3 -c "
import json,sys
for l in open(sys.argv[1]): sys.stdout.write(l)
for l in open(sys.argv[1]):
    r=json.loads(l); r['uuid']=r.get('uuid','')+':x2'; sys.stdout.write(json.dumps(r, ensure_ascii=False)+'\n')" "$1"; }
cat "$A/usab/out/export/usab_v1_think.jsonl" "$A/usab/out/export/usab_v1_nothink.jsonl" > "$SRC/usab_v1.jsonl"
dup2 "$A/kotool/out/when2call/when2call_v1.jsonl" > "$SRC/when2call_v1_x2.jsonl"
dup2 "$A/kotool/out/export/kotool_v1.jsonl" > "$SRC/kotool_v1_x2.jsonl"
wc -l "$SRC"/*.jsonl
run() {
  local name=$1 input=$2; shift 2
  echo "== $name: $input ($(date +%H:%M:%S))"; mkdir -p "$OUT/$name"
  nice -n 10 python3 "$REPO/toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py" --input "$input" --tokenizer "$TOK" --output-prefix "$OUT/$name/data" \
    --seq-length 131072 --pad-doc-multiple 16 --workers 32 "$@" 2>&1 | grep -E "^\[(encode|pack|done)\]|Error|Traceback" || echo "!! $name FAILED"
}
run usab_v1         "$SRC/usab_v1.jsonl"
run when2call_v1_x2 "$SRC/when2call_v1_x2.jsonl"
run kotool_v1_x2    "$SRC/kotool_v1_x2.jsonl"
echo "== DONE ($(date +%H:%M:%S))"
