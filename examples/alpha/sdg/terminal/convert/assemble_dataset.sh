#!/bin/bash
# assemble_dataset.sh — 수집 배치별 rows 를 모아 품질 필터 → 코드:수학 비율 조립 → 정본 jsonl + MANIFEST → bins 빌드·게이트.
#
# 사용: bash assemble_dataset.sh [--ratio 7:3] [--name alpha-SFT-Terminal-v1] [--strict] [--collect-root DIR]
#   비율은 **행 수 기준**, 코드 = oc-* + sc-*(시나리오는 코드/터미널 쪽으로 셈), 수학 = om-*. 수학이 남으면 결정적(seed 0) 서브샘플,
#   수학이 모자라면 코드를 줄이지 않고 실제 비율을 MANIFEST 에 기록한다.
# 산출:
#   /home/work/Datasets/LL_datasets/posttraining/SFT/<name>/{train.jsonl, MANIFEST.json, FILTER_STATS.json}
#   /home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_terminal_pad16/terminal_terminus2_synth/  (bins, --keep-history-think)
#   + verify_sft_bins PASS + render_check RENDER_CHECK.md  — 통과해야 블렌드 편입 (루트 CLAUDE.md 검증 규칙)
set -uo pipefail
RATIO="7:3"; NAME="alpha-SFT-Terminal-v1"; STRICT=""; CROOT=/home/work/vidsearch/tools/glm53/collect
while [ $# -gt 0 ]; do case "$1" in
  --ratio) RATIO="$2"; shift 2;; --name) NAME="$2"; shift 2;; --strict) STRICT="--strict"; shift;; --collect-root) CROOT="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 2;; esac; done
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../../../../.." && pwd)"
DS=/home/work/Datasets/LL_datasets/posttraining/SFT/$NAME
BINS=/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_terminal_pad16/terminal_terminus2_synth
TOK=$REPO/examples/alpha/tokenizer_v5
mkdir -p "$DS"
echo "[assemble] 0) job 디렉토리 재변환 (--repair-json-escapes: LaTeX 역슬래시 구제, 2026-09-08 수학 배치 ≈+15% 행)"
for J in "$CROOT"/*/job; do
  [ -d "$J" ] || continue; T=$(basename "$(dirname "$J")"); R="$(dirname "$J")/rows"
  python3 "$HERE/traj_to_terminus.py" "$J" --out "$R" --tag "$T" --min-reward 1.0 --repair-json-escapes | grep traj_to | cut -c1-160
done
ROWS=$(ls "$CROOT"/*/rows/*.jsonl 2>/dev/null | grep -v DONOTTRAIN)
[ -n "$ROWS" ] || { echo "[assemble] rows 없음 ($CROOT)"; exit 1; }
echo "[assemble] 입력 $(echo "$ROWS" | wc -l) 파일, 총 $(cat $ROWS | wc -l) 행"

echo "[assemble] 1) 품질 필터"; python3 "$HERE/filter_rows.py" $ROWS --out "$DS/.filter" --name all $STRICT || exit 1
cp "$DS/.filter/FILTER_STATS.json" "$DS/FILTER_STATS.json"

echo "[assemble] 2) 비율 조립 ($RATIO)"
python3 - "$DS/.filter/all.jsonl" "$DS/train.jsonl" "$RATIO" "$DS/MANIFEST.json" "$NAME" <<'PY'
import json, random, sys, collections
src, dst, ratio, man_p, name = sys.argv[1:6]
c_w, m_w = [float(x) for x in ratio.split(":")]
rows = [json.loads(l) for l in open(src) if l.strip()]
def kind(r):
    t = (r.get("metadata") or {}).get("task", "")
    return "math" if t.startswith("om-") else "code"
code = [r for r in rows if kind(r) == "code"]; math = [r for r in rows if kind(r) == "math"]
target_math = int(round(len(code) * m_w / c_w))
rng = random.Random(0)
if len(math) > target_math:
    math = rng.sample(math, target_math)
out = code + math; rng.shuffle(out)
by_src = collections.Counter((r.get("metadata") or {}).get("task", "")[:2] for r in out)
by_batch = collections.Counter(r["uuid"].rsplit("-", 1)[0] for r in out)
turns = sum((r.get("metadata") or {}).get("assistant_turns", 0) for r in out)
with open(dst, "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
man = {"name": name, "rows": len(out), "code_rows": len(code), "math_rows": len(math), "requested_ratio": ratio,
       "actual_ratio_code_to_math": round(len(code) / max(len(math), 1), 3), "by_prefix": dict(by_src), "by_batch": dict(by_batch),
       "assistant_turns": turns, "teacher": "GLM-5.3-Flash (vLLM TP8, effort high/low mix)", "harness": "Harbor 0.22 terminus-2 json",
       "schema": "Nemotron messages jsonl: system/user/assistant(+reasoning_content), SWE-v3 Terminus 행과 동일 역할 구조",
       "conversion_note": "build_alpha_sft_idxmap.py --keep-history-think --seq-length 131072 --pad-doc-multiple 16 (think 보존 렌더, sdg/terminal/README.md §1.1)",
       "license": "synthetic; seeds OpenCodeReasoning/OpenMathReasoning cc-by-4.0; teacher GLM-5.3-Flash MIT",
       "do_not_train": "tools/glm53/pilot/* (TB-2 canary)"}
json.dump(man, open(man_p, "w"), ensure_ascii=False, indent=2)
print(f"[assemble] rows={len(out)} code={len(code)} math={len(math)} (요청 {ratio}, 실제 {man['actual_ratio_code_to_math']}:1) turns={turns}")
PY

echo "[assemble] 3) bins 빌드 (--keep-history-think)"
rm -rf "$BINS"; mkdir -p "$BINS"
python3 "$REPO/toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py" --input "$DS/train.jsonl" --tokenizer "$TOK" \
  --output-prefix "$BINS/data" --seq-length 131072 --pad-doc-multiple 16 --workers 16 --keep-history-think 2>&1 | grep -v -i "warn\|nthreads" | tail -3
python3 -c "import json; s=json.load(open('$BINS/data.stats.json')); print('[assemble] stats', {k: s.get(k) for k in ('rows_read','samples_kept','drops','n_bins','real_tokens','trainable_tokens')})"
echo "[assemble] 4) verify_sft_bins"; python3 "$REPO/toolkits/sft_data_preprocessing/verify_sft_bins.py" --tree "$(dirname "$BINS")" --seq-length 131072 2>&1 | grep -v -i "warn\|nthreads" | tail -4
echo "[assemble] 5) render_check (규칙 9)"; python3 "$REPO/toolkits/sft_data_preprocessing/render_check.py" --member "$BINS" --docs 0,1000,-1 --tokenizer "$TOK" --write 2>&1 | grep -v -i "warn\|nthreads" | grep -E "^## doc|tokens|RESULT"
echo "[assemble] 완료 → $DS ; bins $BINS"
