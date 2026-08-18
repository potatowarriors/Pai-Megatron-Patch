#!/usr/bin/env bash
# =============================================================================
# finalize.sh — 생성 완료 후 데이터셋을 /home/work/Datasets 하위로 이관한다.
#
# 이 스크립트 하나만 실행하면 끝난다. Claude Code 세션이 없어도 동작한다.
#
#   bash finalize.sh
#
# 하는 일:
#   1. 생성 프로세스가 아직 도는지 확인 (돌면 경고 후 중단 — --force 로 무시 가능)
#   2. export_sft.py 로 필터링·중복제거·SFT 포맷 변환
#   3. 산출물 + 재현 자료를 데이터셋 디렉토리에 배치
#   4. 데이터셋 README(provenance) 생성
#
# 부분 결과도 안전하다. 체크포인트(batch_*.parquet)에 쌓인 만큼만 변환한다.
# =============================================================================
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${DST:-/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Identity-v1}"
ARTIFACTS="${ARTIFACTS:-$SRC/artifacts/alpha_identity}"
PY="${PY:-/home/work/vidsearch/repos/project_s/DataDesigner/.venv/bin/python}"
HOLDOUT="${HOLDOUT:-400}"
FORCE="${1:-}"

echo "==> 소스     : $ARTIFACTS"
echo "==> 대상     : $DST"

# ── 1. 생성이 아직 도는지 ────────────────────────────────────────────────────
if pgrep -f "identity_sdg.py --vllm" > /dev/null 2>&1; then
    if [ "$FORCE" != "--force" ]; then
        echo
        echo "🛑 생성 프로세스가 아직 실행 중입니다."
        echo "   진행률: $(grep "column 'judge'" "$SRC/logs/run_full.log" | tail -1 | sed 's/.*judge.: //')"
        echo
        echo "   완료를 기다리거나, 지금까지의 부분 결과로 진행하려면:"
        echo "     bash finalize.sh --force"
        exit 1
    fi
    echo "==> [warn] 생성 진행 중 — 완료된 배치까지만 변환합니다"
fi

BATCHES=$(find "$ARTIFACTS/parquet-files" -name '*.parquet' 2>/dev/null | wc -l)
if [ "$BATCHES" -eq 0 ]; then
    echo "🛑 변환할 배치 파일이 없습니다: $ARTIFACTS/parquet-files/"
    exit 1
fi
echo "==> 배치 파일: ${BATCHES}개"

mkdir -p "$DST/data" "$DST/raw" "$DST/repro"

# ── 2. SFT 포맷으로 변환 ─────────────────────────────────────────────────────
echo
echo "==> export_sft.py 실행"
cd "$SRC"
"$PY" export_sft.py \
    --dataset "$ARTIFACTS/parquet-files/*.parquet" \
    --out-dir "$DST/data" \
    --holdout "$HOLDOUT" \
    2>&1 | tee "$DST/export.log"

# ── 2-b. chat template 검증 ──────────────────────────────────────────────────
# DataDesigner venv 에는 transformers 가 없다. 있는 인터프리터를 찾아서 검증한다.
echo
echo "==> chat template 검증"
for CAND in "$PY" /usr/bin/python3 python3; do
    if "$CAND" -c "import transformers" 2>/dev/null; then
        "$CAND" -c "
import json, sys; sys.path.insert(0, '$SRC')
from export_sft import verify_template, TOKENIZER_DIR
recs=[json.loads(l) for l in open('$DST/data/train.jsonl')]
verify_template(recs, TOKENIZER_DIR, n=5)
" 2>&1 | tee -a "$DST/export.log"
        break
    fi
done

# ── 3. 재현 자료 배치 ────────────────────────────────────────────────────────
echo
echo "==> 재현 자료 복사"
cp -f "$SRC"/identity_card.yaml "$SRC"/prepare_seed.py "$SRC"/identity_sdg.py \
      "$SRC"/export_sft.py "$SRC"/finalize.sh "$SRC"/README.md "$DST/repro/"
cp -f "$SRC/seed.parquet" "$DST/repro/" 2>/dev/null || true
cp -f "$SRC/logs/run_full.log" "$DST/repro/" 2>/dev/null || true
cp -rf "$ARTIFACTS/parquet-files" "$DST/raw/" 2>/dev/null || true
cp -f "$ARTIFACTS/builder_config.json" "$ARTIFACTS/metadata.json" "$DST/raw/" 2>/dev/null || true

# ── 4. provenance README ────────────────────────────────────────────────────
TRAIN_N=$(wc -l < "$DST/data/train.jsonl" 2>/dev/null || echo 0)
EVAL_N=$(wc -l < "$DST/data/eval.jsonl" 2>/dev/null || echo 0)

cat > "$DST/README.md" <<EOF
# alpha-SFT-Identity-v1

alpha-banana-v1 의 **정체성 instruction-following** 합성 데이터셋.
\`LC-phase → SFT → RL(MOPD)\` 중 **SFT 단계** 투입용.

생성일: $(date '+%Y-%m-%d %H:%M')

## 규모

| 파일 | 행 수 | 용도 |
|---|---|---|
| \`data/train.jsonl\` | ${TRAIN_N} | SFT 블렌드 투입 |
| \`data/eval.jsonl\` | ${EVAL_N} | **학습 금지** — 정체성 유지율 측정용 홀드아웃 |

## 스키마

\`Nemotron-SFT-Instruction-Following-Chat-v3\` 와 동일하다.

\`\`\`json
{"messages": [...], "used_in": ["alpha_v1"], "uuid": "...",
 "metadata": {"seed_dataset": ..., "seed_prompt_sha256": ..., "model": ...,
              "reward_model": null, "train_turns": [bool, ...]}}
\`\`\`

\`metadata.train_turns\` 가 턴별 loss 마스크다 (assistant 턴만 true).
assistant 메시지는 alpha chat template 규약을 따른다 —
\`reasoning_content\` 가 있으면 \`<think>\\n{rc}</think>{content}\`, 없으면 \`<think></think>{content}\`.

## 생성 조건

| 항목 | 값 |
|---|---|
| 프레임워크 | NeMo Data Designer |
| 교사 모델 | \`google/gemma-4-12B-it\` (vLLM) |
| 사실 원천 | \`repro/identity_card.yaml\` (APPROVED v1.0) |
| 시드 프롬프트 | \`nvidia/Nemotron-RL-Identity-Following-v1\` (CC-BY-4.0, 상업 사용 가능) |
| 언어 | ko 40% / en 30% / ja·zh 5% / es·fr·de·pt·it 4% (hi 제외 — alpha 미지원) |

## ⚠️ 학습 시 주의

- **SFT 블렌드 내 비중 0.3~1.0%** 를 넘기지 말 것.
  정체성 데이터를 과다 주입하면 무엇을 물어도 자기소개부터 하는 모델이 된다.
  NVIDIA 조차 21,660행만 썼고 그마저 RL 단계용이었다.
- **identity 데이터만 반복 에폭 금지.**
- 일반 IF 데이터(\`Nemotron-SFT-Instruction-Following-Chat-v3\`)와 반드시 섞을 것.
- \`data/eval.jsonl\` 은 학습에 넣지 말 것.

## 디렉토리

\`\`\`
data/     train.jsonl / eval.jsonl   ← 최종 산출물
raw/      Data Designer 원본 parquet + config
repro/    파이프라인 스크립트·시드·로그 (재현용 스냅샷)
export.log
\`\`\`

재현: \`repro/README.md\` 참조. 원본 파이프라인은
\`Pai-Megatron-Patch/examples/alpha/sdg/identity/\`.
EOF

echo
echo "============================================================"
echo "✅ 완료"
echo "   train : ${TRAIN_N} 행  → $DST/data/train.jsonl"
echo "   eval  : ${EVAL_N} 행  → $DST/data/eval.jsonl"
echo "   문서  : $DST/README.md"
echo "============================================================"
