#!/bin/bash
# gen_sft_128k_terminal_blend_p3.sh — 터미널 에이전트 보정 스테이지(phase-3) 블렌드 생성.
#
# 설계 (examples/alpha/sdg/terminal/README.md §P4, 사용자 결정 2026-09-07~09):
#   phase-2 완주 ckpt 위에 짧은 연속학습. 신규 멤버 2종을 ep 로 고정하고, 나머지 phase-2 멤버는 phase-2 비중의 f 배로 리플레이.
#     terminal_terminus2_synth  — GLM-5.3-Flash 합성 Terminus-2 트라젝토리 (코드:수학 7:3, 보존 렌더)         ep=TERM_EP (기본 2.0)
#     swe_v3_terminus_keephist  — SWE-v3 의 Terminus 행 3.5k 을 보존 렌더로 재변환한 멤버 (기존 swe_v3_keepthink 와 중복 — 그쪽은 scale 로 축소) ep=SWE_EP (기본 1.0)
#   리플레이 f=REPLAY (기본 0.10): 신규 2종이 스테이지 토큰의 ≈15~20% 가 되도록 --solve-iters 로 예산을 푼다 (망각 방지 1차 장치는 LR 0.4×, docs/SFT_PHASE2_PLAN.md).
# 전제: P3 트리에 두 멤버의 bins 와 data.stats.json 이 있어야 한다 (assemble_dataset.sh, extract_swe_v3_terminus.py + build).
# 사용: TERM_EP=2.0 SWE_EP=1.0 REPLAY=0.10 bash gen_sft_128k_terminal_blend_p3.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
P2=/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_mixed_p2_pad16
P3=/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_terminal_pad16
OUT=${OUT:-$REPO/examples/alpha/configs/data/sft_128k_terminal_blend_p3.yaml}
TERM_EP=${TERM_EP:-2.0}; SWE_EP=${SWE_EP:-1.0}; REPLAY=${REPLAY:-0.10}
for m in terminal_terminus2_synth swe_v3_terminus_keephist; do
  [ -f "$P3/$m/data.stats.json" ] || { echo "❌ $P3/$m/data.stats.json 없음 — 먼저 bins 를 만들 것"; exit 1; }
done
# phase-2 블렌드의 모든 멤버를 REPLAY 배로 (신규 2종 제외). 멤버 목록은 yaml 에서 읽는다.
MEMBERS=$(grep -oE "^\s*-\s*[0-9.]+\s+\S+" "$REPO/examples/alpha/configs/data/sft_128k_mixed_blend_p2.yaml" | awk '{print $NF}' | xargs -n1 basename | sort -u)
SCALE=(); for m in $MEMBERS; do SCALE+=("$m=$REPLAY"); done
python3 "$REPO/toolkits/sft_data_preprocessing/gen_phase2_blend.py" \
  --phase1-yaml "$REPO/examples/alpha/configs/data/sft_128k_mixed_blend_p2.yaml" \
  --phase1-tree "$P2" --tree "$P3" --solve-iters \
  --add terminal_terminus2_synth swe_v3_terminus_keephist \
  --ep terminal_terminus2_synth=$TERM_EP swe_v3_terminus_keephist=$SWE_EP \
  --scale "${SCALE[@]}" \
  --out "$OUT"
echo "-> $OUT"
