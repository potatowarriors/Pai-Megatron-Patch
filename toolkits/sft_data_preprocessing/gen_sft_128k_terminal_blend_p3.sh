#!/bin/bash
# gen_sft_128k_terminal_blend_p3.sh — 터미널 에이전트 보정 스테이지(phase-3) 블렌드 생성.
#
# 설계 (examples/alpha/sdg/terminal/README.md §P4, 사용자 결정 2026-09-07~09):
#   phase-2 완주 ckpt 위에 짧은 연속학습. 신규 멤버 2종을 ep 로 고정하고, 나머지 phase-2 멤버는 phase-2 비중의 f 배로 리플레이.
#     terminal_terminus2_synth  — GLM-5.3-Flash 합성 Terminus-2 트라젝토리 (코드:수학 7:3, 보존 렌더)         ep=TERM_EP (기본 2.0)
#     swe_v3_terminus_keephist  — SWE-v3 의 Terminus 행 3.5k 을 보존 렌더로 재변환한 멤버 (기존 swe_v3_keepthink 와 중복 — 그쪽은 scale 로 축소) ep=SWE_EP (기본 1.0)
#   리플레이 80% = terminal_p3_replay_shares.tsv (phase-1+2 누적 소비 분포 기준, 사용자 결정 2026-09-09), 신규 20% → --solve-iters 로 예산을 푼다.
# 전제: P3 트리에 두 멤버의 bins 와 data.stats.json 이 있어야 한다 (assemble_dataset.sh, extract_swe_v3_terminus.py + build).
# 사용: TERM_EP=4.0 SWE_EP=1.0 bash gen_sft_128k_terminal_blend_p3.sh   (리플레이 80% = tsv 합, 신규 20%)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
P2=/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_mixed_p2_pad16
P3=/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_terminal_pad16
OUT=${OUT:-$REPO/examples/alpha/configs/data/sft_128k_terminal_blend_p3.yaml}
TERM_EP=${TERM_EP:-4.0}; SWE_EP=${SWE_EP:-1.0}
for m in terminal_terminus2_synth swe_v3_terminus_keephist; do
  [ -f "$P3/$m/data.stats.json" ] || { echo "❌ $P3/$m/data.stats.json 없음 — 먼저 bins 를 만들 것"; exit 1; }
done
# 리플레이 비중은 terminal_p3_replay_shares.tsv (49종, 합 0.80) — phase-1+2 누적 소비 분포 기준 (README §P4)
TSV="$REPO/toolkits/sft_data_preprocessing/terminal_p3_replay_shares.tsv"
SHARE=(); while IFS=$'\t' read -r m sh _rest; do [ -z "$m" ] || [ "${m:0:1}" = "#" ] && continue; SHARE+=("$m=$sh"); done < "$TSV"
echo "replay members: ${#SHARE[@]}"
python3 "$REPO/toolkits/sft_data_preprocessing/gen_phase2_blend.py" \
  --phase1-yaml "$REPO/examples/alpha/configs/data/sft_128k_mixed_blend_p2.yaml" \
  --phase1-tree "$P2" --tree "$P3" --solve-iters \
  --add terminal_terminus2_synth swe_v3_terminus_keephist \
  --ep terminal_terminus2_synth=$TERM_EP swe_v3_terminus_keephist=$SWE_EP \
  --share "${SHARE[@]}" \
  --out "$OUT"
echo "-> $OUT"
