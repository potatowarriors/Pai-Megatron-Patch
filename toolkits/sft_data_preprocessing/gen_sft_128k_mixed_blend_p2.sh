#!/usr/bin/env bash
# SFT phase-2 블렌드 재산출 — 절충안 M (docs/SFT_PHASE2_PLAN.md §11, 사용자 승인 2026-09-04).
#   기준 = 교체 재개 yaml(sft_128k_mixed_blend_swap.yaml, phase-1 가중치와 비트 동일).
#   ep=E  : 신규 멤버(형제 셋·능력 추가)는 절대 소비(E × real)      scale=f : phase-1 멤버는 phase-1 비중의 f 배
#     형제 셋이 있는 카테고리(에이전틱·chat·science) 0.15 · 대체재 없는 대형(cp·math) 0.5 ·
#     대체재 없는 소형(한국어·IF+effort·ml ko/ja/pt·identity) 1.0 · safety 는 E_max 상한(0.2ep)
#   --solve-iters : 예산(iters)은 위 규칙에서 풀린다 — 손으로 iters 를 정하지 않는다.
# 실행: bash gen_sft_128k_mixed_blend_p2.sh   (p2 트리에 신규 멤버 stats 가 모두 있어야 한다 — convert_sft_128k_mixed_p2b.sh 후)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
P1=/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_mixed_pad16
P2=/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_mixed_p2_pad16
OUT=${OUT:-$REPO/examples/alpha/configs/data/sft_128k_mixed_blend_p2.yaml}

ML1=()
for lang in de es fr it ja zh; do
  for dom in code math; do ML1+=("ml_super-v3_${dom}_${lang}_translated_final"); done
done
ML1_EP=(); for m in "${ML1[@]}"; do ML1_EP+=("$m=0.05"); done

python3 "$REPO/toolkits/sft_data_preprocessing/gen_phase2_blend.py" \
  --phase1-yaml "$REPO/examples/alpha/configs/data/sft_128k_mixed_blend_swap.yaml" \
  --phase1-tree "$P1" --tree "$P2" --solve-iters \
  --add agentic_v2_search agentic_v2_ia agentic_v2_tc chat_v2_on chat_v2_off \
        swe_v2_openhands swe_v2_agentless swe_v1_r2e science_v1 finance_v1 math_proofs_v1_lean "${ML1[@]}" \
  --ep agentic_v2_search=2.0 agentic_v2_ia=0.25 agentic_v2_tc=0.10 \
       chat_v2_on=0.3 chat_v2_off=0.3 \
       swe_v2_openhands=0.3 swe_v2_agentless=0.3 swe_v1_r2e=0.3 \
       science_v1=1.0 finance_v1=0.1 math_proofs_v1_lean=0.05 "${ML1_EP[@]}" \
       safety_v2=0.2 \
  --scale swe_v3_keepthink=0.15 opencode_fixed=0.15 arc_agi_v1_keepthink=0.15 cuda_v1=0.15 \
          chat_v3_chat=0.15 science_v2=0.15 \
          cp_v2=0.5 math_v4=0.5 math_proofs_v2=0.5 \
          kochat_chat_t2=1.0 kochat_if_fanout_me_t2=1.0 kochat_b_fanout_t2=1.0 \
          chat_v3_if_fanout_me=1.0 budget_trunc_v1_if=1.0 budget_trunc_v1_math=1.0 identity_v2=1.0 \
          ml_ultra-v3_code_ja_translated_final=1.0 ml_ultra-v3_code_ko_translated_final=1.0 \
          ml_ultra-v3_code_pt_translated_final=1.0 ml_ultra-v3_math_ja_translated_final=1.0 \
          ml_ultra-v3_math_ko_translated_final=1.0 ml_ultra-v3_math_pt_translated_final=1.0 \
          ml_ultra-v3_stem_ja_translated_postedit_final=1.0 ml_ultra-v3_stem_ko_translated_postedit_final=1.0 \
          ml_ultra-v3_stem_pt_translated_postedit_final=1.0 \
  --out "$OUT"
echo "-> $OUT"
