#!/usr/bin/env bash
# Phase E — 100-iter smoke test on the final pre-flight config.
#
# Gates pre Stage 1 launch:
#   - Loss must drop from ~12.0 (random init: ln(163968) ≈ 12.01) toward ~11.0
#     by iteration 100 (confirms model + data + tokenizer are wired correctly).
#   - No NaN losses, no exploding grad norm, no OOM at GBS 2688.
#   - WANDB disabled (stage1_v5_blend is real data, not a `mock` preset, so we
#     force WANDB_MODE=disabled explicitly).
#
# Prerequisites (verify before running):
#   - Phase 0.4 complete on ALL THREE Stage 1 sources (DCLM, korean_web, fineweb2hq):
#       check tests/preflight_stage1/B_*_report.json B4 fields should report
#       10000/10000 docs ending in id 0 after re-running run_phase_b.py.
#   - reset-position-ids / reset-attention-mask / eod-mask-loss flags added to
#     examples/alpha/configs/training/stage1.yaml (already done).
#     [Note: pretrain_auxfree.yaml has been deprecated as of 2026-05-12.]
#   - tokenizer_v5/tokenizer_config.json eos_token = "<|endoftext|>" (done).
#   - tokenizer_v5/special_tokens_map.json eos_token = "<|endoftext|>" (done).
#
# Run from repo root:
#   bash tests/preflight_stage1/run_phase_e_smoke.sh
#
# Outputs:
#   - stdout / stderr written to tests/preflight_stage1/E_smoke.log
#   - first/last 5 iter logs extracted into E_smoke_summary.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}/examples/alpha"

LOG="${REPO_ROOT}/tests/preflight_stage1/E_smoke.log"
SUMMARY="${REPO_ROOT}/tests/preflight_stage1/E_smoke_summary.md"

echo "Phase E smoke — 100 iters, WANDB disabled"
echo "Log: ${LOG}"
echo

# Force WANDB off (since 'stage1_v5_blend' is not a smoke preset by name,
# train.sh's auto-disable wouldn't fire).
export WANDB_MODE=disabled

# Run 100 iters (override --train-iters from the YAML's 450000), with no save
# (--no-save uses store_true; if Megatron doesn't recognise it, just bypass it
# by setting save-interval > train-iters — but the YAML already has 25000 so
# nothing will save in 100 iters anyway).
bash train.sh baseline_48L stage1 stage1_v5_blend \
  --train-iters 100 \
  --eval-iters 0 \
  --log-interval 1 \
  2>&1 | tee "${LOG}"

# Extract structured summary
{
  echo "# Phase E — 100-iter smoke result"
  echo
  echo "Final tail of training log:"
  echo
  echo '```'
  tail -25 "${LOG}"
  echo '```'
  echo
  echo "Initial iterations (5 to confirm loss starts near ln(163968)=12.01):"
  echo
  echo '```'
  grep -E 'iteration\s+([1-5])/' "${LOG}" | head -5
  echo '```'
  echo
  echo "Late iterations (96-100, to confirm meaningful descent):"
  echo
  echo '```'
  grep -E 'iteration\s+(9[6-9]|100)/' "${LOG}" | head -5
  echo '```'
} > "${SUMMARY}"

echo
echo "Done. Summary: ${SUMMARY}"
