#!/usr/bin/env bash
# Watcher: waits for the in-progress DCLM remap_eod to finish, then runs
# Phase B re-verification (post-injection B4) and Phase D (training-time
# data flow audit) sequentially.
#
# Logs to stdout (line-buffered) so a Monitor tool can react to milestones.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

DCLM_PID="${DCLM_PID:-1525389}"
DCLM_LOG="/tmp/remap_dclm.log"

echo "[watcher] starting. monitoring DCLM remap pid=${DCLM_PID}"
echo "[watcher] timestamp: $(date '+%F %T')"

# --- 1. Wait for DCLM remap process to finish ---
poll_count=0
while kill -0 "${DCLM_PID}" 2>/dev/null; do
  poll_count=$((poll_count + 1))
  if [ $((poll_count % 30)) -eq 0 ]; then
    echo "[watcher] still running. elapsed: $(ps -o etime= -p ${DCLM_PID} 2>/dev/null | tr -d ' ')  log tail: $(tail -1 ${DCLM_LOG} 2>/dev/null | head -c 100)"
  fi
  sleep 10
done

echo "[watcher] DCLM remap process exited at $(date '+%F %T')"
echo "[watcher] === final DCLM log tail ==="
tail -15 "${DCLM_LOG}"
echo "[watcher] === end DCLM log tail ==="

# Check DCLM exit was clean (look for DONE line or any error)
if grep -q "DONE\." "${DCLM_LOG}" 2>/dev/null; then
  echo "[watcher] ✅ DCLM remap reports DONE"
else
  echo "[watcher] ⚠️  DCLM log does NOT show 'DONE.' — possible failure"
  echo "[watcher] aborting follow-up steps; please investigate manually"
  exit 2
fi

# --- 2. Phase B re-verification (full Phase B for now; B4 will flip) ---
echo
echo "[watcher] running Phase B re-verification (B4 should now show 10000/10000 id 0 endings)"
echo "[watcher] timestamp: $(date '+%F %T')"

PHASE_B_LOG="${REPO_ROOT}/tests/preflight_stage1/B_postinjection.log"
python3 tests/preflight_stage1/run_phase_b.py > "${PHASE_B_LOG}" 2>&1
PHASE_B_RC=$?
echo "[watcher] Phase B exited rc=${PHASE_B_RC}"
echo "[watcher] === Phase B summary (B4 last-token findings per source) ==="
grep -E "B4: doc-boundary EOD audit|docs end in id|top 10 last-token" "${PHASE_B_LOG}" | head -25
echo "[watcher] === end Phase B summary ==="

if [ ${PHASE_B_RC} -ne 0 ]; then
  echo "[watcher] ⚠️ Phase B re-verification failed; aborting Phase D"
  exit 3
fi

# --- 3. Phase D — training-time data flow ---
echo
echo "[watcher] running Phase D (training-time data flow)"
echo "[watcher] timestamp: $(date '+%F %T')"

PHASE_D_LOG="${REPO_ROOT}/tests/preflight_stage1/D_run.log"
python3 tests/preflight_stage1/run_phase_d.py > "${PHASE_D_LOG}" 2>&1
PHASE_D_RC=$?
echo "[watcher] Phase D exited rc=${PHASE_D_RC}"
echo "[watcher] === Phase D summary ==="
grep -E "share:|EOD|verdict|loss_mask_at_eod_zero|position_resets_to_0" "${PHASE_D_LOG}" | head -30
echo "[watcher] === end Phase D summary ==="

if [ ${PHASE_D_RC} -ne 0 ]; then
  echo "[watcher] ⚠️ Phase D failed"
  exit 4
fi

# --- 4. Done ---
echo
echo "[watcher] 🟢 all post-DCLM steps completed at $(date '+%F %T')"
echo "[watcher] artifacts:"
echo "[watcher]   ${PHASE_B_LOG}"
echo "[watcher]   ${PHASE_D_LOG}"
echo "[watcher]   ${REPO_ROOT}/tests/preflight_stage1/B_dataset_integrity.md"
echo "[watcher]   ${REPO_ROOT}/tests/preflight_stage1/D_dataflow.md"
echo "[watcher]   ${REPO_ROOT}/tests/preflight_stage1/D_sample_snapshot.txt"
echo "[watcher] next step: review artifacts and (optionally) launch Phase E smoke:"
echo "[watcher]   bash tests/preflight_stage1/run_phase_e_smoke.sh"
