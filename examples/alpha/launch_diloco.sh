#!/bin/bash
# DiLoCo 2-node launcher for alpha (Backend.AI cluster: main1 + sub1).
#
# Each node runs an INDEPENDENT single-node train.sh; cross-node outer sync is
# handled by diloco_patch.py via per-local-rank Gloo pairs (see that file for
# the algorithm and study/diloco_pilot.md for validation results).
#
# usage:
#   [env knobs] bash launch_diloco.sh <tag> <model> <training> <data> [extra train.sh args...]
#
# env knobs (defaults):
#   DILOCO_H=30              inner steps per outer sync
#   DILOCO_TAU=0             delayed-apply overlap (0 = blocking, validated; 1-2 = overlapped)
#   DILOCO_OUTER_LR=0.7      outer Nesterov lr        (DiLoCo paper default)
#   DILOCO_OUTER_MOMENTUM=0.6 outer Nesterov momentum (MuLoCo best for Muon inner;
#                            was 0.9, the DiLoCo paper default)
#   DILOCO_CKPT_DIR          root for STABLE per-node checkpoint dirs (default
#                            outputs/diloco_<tag>). node r saves to & loads from
#                            <root>/node<r>/checkpoints, so RE-RUNNING THE SAME
#                            COMMAND RESUMES. Pair with a training preset using
#                            `pretrained-checkpoint:` for the first-launch start.
#   NODE1_SEED=4321          node1 data-shuffle seed (weights are broadcast from node0)
#   NODE0_ARGS / NODE1_ARGS  per-node extra Megatron args; override the derived
#                            --save/--load only for a non-standard resume
#   EXTRA_ENV                extra "K=V K=V" passed to both nodes
#                            (resume needs NCCL_MAX_NCHANNELS=16 if not set in train.sh)
#
# Logs: node0 -> ~/run_diloco_<tag>_node0.log (main1), node1 -> same on sub1.
# NOTE: Megatron prints iteration lines on each node's own last rank, so each
# node's log carries its own loss lines.
set -u
TAG=$1; MODEL=$2; TRAIN=$3; DATA=$4; shift 4
ALPHA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_BASE=$((31000 + RANDOM % 400))
# Data mode: DILOCO_DATA_SHARD=1 (PRODUCTION) = exact disjoint sharding over one
# shared global order — node1 must NOT get a different seed. Default (A/B mode)
# = per-node shuffle seeds (node0 stays bit-comparable to 1-node baselines).
if [ "${DILOCO_DATA_SHARD:-0}" = "1" ]; then
    NODE1_SEED_ARG=""
else
    NODE1_SEED_ARG="--seed ${NODE1_SEED:-4321}"
fi
# CUDA_DEVICE_MAX_CONNECTIONS=32: 32 is the CUDA driver's HARD MAX for this variable (it
# maps to GPU hardware work queues; values >32 clamp to 32 with no further effect). The
# sweep 8→16→32 = 171.3→173.6→177.2 TFLOP/s (+3.4% over conn=8, bit-identical) bottoms out
# at 32 — canonical record docs/THROUGHPUT_INVESTIGATION.md §3/§7 (adopt conn=32). Safe
# because alpha is TP=1/CP=1 (Megatron's =1 requirement does not apply). FORCED to 32 (not
# :-32): .pai_megatron_alpha_env exports =1 into the interactive shell, which would defeat
# a :-default on node0, while node1's non-interactive ssh shell may not source the profile
# at all — hardcoding guarantees BOTH nodes match. To A/B a different value, edit this line.
ENVV="NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
CUDA_DEVICE_MAX_CONNECTIONS=32 \
DILOCO_WORLD=2 DILOCO_MASTER=main1 DILOCO_PORT_BASE=$PORT_BASE \
DILOCO_H=${DILOCO_H:-30} DILOCO_TAU=${DILOCO_TAU:-0} DILOCO_OUTER_LR=${DILOCO_OUTER_LR:-0.7} \
DILOCO_OUTER_MOMENTUM=${DILOCO_OUTER_MOMENTUM:-0.6} DILOCO_DATA_SHARD=${DILOCO_DATA_SHARD:-0} \
PRETRAIN_SCRIPT=$ALPHA/pretrain_alpha_diloco.py ${EXTRA_ENV:-}"

echo "[launch_diloco] tag=$TAG presets=$MODEL/$TRAIN/$DATA H=${DILOCO_H:-30} tau=${DILOCO_TAU:-0} port_base=$PORT_BASE data_shard=${DILOCO_DATA_SHARD:-0} node1_seed=${NODE1_SEED:-shared} args=$*"

# Stable per-node checkpoint dirs → crash-safe auto-resume. node r saves to and
# loads from its OWN dir (they hold different shards + different DiLoCo outer
# state), so re-running the same command RESUMES; the training preset's
# `pretrained-checkpoint:` supplies the first-launch fresh start. outputs/ is
# shared NFS, so node0 mkdir's both. Injected AFTER "$@" (override train.sh's
# timestamped --save) but BEFORE NODE{0,1}_ARGS (an explicit override still wins).
CKPT_ROOT="${DILOCO_CKPT_DIR:-$ALPHA/outputs/diloco_${TAG}}"
CKPT0="$CKPT_ROOT/node0/checkpoints"; CKPT1="$CKPT_ROOT/node1/checkpoints"
mkdir -p "$CKPT0" "$CKPT1"
echo "[launch_diloco] ckpt root=$CKPT_ROOT (stable node0/node1 dirs — re-run this command to resume)"

ssh -o StrictHostKeyChecking=no sub1 \
  "cd $ALPHA && nohup env $ENVV DILOCO_RANK=1 bash train.sh $MODEL $TRAIN $DATA $* ${NODE1_SEED_ARG} --save $CKPT1 --load $CKPT1 ${NODE1_ARGS:-} > \$HOME/run_diloco_${TAG}_node1.log 2>&1 < /dev/null & echo node1_pid=\$!"

cd "$ALPHA"
env $ENVV DILOCO_RANK=0 bash train.sh "$MODEL" "$TRAIN" "$DATA" "$@" --save "$CKPT0" --load "$CKPT0" ${NODE0_ARGS:-} \
  > "$HOME/run_diloco_${TAG}_node0.log" 2>&1
RC=$?
echo "[launch_diloco] node0 exit=$RC (log: ~/run_diloco_${TAG}_node0.log)"
grep -F "elapsed time" "$HOME/run_diloco_${TAG}_node0.log" | tail -4 | cut -c1-160
grep -F "[diloco]" "$HOME/run_diloco_${TAG}_node0.log" | grep -v "wrapped" | tail -6
exit $RC
