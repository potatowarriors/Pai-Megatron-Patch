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
#   DILOCO_OUTER_LR=0.7      outer Nesterov lr        (DiLoCo paper defaults)
#   DILOCO_OUTER_MOMENTUM=0.9
#   NODE1_SEED=4321          node1 data-shuffle seed (weights are broadcast from node0)
#   NODE0_ARGS / NODE1_ARGS  per-node extra args (e.g. --load <own ckpt> on resume —
#                            each node must load ITS OWN checkpoint)
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
ENVV="NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
DILOCO_WORLD=2 DILOCO_MASTER=main1 DILOCO_PORT_BASE=$PORT_BASE \
DILOCO_H=${DILOCO_H:-30} DILOCO_TAU=${DILOCO_TAU:-0} DILOCO_OUTER_LR=${DILOCO_OUTER_LR:-0.7} \
DILOCO_OUTER_MOMENTUM=${DILOCO_OUTER_MOMENTUM:-0.9} DILOCO_DATA_SHARD=${DILOCO_DATA_SHARD:-0} \
PRETRAIN_SCRIPT=$ALPHA/pretrain_alpha_diloco.py ${EXTRA_ENV:-}"

echo "[launch_diloco] tag=$TAG presets=$MODEL/$TRAIN/$DATA H=${DILOCO_H:-30} tau=${DILOCO_TAU:-0} port_base=$PORT_BASE data_shard=${DILOCO_DATA_SHARD:-0} node1_seed=${NODE1_SEED:-shared} args=$*"

ssh -o StrictHostKeyChecking=no sub1 \
  "cd $ALPHA && nohup env $ENVV DILOCO_RANK=1 bash train.sh $MODEL $TRAIN $DATA $* ${NODE1_SEED_ARG} ${NODE1_ARGS:-} > \$HOME/run_diloco_${TAG}_node1.log 2>&1 < /dev/null & echo node1_pid=\$!"

cd "$ALPHA"
env $ENVV DILOCO_RANK=0 bash train.sh "$MODEL" "$TRAIN" "$DATA" "$@" ${NODE0_ARGS:-} \
  > "$HOME/run_diloco_${TAG}_node0.log" 2>&1
RC=$?
echo "[launch_diloco] node0 exit=$RC (log: ~/run_diloco_${TAG}_node0.log)"
grep -F "elapsed time" "$HOME/run_diloco_${TAG}_node0.log" | tail -4 | cut -c1-160
grep -F "[diloco]" "$HOME/run_diloco_${TAG}_node0.log" | grep -v "wrapped" | tail -6
exit $RC
