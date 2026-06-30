#!/bin/bash
# Back-compat shim. The Alpha converter is now GPU-count agnostic — see
# run_convert.sh (EP derived from GPU count, model flags derived from the
# checkpoint's common.pt). This shim preserves the historical 4-GPU entrypoint;
# torch_dist reshards an EP=8-trained checkpoint to EP=4 on load.
#
# Prefer:  [GPUS=N] bash run_convert.sh <MODEL_SIZE> <LOAD_DIR> <SAVE_DIR> <MG2HF> <USE_CUDA> <PRECISION> [HF_DIR]
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
exec env GPUS="${GPUS:-4}" bash "${CURRENT_DIR}/run_convert.sh" "$@"
