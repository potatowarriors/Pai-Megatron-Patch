"""
lm_eval wrapper for multi-GPU evaluation.
Disables WandB on non-main processes to prevent duplicate runs.
"""
import os
import sys

# Only enable WandB on main process (LOCAL_RANK=0)
# Other ranks get WANDB_MODE=disabled to prevent duplicate runs
if int(os.environ.get("LOCAL_RANK", 0)) != 0:
    os.environ["WANDB_MODE"] = "disabled"

from lm_eval.__main__ import cli_evaluate
cli_evaluate()
