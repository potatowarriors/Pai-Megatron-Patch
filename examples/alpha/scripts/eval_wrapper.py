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

# datasets>=3.0에서 제거된 load_metric을 lm_eval scrolls 태스크가 import함
# (2026-08-26 main1 datasets 3.6.0에서 ImportError). scrolls는 안 쓰므로 스텁으로 통과.
import datasets  # noqa: E402
if not hasattr(datasets, "load_metric"):
    def _load_metric_removed(*a, **k):
        raise RuntimeError("datasets.load_metric was removed; scrolls tasks unsupported here")
    datasets.load_metric = _load_metric_removed

from lm_eval.__main__ import cli_evaluate
cli_evaluate()
