#!/usr/bin/env python3
"""tau_tasks_count.py — 설치된 tau2-bench 의 도메인별 과제 수를 split 별로 찍는다 (설치 검증·문서 수치 정본).

사용: /home/work/vidsearch/tools/tau2-bench/.venv/bin/python eval_sft/tau_tasks_count.py [domain ...]
  기본 도메인: mock airline retail telecom. `base` split 이 없는 도메인은 그렇게 표시한다
  (telecom 은 small/train/test/full 만 있을 수 있다 — run_tau.sh 는 그 경우 test 로 대체하고 기록).
"""
import json
import logging
import sys

logging.disable(logging.CRITICAL)
try:
    from loguru import logger
    logger.remove()
except Exception:  # noqa: BLE001
    pass

from tau2.registry import registry  # noqa: E402

doms = sys.argv[1:] or ["mock", "airline", "retail", "telecom"]
out = {}
for d in doms:
    row = {}
    try:
        splits = registry.get_task_splits_loader(d)()
        row["splits"] = {k: len(v) for k, v in splits.items()}
    except Exception as e:  # noqa: BLE001
        row["splits"] = f"(no splits: {type(e).__name__})"
    loader = registry.get_tasks_loader(d)
    for name in ("base", None):
        try:
            row["base" if name else "default"] = len(loader(name) if name else loader())
        except Exception as e:  # noqa: BLE001
            row["base" if name else "default"] = f"ERR {type(e).__name__}: {str(e)[:80]}"
    out[d] = row
print(json.dumps(out, ensure_ascii=False, indent=1))
