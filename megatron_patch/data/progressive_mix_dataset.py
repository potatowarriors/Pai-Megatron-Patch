# Copyright (c) 2026 Alpha Project. Licensed under Apache 2.0.
"""Progressive mixing of N auxiliary datasets on top of a base blended dataset.

The wrapper draws each sample from either the base dataset or one of the auxiliary
datasets according to per-aux schedules that vary with global sample index.

Schedule units: ``tokens`` (default, robust to GBS/seq_length), ``samples``, or
``iterations`` (resolved against an optional step-wise GBS schedule).

Selection at each ``__getitem__(idx)`` is deterministic: a numpy RNG seeded by
``(idx, seed)`` decides which dataset to draw from. Combined with Megatron's
sample-index-stable :class:`MegatronPretrainingSampler`, this gives bit-reproducible
training and clean checkpoint resumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch.utils.data

logger = logging.getLogger(__name__)


_SUFFIX_MULT = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}


def _parse_int_with_suffix(value) -> int:
    """Parse int with optional K/M/B/T suffix (str), or pass through int/float."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    if not s:
        raise ValueError("empty integer literal")
    if s[-1].upper() in _SUFFIX_MULT:
        return int(float(s[:-1]) * _SUFFIX_MULT[s[-1].upper()])
    return int(s)


# ---------------------------------------------------------------------------
# Schedule


@dataclass
class LinearRampSchedule:
    """A single auxiliary dataset's per-sample-index ramp."""

    start_at: int        # in *samples* (after unit conversion)
    reach_full_at: int   # in *samples*
    start_ratio: float
    final_ratio: float

    def __post_init__(self) -> None:
        assert 0.0 <= self.start_ratio <= 1.0, self.start_ratio
        assert 0.0 <= self.final_ratio <= 1.0, self.final_ratio
        assert self.start_at >= 0
        assert self.reach_full_at >= self.start_at, (self.start_at, self.reach_full_at)

    def __call__(self, sample_idx: int) -> float:
        if sample_idx <= self.start_at:
            return self.start_ratio
        if sample_idx >= self.reach_full_at:
            return self.final_ratio
        if self.reach_full_at == self.start_at:
            return self.final_ratio
        frac = (sample_idx - self.start_at) / (self.reach_full_at - self.start_at)
        return self.start_ratio + frac * (self.final_ratio - self.start_ratio)


def _to_sample_threshold(
    raw_value,
    unit: str,
    seq_length: int,
    iter_to_samples: Optional[List[int]],
) -> int:
    """Convert a schedule threshold to *samples* given unit semantics."""
    n = _parse_int_with_suffix(raw_value)
    if unit == "samples":
        return n
    if unit == "tokens":
        return n // int(seq_length)
    if unit == "iterations":
        if iter_to_samples is None:
            raise ValueError(
                "schedule_unit='iterations' requires a step batch size schedule "
                "(or constant GBS) so iterations can be mapped to samples."
            )
        if n < len(iter_to_samples):
            return iter_to_samples[n]
        # Beyond the precomputed table, extrapolate using the last known GBS.
        if len(iter_to_samples) >= 2:
            tail_gbs = iter_to_samples[-1] - iter_to_samples[-2]
        else:
            tail_gbs = iter_to_samples[-1] if iter_to_samples else 0
        return iter_to_samples[-1] + (n - len(iter_to_samples) + 1) * tail_gbs
    raise ValueError(f"unknown schedule unit '{unit}'")


def build_iter_to_samples_table(
    step_batch_size_schedule: Optional[List[Tuple[int, int]]],
    constant_gbs: Optional[int],
    horizon_samples: int,
) -> List[int]:
    """Build a list ``T`` such that ``T[i]`` = cumulative samples after ``i`` iterations.

    Either ``step_batch_size_schedule`` (sorted ``[(sample_thr, gbs), ...]``) or
    ``constant_gbs`` must be provided. The table is built up to ``horizon_samples``
    (rounded up by one segment to avoid off-by-one at boundaries).
    """
    table: List[int] = [0]
    if step_batch_size_schedule is not None:
        schedule = sorted(step_batch_size_schedule, key=lambda p: p[0])
        for i, (thr, gbs) in enumerate(schedule):
            seg_end = (
                schedule[i + 1][0] if (i + 1) < len(schedule) else max(horizon_samples, thr + 1)
            )
            while table[-1] < seg_end and table[-1] < horizon_samples:
                table.append(table[-1] + gbs)
                if len(table) > 10_000_000:
                    raise RuntimeError("iter_to_samples table exceeded 10M entries")
        return table
    if constant_gbs is not None and constant_gbs > 0:
        cur = 0
        while cur < horizon_samples:
            cur += constant_gbs
            table.append(cur)
        return table
    raise ValueError("must provide either step_batch_size_schedule or constant_gbs")


# ---------------------------------------------------------------------------
# Wrapper


class ProgressiveMixDataset(torch.utils.data.Dataset):
    """Wrap a base dataset and N auxiliary datasets with per-aux smooth ramp schedules.

    Args:
        base_dataset: Underlying base BlendedDataset (or any torch Dataset returning a dict).
        aux_datasets: ``OrderedDict`` mapping aux name -> dataset. Iteration order matters
            for deterministic cumulative selection.
        aux_schedules: ``Dict[name -> Callable[[int], float]]`` returning per-sample-index
            mixing ratio for that aux. Must have the same keys as ``aux_datasets``.
        seed: Base seed; combined with each idx for deterministic selection.
        size: ``__len__`` to expose. Defaults to ``len(base_dataset)``.
        emit_dataset_source: If True, attach an ``"aux_source"`` key (str name or "base")
            to every returned sample for diagnostic logging.
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        aux_datasets: Dict[str, torch.utils.data.Dataset],
        aux_schedules: Dict[str, Callable[[int], float]],
        seed: int,
        size: Optional[int] = None,
        emit_dataset_source: bool = False,
    ) -> None:
        super().__init__()
        assert set(aux_datasets.keys()) == set(aux_schedules.keys()), (
            "aux_datasets and aux_schedules must have the same keys"
        )
        # Preserve insertion order.
        self.aux_names: List[str] = list(aux_datasets.keys())
        self.base_dataset = base_dataset
        self.aux_datasets = aux_datasets
        self.aux_schedules = aux_schedules
        self.seed = int(seed)
        self.size = int(size) if size is not None else len(base_dataset)
        self.emit_dataset_source = emit_dataset_source
        self._validate_schedules()

    def __len__(self) -> int:
        return self.size

    def _validate_schedules(self) -> None:
        # Sample 200 evenly-spaced points to assert sum(aux) <= 1 across schedule.
        if not self.aux_names:
            return
        n = max(2, min(self.size, 200))
        for k in range(n):
            idx = (self.size - 1) * k // (n - 1)
            total = sum(float(self.aux_schedules[name](idx)) for name in self.aux_names)
            if total > 1.0 + 1e-9:
                raise ValueError(
                    f"sum of aux fractions exceeds 1.0 at idx={idx}: total={total}"
                )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self.size + idx
        # Deterministic per-idx RNG.
        rng = np.random.default_rng(np.array([self.seed, idx], dtype=np.uint64))
        r = float(rng.random())
        cum = 0.0
        for name in self.aux_names:
            p = float(self.aux_schedules[name](idx))
            cum += p
            if r < cum:
                ds = self.aux_datasets[name]
                sample = ds[idx % len(ds)]
                if self.emit_dataset_source:
                    sample = {**sample, "aux_source": name}
                return sample
        sample = self.base_dataset[idx % len(self.base_dataset)]
        if self.emit_dataset_source:
            sample = {**sample, "aux_source": "base"}
        return sample


# ---------------------------------------------------------------------------
# YAML config parsing


def parse_progressive_blend_config(
    config_path: str,
    seq_length: int,
    step_batch_size_schedule: Optional[List[Tuple[int, int]]] = None,
    constant_gbs: Optional[int] = None,
    horizon_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Parse a progressive-blend YAML/JSON config and resolve schedule thresholds to samples.

    Returns a dict with keys:
        - ``base``: dict with ``data_path`` (list)
        - ``aux``: ordered list of dicts ``{name, data_path, schedule: LinearRampSchedule}``

    Threshold conversion:
        Each aux's ``schedule.unit`` (default = top-level ``schedule_unit`` or ``"tokens"``)
        decides whether ``start_at``/``reach_full_at`` are in tokens, samples, or iterations.
        For ``iterations``, an iter→samples table is precomputed from the GBS schedule.
    """
    import yaml

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"progressive blend config must be a YAML mapping, got {type(raw)}")

    if "base" not in raw or "data_path" not in raw["base"]:
        raise ValueError("progressive blend config must define base.data_path")

    default_unit = str(raw.get("schedule_unit", "tokens"))
    if default_unit not in {"tokens", "samples", "iterations"}:
        raise ValueError(f"schedule_unit must be tokens|samples|iterations, got '{default_unit}'")

    iter_to_samples = None
    if any(
        (a.get("schedule", {}).get("unit", default_unit) == "iterations")
        for a in raw.get("auxiliary", []) or []
    ) or default_unit == "iterations":
        if horizon_samples is None:
            raise ValueError(
                "horizon_samples must be provided when any aux uses unit='iterations'"
            )
        iter_to_samples = build_iter_to_samples_table(
            step_batch_size_schedule, constant_gbs, horizon_samples
        )

    aux_list: List[Dict[str, Any]] = []
    for entry in raw.get("auxiliary", []) or []:
        if "name" not in entry or "data_path" not in entry or "schedule" not in entry:
            raise ValueError(f"aux entry must have name/data_path/schedule, got: {entry}")
        sched = entry["schedule"]
        unit = str(sched.get("unit", default_unit))
        start_samples = _to_sample_threshold(
            sched["start_at"], unit, seq_length, iter_to_samples
        )
        full_samples = _to_sample_threshold(
            sched["reach_full_at"], unit, seq_length, iter_to_samples
        )
        schedule_obj = LinearRampSchedule(
            start_at=start_samples,
            reach_full_at=full_samples,
            start_ratio=float(sched.get("start_ratio", 0.0)),
            final_ratio=float(sched["final_ratio"]),
        )
        aux_list.append(
            {
                "name": str(entry["name"]),
                "data_path": list(entry["data_path"]),
                "schedule": schedule_obj,
            }
        )

    seen = set()
    for a in aux_list:
        if a["name"] in seen:
            raise ValueError(f"duplicate aux name '{a['name']}'")
        seen.add(a["name"])
        if a["name"] == "base":
            raise ValueError("aux name 'base' is reserved")

    return {
        "base": {"data_path": list(raw["base"]["data_path"])},
        "aux": aux_list,
    }
