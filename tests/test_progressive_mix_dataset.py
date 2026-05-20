"""Unit tests for ProgressiveMixDataset and config parser."""

import importlib.util
import os
import sys
import tempfile

import numpy as np
import pytest


# Load the module directly by file path so we don't trigger
# megatron_patch.data.__init__ which pulls in heavy megatron.training imports
# that aren't relevant to this unit test.
_MODULE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir,
    "megatron_patch", "data", "progressive_mix_dataset.py",
))
_spec = importlib.util.spec_from_file_location("progressive_mix_dataset", _MODULE_PATH)
_pmd = importlib.util.module_from_spec(_spec)
sys.modules["progressive_mix_dataset"] = _pmd  # required by @dataclass introspection
_spec.loader.exec_module(_pmd)

LinearRampSchedule = _pmd.LinearRampSchedule
ProgressiveMixDataset = _pmd.ProgressiveMixDataset
build_iter_to_samples_table = _pmd.build_iter_to_samples_table
parse_progressive_blend_config = _pmd.parse_progressive_blend_config


class _MockDataset:
    """Minimal stand-in for a BlendedDataset returning a tagged dict."""

    def __init__(self, tag: str, length: int = 10_000):
        self.tag = tag
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"tag": self.tag, "i": int(i)}


# ---------------------------------------------------------------------------
# LinearRampSchedule


def test_linear_ramp_endpoints_and_interior():
    s = LinearRampSchedule(
        start_at=100, reach_full_at=300, start_ratio=0.0, final_ratio=0.5
    )
    assert s(0) == 0.0
    assert s(100) == 0.0
    assert s(200) == pytest.approx(0.25)
    assert s(300) == 0.5
    assert s(999) == 0.5


def test_linear_ramp_zero_width():
    s = LinearRampSchedule(start_at=100, reach_full_at=100, start_ratio=0.0, final_ratio=0.3)
    assert s(99) == 0.0
    assert s(100) == 0.0  # boundary uses start_ratio (<=)
    assert s(101) == 0.3


# ---------------------------------------------------------------------------
# build_iter_to_samples_table


def test_iter_to_samples_step_schedule():
    # GBS=10 for first 50 samples, then GBS=20.
    table = build_iter_to_samples_table([(0, 10), (50, 20)], None, horizon_samples=200)
    assert table[0] == 0
    assert table[5] == 50  # 5 iters * 10
    assert table[6] == 70  # +1 iter * 20
    assert table[7] == 90


def test_iter_to_samples_constant():
    table = build_iter_to_samples_table(None, constant_gbs=8, horizon_samples=40)
    assert table == [0, 8, 16, 24, 32, 40]


# ---------------------------------------------------------------------------
# ProgressiveMixDataset


def test_wrapper_size_and_validation():
    base = _MockDataset("base")
    aux = _MockDataset("code")
    sched = LinearRampSchedule(0, 100, 0.0, 0.3)
    ds = ProgressiveMixDataset(base, {"code": aux}, {"code": sched}, seed=42, size=1000)
    assert len(ds) == 1000


def test_wrapper_rejects_oversum():
    base = _MockDataset("base")
    a1 = _MockDataset("a")
    a2 = _MockDataset("b")
    s1 = LinearRampSchedule(0, 100, 0.0, 0.6)
    s2 = LinearRampSchedule(0, 100, 0.0, 0.5)  # 0.6 + 0.5 = 1.1 at idx >= 100
    with pytest.raises(ValueError, match="exceeds 1.0"):
        ProgressiveMixDataset(base, {"a": a1, "b": a2}, {"a": s1, "b": s2}, seed=0, size=200)


def test_wrapper_reproducibility():
    base = _MockDataset("base")
    aux = _MockDataset("code")
    sched = LinearRampSchedule(0, 100, 0.5, 0.5)  # constant 50% aux
    ds1 = ProgressiveMixDataset(base, {"code": aux}, {"code": sched}, seed=7, size=1000)
    ds2 = ProgressiveMixDataset(base, {"code": aux}, {"code": sched}, seed=7, size=1000)
    for i in (0, 1, 2, 100, 999):
        assert ds1[i] == ds2[i]


def test_wrapper_distribution_before_start():
    base = _MockDataset("base")
    aux = _MockDataset("code")
    # ramp starts at 500
    sched = LinearRampSchedule(500, 800, 0.0, 0.5)
    ds = ProgressiveMixDataset(base, {"code": aux}, {"code": sched}, seed=42, size=2000)
    # All draws before idx 500 must come from base.
    for i in range(0, 500):
        assert ds[i]["tag"] == "base"


def test_wrapper_distribution_after_full_ramp():
    np.random.seed(0)
    base = _MockDataset("base")
    aux = _MockDataset("code")
    sched = LinearRampSchedule(0, 100, 0.2, 0.2)  # constant 20%
    ds = ProgressiveMixDataset(base, {"code": aux}, {"code": sched}, seed=123, size=20_000)
    # Sample 5000 idxs and check empirical aux ratio close to 0.2.
    n = 5000
    aux_count = sum(1 for i in range(1000, 1000 + n) if ds[i]["tag"] == "code")
    empirical = aux_count / n
    assert abs(empirical - 0.2) < 0.025  # within ~2.5%


def test_wrapper_emit_dataset_source():
    base = _MockDataset("base")
    aux = _MockDataset("code")
    sched = LinearRampSchedule(0, 100, 1.0, 1.0)  # always aux
    ds = ProgressiveMixDataset(
        base, {"code": aux}, {"code": sched}, seed=0, size=10, emit_dataset_source=True
    )
    s = ds[5]
    assert s["aux_source"] == "code"


# ---------------------------------------------------------------------------
# parse_progressive_blend_config


def test_parse_yaml_tokens_unit(tmp_path):
    yaml_path = tmp_path / "blend.yaml"
    yaml_path.write_text(
        """
base:
  data_path: ["1.0", "/tmp/a"]
schedule_unit: tokens
auxiliary:
  - name: code
    data_path: ["1.0", "/tmp/code"]
    schedule:
      start_at: "100B"
      reach_full_at: "500B"
      start_ratio: 0.0
      final_ratio: 0.1
"""
    )
    cfg = parse_progressive_blend_config(
        str(yaml_path), seq_length=2048, constant_gbs=1024, horizon_samples=10**8
    )
    assert cfg["base"]["data_path"] == ["1.0", "/tmp/a"]
    assert len(cfg["aux"]) == 1
    sched = cfg["aux"][0]["schedule"]
    assert sched.start_at == 100 * 10**9 // 2048
    assert sched.reach_full_at == 500 * 10**9 // 2048
    assert sched.final_ratio == 0.1


def test_parse_yaml_iterations_unit(tmp_path):
    yaml_path = tmp_path / "blend.yaml"
    yaml_path.write_text(
        """
base:
  data_path: ["1.0", "/tmp/a"]
auxiliary:
  - name: math
    data_path: ["1.0", "/tmp/math"]
    schedule:
      unit: iterations
      start_at: 5
      reach_full_at: 10
      start_ratio: 0.0
      final_ratio: 0.05
"""
    )
    # GBS=8 constant; iter 5 => 40 samples, iter 10 => 80 samples
    cfg = parse_progressive_blend_config(
        str(yaml_path), seq_length=128, constant_gbs=8, horizon_samples=200
    )
    sched = cfg["aux"][0]["schedule"]
    assert sched.start_at == 40
    assert sched.reach_full_at == 80


def test_parse_rejects_reserved_aux_name(tmp_path):
    yaml_path = tmp_path / "blend.yaml"
    yaml_path.write_text(
        """
base:
  data_path: ["1.0", "/tmp/a"]
auxiliary:
  - name: base
    data_path: ["1.0", "/tmp/x"]
    schedule:
      start_at: 0
      reach_full_at: 100
      final_ratio: 0.1
"""
    )
    with pytest.raises(ValueError, match="reserved"):
        parse_progressive_blend_config(
            str(yaml_path), seq_length=128, constant_gbs=8, horizon_samples=200
        )
