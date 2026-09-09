"""--mid-level-dataset-surplus 가 alpha 데이터 제공자(core_gpt_dataset_config_from_args)를 거쳐 GPTDatasetConfig 에 닿는지.

배경 (2026-09-09, phase-3 스모크 IndexError): BlendedMegatronDatasetBuilder 는 top-level 블렌드 크기를 멤버별 ceil(w×N) 의 합으로
잡는다. 멤버가 51 이면 valid N=3,200 이 3,241 로 부풀지만 멤버 버퍼는 ceil(w×N×(1+surplus)) 라 기본 0.005 로는 큰 멤버가 모자란다.
제공자가 인자를 config 로 넘기지 않으면 YAML 의 값이 조용히 무시되므로 여기서 고정한다.
"""
import math
import os
import sys
import types

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for p in (REPO, os.path.join(REPO, "backends", "megatron", "Megatron-LM-251125")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _args(**over):
    a = types.SimpleNamespace(
        seed=1234, seq_length=128, data_path=["1.0", "/tmp/nonexistent/data_text_document"],
        train_data_path=None, valid_data_path=None, test_data_path=None, split="99,1,0", data_cache_path=None,
        reset_position_ids=True, reset_attention_mask=False, eod_mask_loss=False, mmap_bin_files=True,
        create_attention_mask_in_dataloader=False, num_dataset_builder_threads=1,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def provider(monkeypatch):
    mod = pytest.importorskip("megatron_patch.data")
    monkeypatch.setattr(mod, "get_tokenizer", lambda: object())
    return mod


def test_surplus_reaches_config(provider):
    cfg = provider.core_gpt_dataset_config_from_args(_args(mid_level_dataset_surplus=0.05))
    assert cfg.mid_level_dataset_surplus == 0.05


def test_surplus_default_when_arg_absent(provider):
    cfg = provider.core_gpt_dataset_config_from_args(_args())
    assert cfg.mid_level_dataset_surplus == 0.005


def test_phase3_valid_blend_inflation_needs_surplus():
    """실제 phase-3 블렌드 51 멤버, valid N=3,200: 기본 여유분은 모자라고(스모크 재현) 프리셋 값 0.05 는 통과."""
    helpers = pytest.importorskip("megatron.core.datasets.helpers")
    import numpy as np
    import yaml

    y = os.path.join(REPO, "examples", "alpha", "configs", "data", "sft_128k_terminal_blend_p3.yaml")
    if not os.path.exists(y):
        pytest.skip("phase-3 blend yaml 없음")
    toks = yaml.safe_load(open(y))["data-path"].split()
    w = np.array([float(x) for x in toks[0::2]]); w = w / w.sum()
    N = 3200
    size = sum(math.ceil(x * N) for x in w)          # builder: size_i = sum(sizes_per_dataset_target)
    assert size > N
    di = np.zeros(size, dtype=np.int16); si = np.zeros(size, dtype=np.int64)
    helpers.build_blending_indices(di, si, w, len(w), size, False)
    cnt = np.bincount(di, minlength=len(w))
    over = lambda s: max(int(cnt[i]) - math.ceil(w[i] * N * (1 + s)) for i in range(len(w)))
    assert over(0.005) > 0
    assert over(0.05) <= 0
