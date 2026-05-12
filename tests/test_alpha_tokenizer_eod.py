"""Regression tests guarding the alpha v5 tokenizer EOD contract.

History: a 79 h DCLM tokenization completed with `--append-eod` on the CLI but
produced a `.bin` containing zero EOD markers. Root cause was a transient
local-only state of `preprocess_data_megatron.py` + `megatron_patch/tokenizer/__init__.py`
during the 2026-05 alpha v5 migration (see `tests/preflight_stage1/01_eod_repro.md`).

These tests close the verification net so the same class of silent failure
cannot recur — they fail loudly if (a) the tokenizer's designated EOS drifts
back to a chat-only token, (b) `_AlphaTokenizer.eod` regresses, (c) the
preprocess Encoder stops appending EOD when asked.
"""

import argparse
import json
import os
import sys
import tempfile

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "examples", "alpha", "tokenizer_v5")
_MEGATRON_PATH = os.path.join(_REPO_ROOT, "backends", "megatron", "Megatron-LM-251125")
_PREPROCESS_DIR = os.path.join(_REPO_ROOT, "toolkits", "pretrain_data_preprocessing")

# Make repo importable for `megatron_patch` + Megatron core (for IndexedDataset).
for _p in (_REPO_ROOT, _MEGATRON_PATH, _PREPROCESS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# A. Tokenizer-config contract
# ---------------------------------------------------------------------------


def test_tokenizer_config_eos_is_endoftext():
    """The tokenizer config must designate <|endoftext|> as EOS, not <|im_end|>.

    Why: pre-training EOD must be separated from chat-turn-end marker, per
    frontier convention (Qwen3, Llama 3, DSV3). Drifting back to <|im_end|>
    would (a) conflate doc-end with chat-turn-end at pre-training, and (b)
    break HF generation_config inheritance for SFT models.
    """
    with open(os.path.join(_TOKENIZER_PATH, "tokenizer_config.json")) as f:
        cfg = json.load(f)
    assert cfg["eos_token"] == "<|endoftext|>", (
        f"tokenizer_config.json:eos_token must be '<|endoftext|>' "
        f"(alpha v5 pre-training EOS/EOD per Phase 0.0); got {cfg['eos_token']!r}"
    )
    # Defensive: bos must be null (alpha has no BOS), pad must be <|pad|>.
    assert cfg["bos_token"] is None
    assert cfg["pad_token"] == "<|pad|>"


def test_special_tokens_map_eos_is_endoftext():
    """special_tokens_map.json must agree with tokenizer_config.json on EOS.

    Why: vLLM / SGLang / some chat-template utilities read this file directly
    (not via HF AutoTokenizer), so an inconsistency between the two files
    silently causes wrong-stop-token behavior at serving time. This test
    locks the two files to the same designation.
    """
    with open(os.path.join(_TOKENIZER_PATH, "special_tokens_map.json")) as f:
        smap = json.load(f)
    assert smap["eos_token"] == "<|endoftext|>", (
        f"special_tokens_map.json:eos_token must match tokenizer_config.json — "
        f"got {smap['eos_token']!r}"
    )
    assert smap["pad_token"] == "<|pad|>"


def test_hf_tokenizer_resolves_eos_to_id_0():
    """AutoTokenizer must resolve the new EOS to id 0 in alpha v5 vocab."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_TOKENIZER_PATH, use_fast=True, trust_remote_code=False)
    assert tok.eos_token == "<|endoftext|>"
    assert tok.eos_token_id == 0
    assert tok.pad_token_id == 1
    assert tok.bos_token_id is None
    # <|im_end|> must still be in vocab (just not designated EOS).
    assert tok.decode([3], skip_special_tokens=False) == "<|im_end|>"


# ---------------------------------------------------------------------------
# B. _AlphaTokenizer wrapper contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def alpha_tokenizer():
    """Build _AlphaTokenizer via the real build_tokenizer entry point."""
    from megatron_patch.tokenizer import build_tokenizer
    args = argparse.Namespace(
        patch_tokenizer_type="AlphaTokenizer",
        load=_TOKENIZER_PATH,
        extra_vocab_size=0,
        padded_vocab_size=163968,
        rank=0,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
    )
    return build_tokenizer(args)


def test_alpha_tokenizer_eod_returns_zero(alpha_tokenizer):
    """The Megatron-facing .eod property must match the tokenizer-config EOS.

    `_AlphaTokenizer.eod` is a @property delegating to `tokenizer.eos_token_id`.
    If anyone reassigns it to a literal (as the historical generation.py:162
    pattern did for non-property tokenizers), the property would shadow and
    return the wrong value silently.
    """
    assert alpha_tokenizer.eod == 0
    assert alpha_tokenizer.eos_token_id == 0
    assert alpha_tokenizer.pad_token_id == 1


def test_alpha_tokenizer_roundtrip(alpha_tokenizer):
    samples = [
        "Hello world.",
        "한국어 문서 샘플입니다.",
        "def foo():\n    return 42",
        "العربية النص",
    ]
    for text in samples:
        ids = alpha_tokenizer.tokenize(text)
        decoded = alpha_tokenizer.detokenize(ids)
        assert decoded == text, f"round-trip failed for {text!r}: got {decoded!r}"


# ---------------------------------------------------------------------------
# C. preprocess_data_megatron.py Encoder.encode must append EOD when asked
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def encoder_with_append_eod():
    """Replicate preprocess_data_megatron.py's Encoder + initializer."""
    from preprocess_data_megatron import Encoder
    args = argparse.Namespace(
        patch_tokenizer_type="AlphaTokenizer",
        load=_TOKENIZER_PATH,
        extra_vocab_size=0,
        padded_vocab_size=163968,
        append_eod=True,
        rank=0,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
        json_keys=["text"],
        split_sentences=False,
    )
    e = Encoder(args)
    e.initializer()
    return e


def test_preprocess_encoder_appends_eod(encoder_with_append_eod):
    """If --append-eod is on, every non-empty doc must end with id 0."""
    samples = [
        '{"text": "Hello world."}',
        '{"text": "한국어 sample"}',
        '{"text": "def x(): return 1"}',
    ]
    for s in samples:
        ids, lens, _, _ = encoder_with_append_eod.encode(s)
        doc_ids = ids["text"]
        assert len(doc_ids) > 0
        assert doc_ids[-1] == 0, (
            f"last token must be EOD (0) when --append-eod is set; "
            f"got {doc_ids[-1]} for input {s!r}"
        )
        # sentence_lens last entry must account for the appended EOD.
        assert sum(lens["text"]) == len(doc_ids)


def test_preprocess_encoder_no_eod_when_flag_off():
    """If --append-eod is off, docs must NOT have a trailing id 0
    (other than coincidentally — but our test inputs don't include the
    literal <|endoftext|> token, so id 0 must not appear at all)."""
    from preprocess_data_megatron import Encoder
    args = argparse.Namespace(
        patch_tokenizer_type="AlphaTokenizer",
        load=_TOKENIZER_PATH,
        extra_vocab_size=0,
        padded_vocab_size=163968,
        append_eod=False,  # critical
        rank=0,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
        json_keys=["text"],
        split_sentences=False,
    )
    e = Encoder(args)
    e.initializer()
    ids, _, _, _ = e.encode('{"text": "Hello world."}')
    doc_ids = ids["text"]
    assert 0 not in doc_ids, (
        f"--append-eod is off but EOD (0) appeared in tokens: {doc_ids}"
    )


# ---------------------------------------------------------------------------
# D. alpha_config.py TokenConfig defaults must reflect alpha v5
# ---------------------------------------------------------------------------


def test_alpha_config_token_defaults_are_alpha_v5():
    """`generate-hf-config` emits these into the model's HF config.json.

    If they drift back to Qwen3 IDs (151643, 151645), HF inference / SGLang
    serving / chat templates would all silently stop at the wrong token.
    """
    spec_dir = os.path.join(_REPO_ROOT, "examples", "alpha", "tools")
    if spec_dir not in sys.path:
        sys.path.insert(0, spec_dir)
    import alpha_config
    tc = alpha_config.TokenConfig()
    assert tc.bos_token_id is None, "alpha has no BOS — bos_token_id must be None"
    assert tc.eos_token_id == 0, "alpha EOS/EOD = <|endoftext|> id 0"
    assert tc.pad_token_id == 1, "alpha pad = <|pad|> id 1"


# ---------------------------------------------------------------------------
# E. End-to-end through IndexedDatasetBuilder (catches finalize() drift)
# ---------------------------------------------------------------------------


def test_preprocess_end_to_end_writes_eod_in_bin():
    """A round-trip through preprocess_data_megatron.py + IndexedDataset
    reader must see EOD as the final token of every doc.

    This is the actual assertion that historical preprocessing failed: the
    encode path correctly appended EOD to its in-memory list, but the
    resulting .bin still has docs ending in non-EOD tokens. This test runs
    the full pipeline (subprocess + read back) so the assertion mirrors
    what Phase 0.4's data-injection postcondition will check.
    """
    import subprocess
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "sample.jsonl")
        with open(in_path, "w") as f:
            for text in ["Hello world.", "한국어 doc", "third one"]:
                f.write(json.dumps({"text": text}) + "\n")
        out_prefix = os.path.join(td, "data")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{_REPO_ROOT}:{_MEGATRON_PATH}:" + env.get("PYTHONPATH", "")
        cmd = [
            sys.executable,
            os.path.join(_PREPROCESS_DIR, "preprocess_data_megatron.py"),
            "--input", in_path,
            "--output-prefix", out_prefix,
            "--json-keys", "text",
            "--patch-tokenizer-type", "AlphaTokenizer",
            "--load", _TOKENIZER_PATH,
            "--workers", "1",
            "--partitions", "1",
            "--append-eod",
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, (
            f"preprocess_data_megatron.py failed:\n{result.stderr}"
        )

        ds = IndexedDataset(out_prefix + "_text_document")
        assert int(ds.sequence_lengths.shape[0]) == 3
        for i in range(3):
            tokens = ds.get(i)
            assert tokens[-1] == 0, (
                f"doc {i} last token = {int(tokens[-1])}, expected 0 (EOD). "
                f"This is the exact regression that produced Stage 1's "
                f"historical EOD-less .bin files."
            )


# ---------------------------------------------------------------------------
# F. hf_model/configuration_alpha.py — AlphaConfig() defaults must match baseline_48L
# ---------------------------------------------------------------------------


def test_configuration_alpha_defaults_match_baseline_48L():
    """no-kwargs `AlphaConfig()` must yield a model matching baseline_48L.yaml.

    History: the 2026-05 Qwen3.5+DSV3+v5 migration updated baseline_48L.yaml but
    overlooked the mirrored defaults in `hf_model/configuration_alpha.py` (the
    `AlphaConfig.__init__` signature). 1st-pass preflight (F_decisions.md Item 12)
    deferred this as cosmetic; the 2nd-pass (2026-05-12) promoted it to a fix
    because any HF/SGLang serving path that calls `AlphaConfig.from_pretrained`
    on a config.json missing one of these keys falls back to the (then-stale)
    default — silent corruption at deploy time.

    This test pins each of the 7 corrected defaults so future migration drift
    fails CI loudly instead of silently.
    """
    hf_model_dir = os.path.join(_REPO_ROOT, "examples", "alpha", "hf_model")
    if hf_model_dir not in sys.path:
        sys.path.insert(0, hf_model_dir)
    from configuration_alpha import AlphaConfig

    c = AlphaConfig()
    assert c.vocab_size == 163968, f"vocab_size default drift: {c.vocab_size}"
    assert c.intermediate_size == 8192, f"intermediate_size default drift: {c.intermediate_size}"
    assert c.max_position_embeddings == 262144, (
        f"max_position_embeddings default drift: {c.max_position_embeddings}"
    )
    assert c.rope_theta == 10_000_000.0, f"rope_theta default drift: {c.rope_theta}"
    assert c.num_experts_per_tok == 8, f"num_experts_per_tok default drift: {c.num_experts_per_tok}"
    assert c.num_experts == 184, f"num_experts default drift: {c.num_experts}"
    assert c.router_aux_loss_coef == 1.0e-4, (
        f"router_aux_loss_coef default drift: {c.router_aux_loss_coef}"
    )
    # Sanity: the defaults that should not drift either.
    assert c.hidden_size == 2048
    assert c.num_hidden_layers == 48
    assert c.head_dim == 256
    assert c.moe_intermediate_size == 512
    assert c.shared_expert_intermediate_size == 512
