"""Regression tests guarding the alpha v2 evaluation pipeline's config plumbing.

History: the v1→v2 migration (tokenizer v5, MoE 128→192 experts, head_dim
128→256, moe-ffn 768→512, score softmax→sigmoid) left the *evaluation* path
behind. The same model config lived in three places that silently drifted:
  - training YAML (configs/model/baseline_48L.yaml)          — correct (192)
  - convert .sh  (scripts/alpha/configs/baseline_48L.sh)     — stale (128)
  - validate.sh  (hardcoded vocab 151936 / Qwen3Tokenizer / nested YAML parse)

The fix single-sources everything from the checkpoint's common.pt via
`alpha_config.emit_megatron_flags` / `generate_hf_config`. These tests fail
loudly if (a) the YAML config diverges from the trained checkpoint, (b) the
generated HF config.json regresses a v2 field, (c) emit-megatron-flags drops the
mamba dims / expert count, or (d) a stale v1 tokenizer path / class reappears in
the convert/validate/evaluate scripts.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_ALPHA_DIR = os.path.join(_REPO_ROOT, "examples", "alpha")
_TOOLS_DIR = os.path.join(_ALPHA_DIR, "tools")
_MEGATRON_PATH = os.path.join(_REPO_ROOT, "backends", "megatron", "Megatron-LM-251125")

for _p in (_REPO_ROOT, _MEGATRON_PATH, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alpha_config as ac  # noqa: E402

# The first alpha_v2 checkpoint. Checkpoint-bound tests skip when absent (CI).
_CKPT = os.path.join(
    _ALPHA_DIR,
    "outputs",
    "alpha_baseline_48L_stage1_20260512_170157",
    "checkpoints",
    "iter_0010000",
)

# Ground truth recorded from common.pt (the immutable record of what was trained).
_GROUND_TRUTH = {
    "num_layers": 48,
    "hidden_size": 2048,
    "ffn_hidden_size": 8192,
    "num_attention_heads": 16,
    "kv_channels": 256,
    "num_query_groups": 2,
    "num_experts": 192,
    "moe_ffn_hidden_size": 512,
    "moe_router_topk": 8,
    "padded_vocab_size": 163968,
    "hybrid_override_pattern": "M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-",
    "mamba_head_dim": 128,
    "mamba_state_dim": 128,
    "mamba_num_heads": 32,
    "rotary_base": 10000000,
}


# ---------------------------------------------------------------------------
# A. YAML config must agree with the trained checkpoint
# ---------------------------------------------------------------------------


def test_yaml_config_matches_checkpoint_structure():
    """configs/model/baseline_48L.yaml must match what was actually trained.

    This is the drift tripwire: if someone edits the YAML (or the checkpoint
    changes), the structural fields must stay in lock-step or eval converts a
    structurally-wrong model.
    """
    if not os.path.exists(os.path.join(_CKPT, "common.pt")):
        pytest.skip(f"checkpoint not present: {_CKPT}")
    yaml_cfg = ac.load_config("baseline_48L")
    ckpt_cfg = ac.load_config_from_checkpoint(_CKPT)

    assert yaml_cfg.num_layers == ckpt_cfg.num_layers == _GROUND_TRUTH["num_layers"]
    assert yaml_cfg.kv_channels == ckpt_cfg.kv_channels == _GROUND_TRUTH["kv_channels"]
    assert yaml_cfg.num_attention_heads == ckpt_cfg.num_attention_heads
    assert yaml_cfg.moe.num_experts == ckpt_cfg.moe.num_experts == _GROUND_TRUTH["num_experts"]
    assert yaml_cfg.moe.moe_ffn_hidden_size == ckpt_cfg.moe.moe_ffn_hidden_size == 512
    assert yaml_cfg.padded_vocab_size == ckpt_cfg.padded_vocab_size == 163968
    assert yaml_cfg.get_pattern() == ckpt_cfg.get_pattern() == _GROUND_TRUTH["hybrid_override_pattern"]
    assert ckpt_cfg.hybrid.mamba_head_dim == _GROUND_TRUTH["mamba_head_dim"]


def test_checkpoint_load_recovers_dsv3_routing():
    """common.pt-derived config must carry the DSV3 routing the run actually used
    (sigmoid + group-limited + aux-loss-free bias), not aux_loss/softmax defaults."""
    if not os.path.exists(os.path.join(_CKPT, "common.pt")):
        pytest.skip(f"checkpoint not present: {_CKPT}")
    cfg = ac.load_config_from_checkpoint(_CKPT)
    assert cfg.moe.router_score_function == "sigmoid"
    assert cfg.moe.router_num_groups == 8
    assert cfg.moe.router_group_topk == 4
    assert cfg.moe.router_topk_scaling_factor == 2.5
    assert cfg.moe.router_enable_expert_bias is True
    assert cfg.moe.shared_expert_gate is True


# ---------------------------------------------------------------------------
# B. Generated HF config.json must carry v2 fields
# ---------------------------------------------------------------------------


def test_generate_hf_config_v2_fields():
    hf = ac.generate_hf_config(ac.load_config("baseline_48L"))
    assert hf["num_experts"] == 192
    assert hf["num_experts_per_tok"] == 8
    assert hf["vocab_size"] == 163968
    assert hf["head_dim"] == 256
    assert hf["num_hidden_layers"] == 24
    assert hf["moe_intermediate_size"] == 512
    assert hf["shared_expert_intermediate_size"] == 512
    assert hf["rope_theta"] == 10000000
    assert hf["partial_rotary_factor"] == 0.25
    assert hf["max_position_embeddings"] == 262144
    assert hf["eos_token_id"] == 0
    # Alpha has no BOS: the key must be omitted (not emitted as null).
    assert "bos_token_id" not in hf
    # DSV3 routing must be carried into config.json (else HF eval routes wrongly).
    assert hf["scoring_func"] == "sigmoid"
    assert hf["n_group"] == 8
    assert hf["topk_group"] == 4
    assert hf["routed_scaling_factor"] == 2.5


def test_modeling_alpha_moe_is_dsv3_routing():
    """The HF MoE block must implement DSV3 routing (sigmoid + group-limited +
    aux-loss-free bias), not plain softmax/top-k, and expose the bias buffer the
    converter copies into."""
    import torch

    sys.path.insert(0, _ALPHA_DIR)
    from hf_model.configuration_alpha import AlphaConfig
    from hf_model.modeling_alpha import AlphaSparseMoeBlock

    cfg = AlphaConfig(
        hidden_size=32, num_experts=8, num_experts_per_tok=2, n_group=2, topk_group=1,
        routed_scaling_factor=2.5, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, num_hidden_layers=2,
    )
    blk = AlphaSparseMoeBlock(cfg).eval()
    # bias buffer exists, is persistent (so convert→save_pretrained round-trips)
    assert hasattr(blk.gate, "e_score_correction_bias")
    assert any(k.endswith("gate.e_score_correction_bias") for k in blk.state_dict())
    # group-limited: every token's top-k experts live within its selected group(s)
    for row in blk._select_experts(torch.rand(5, 8)):
        grp = row // 4
        assert (grp == grp[0]).all(), f"selection crossed groups: {row.tolist()}"
    # aux-loss-free bias steers selection
    blk.gate.e_score_correction_bias[7] = 100.0
    assert 7 in blk._select_experts(torch.zeros(1, 8))[0].tolist()


def test_modeling_alpha_rmsnorm_is_standard_not_1p():
    """AlphaRMSNorm must apply standard `x_norm * gamma`, NOT the zero-centered
    `x_norm * (1 + gamma)` form inherited from Qwen3-Next.

    Alpha v2 trains Megatron with apply-layernorm-1p OFF
    (layernorm_zero_centered_gamma=False), so the checkpoint stores standard
    gammas (~1.0) and Megatron applies `x * gamma`. The `(1 + gamma)` form scaled
    every norm by ~1.7-2.5x and made all benchmarks random (ARC-easy = 25%) while
    weight-only validation still passed. This guards against that regression and
    keeps AlphaRMSNorm consistent with AlphaRMSNormGated (also standard)."""
    import torch

    sys.path.insert(0, _ALPHA_DIR)
    from hf_model.modeling_alpha import AlphaRMSNorm

    norm = AlphaRMSNorm(4).eval()
    # default init must be ones (identity gain), not zeros (which only makes sense
    # for the zero-centered `1 + gamma` convention)
    assert torch.allclose(norm.weight, torch.ones(4)), "AlphaRMSNorm gamma must init to 1.0"

    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = norm(x)
        normed = norm._norm(x.float())
        expected_standard = (normed * 0.5).type_as(x)        # x_norm * gamma
        expected_1p = (normed * (1.0 + 0.5)).type_as(x)      # x_norm * (1 + gamma)  <-- bug

    assert torch.allclose(out, expected_standard, atol=1e-6), "AlphaRMSNorm must be standard `x*gamma`"
    assert not torch.allclose(out, expected_1p, atol=1e-4), "AlphaRMSNorm must NOT be zero-centered `x*(1+gamma)`"


# ---------------------------------------------------------------------------
# C. emit-megatron-flags must include everything needed to build the skeleton
# ---------------------------------------------------------------------------


def _flag_value(flags, name):
    for i, tok in enumerate(flags):
        if tok == name:
            return flags[i + 1] if i + 1 < len(flags) else True
    return None


def test_emit_flags_include_mamba_and_experts_from_yaml():
    """The historical bug: mamba dims + max-position were never passed to the
    converter (defaults 64 / 4096 != trained 128 / 262144)."""
    flags = ac.emit_megatron_flags(ac.load_config("baseline_48L"))
    assert _flag_value(flags, "--num-experts") == "192"
    assert _flag_value(flags, "--moe-ffn-hidden-size") == "512"
    assert _flag_value(flags, "--kv-channels") == "256"
    assert _flag_value(flags, "--mamba-head-dim") == "128"
    assert _flag_value(flags, "--mamba-state-dim") == "128"
    assert _flag_value(flags, "--mamba-num-heads") == "32"
    assert _flag_value(flags, "--max-position-embeddings") == "262144"
    assert _flag_value(flags, "--padded-vocab-size") == "163968"
    pattern = _flag_value(flags, "--hybrid-override-pattern")
    assert pattern == _GROUND_TRUTH["hybrid_override_pattern"]
    assert len(pattern) == 48 and "D" not in pattern  # no Dense MLP in v2


def test_emit_flags_from_checkpoint_carry_routing():
    if not os.path.exists(os.path.join(_CKPT, "common.pt")):
        pytest.skip(f"checkpoint not present: {_CKPT}")
    flags = ac.emit_megatron_flags(ac.load_config_from_checkpoint(_CKPT))
    assert _flag_value(flags, "--num-experts") == "192"
    assert _flag_value(flags, "--moe-router-score-function") == "sigmoid"
    assert _flag_value(flags, "--moe-router-num-groups") == "8"
    assert "--moe-router-enable-expert-bias" in flags
    assert "--moe-shared-expert-gate" in flags
    assert "--disable-bias-linear" in flags


# ---------------------------------------------------------------------------
# D. No stale v1 references in the convert/validate/evaluate scripts
# ---------------------------------------------------------------------------


def test_no_stale_v1_references_in_scripts():
    """Guards against the old TOKENIZER_PATH=${ALPHA_DIR}/tokenizer (v1 dir),
    Qwen3Tokenizer, and vocab 151936 creeping back into the eval path."""
    targets = {
        "validate.sh": os.path.join(_ALPHA_DIR, "validate.sh"),
        "evaluate.sh": os.path.join(_ALPHA_DIR, "evaluate.sh"),
        "run_convert.sh": os.path.join(
            _REPO_ROOT,
            "toolkits",
            "distributed_checkpoints_convertor",
            "scripts",
            "alpha",
            "run_convert.sh",
        ),
    }
    forbidden = ["151936", "Qwen3Tokenizer", "tokenizer_v4", '/tokenizer"']
    for name, path in targets.items():
        assert os.path.exists(path), f"missing script: {path}"
        text = open(path).read()
        for bad in forbidden:
            assert bad not in text, f"{name} still references stale v1 token {bad!r}"


def test_configuration_alpha_default_num_experts_is_192():
    """The HF AlphaConfig no-kwargs default must match the trained 192 (footgun
    when instantiated without a config.json)."""
    cfg_path = os.path.join(_ALPHA_DIR, "hf_model", "configuration_alpha.py")
    text = open(cfg_path).read()
    assert "num_experts=192," in text
    assert "num_experts=184," not in text


# ---------------------------------------------------------------------------
# fp32 router bias: faithful conversion + strict validation
# ---------------------------------------------------------------------------
# History: the MG↔HF weight validator failed exactly one comparison
# (14180/14181) on router.expert_bias of layer 0 (magnitude ~4.5). Root cause was
# NOT the converter — it already saves the bias fp32, inheriting MG's fp32 source
# dtype (Megatron keeps router.expert_bias in fp32, router.py). The bug was the
# validator's HF *load*: from_pretrained(torch_dtype=bfloat16) homogenised every
# tensor to bf16, downcasting the fp32 bias and creating a ~0.015 phantom diff
# (bf16 ulp/2 at ~4.5) that grew every iteration. Fix keeps the bias fp32
# end-to-end so the faithful fp32-vs-fp32 comparison is exact and the validator
# stays strict (no tolerance loosening):
#   - modeling: e_score_correction_bias is an nn.Parameter (buffers are NOT
#     protected by _keep_in_fp32_modules_strict) listed in that strict flag.
#   - validation: strict tolerance, unchanged from the original.

def _load_compare_tensors():
    import importlib.util
    path = os.path.join(_ALPHA_DIR, "validate_mg_hf_full.py")
    spec = importlib.util.spec_from_file_location("_vmhf", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compare_tensors


def test_modeling_alpha_router_bias_is_fp32_parameter():
    """e_score_correction_bias must be an nn.Parameter (DSV3-style) listed in
    _keep_in_fp32_modules_strict — buffers are NOT kept fp32 by that flag, so a
    bf16 load would silently downcast the fp32 router bias and reintroduce the
    routing drift / validation failure."""
    src = open(os.path.join(_ALPHA_DIR, "hf_model", "modeling_alpha.py")).read()
    assert "self.gate.e_score_correction_bias = nn.Parameter(" in src, \
        "router bias must be an nn.Parameter, not register_buffer"
    assert '"e_score_correction_bias", torch.zeros' not in src, \
        "router bias must no longer be a register_buffer"
    assert '_keep_in_fp32_modules_strict = ["e_score_correction_bias"]' in src, \
        "router bias must be kept fp32 across bf16 loads via _keep_in_fp32_modules_strict"


def test_keep_in_fp32_modules_strict_protects_param_not_buffer():
    """Lock the transformers behaviour the fix relies on: under a bf16 load,
    _keep_in_fp32_modules_strict keeps a same-named *parameter* fp32 but does NOT
    protect a buffer. A transformers bump that changes this would silently
    downcast the router bias — catch it here, not in a multi-hour eval run."""
    import torch, tempfile
    import torch.nn as nn
    from transformers import PreTrainedModel, PretrainedConfig

    class _C(PretrainedConfig):
        model_type = "toy_kif32"

    class _G(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 8))
            self.e_score_correction_bias = nn.Parameter(torch.full((4,), 4.5), requires_grad=False)
            self.register_buffer("buf_bias", torch.full((4,), 4.5), persistent=True)

    class _M(PreTrainedModel):
        config_class = _C
        base_model_prefix = "toy"
        _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

        def __init__(self, c):
            super().__init__(c)
            self.gate = _G()
            self.post_init()

        def forward(self, x):
            return x

    d = tempfile.mkdtemp()
    _M(_C()).save_pretrained(d)
    m = _M.from_pretrained(d, torch_dtype=torch.bfloat16)
    assert m.gate.weight.dtype == torch.bfloat16, "ordinary weight should load bf16"
    assert m.gate.e_score_correction_bias.dtype == torch.float32, "_strict must keep the param fp32"
    assert m.gate.buf_bias.dtype == torch.bfloat16, "buffers are not protected by _strict"


def test_compare_tensors_is_strict():
    """The validator must stay strict (the original criterion, not relaxed). A
    faithful fp32-vs-fp32 comparison is exact and passes; a real +0.5 drift fails;
    and — the point of the fp32-bias fix — a bf16 downcast of a ~4.5 fp32 tensor
    FAILS strict, which is exactly why the bias is kept fp32 end-to-end rather
    than waved through with a wider tolerance."""
    import torch
    compare_tensors = _load_compare_tensors()
    bias = torch.linspace(4.28, 4.56, 192, dtype=torch.float32)
    assert compare_tensors(bias, bias.clone(), "mg", "hf").matched, "fp32-vs-fp32 must be exact"
    assert not compare_tensors(bias, bias + 0.5, "mg", "hf").matched, "+0.5 drift must fail"
    assert not compare_tensors(bias, bias.to(torch.bfloat16), "mg", "hf").matched, \
        "bf16 downcast of a ~4.5 fp32 bias must fail strict (justifies keeping it fp32)"
