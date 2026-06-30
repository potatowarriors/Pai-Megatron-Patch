#!/usr/bin/env python3
# Copyright (c) 2025 Alibaba PAI Team.
# Copyright (c) 2025 Alpha Project Team.
#
# Alpha Config Generator
# ======================
# 단일 YAML 설정에서 학습/변환/HF 설정을 자동 생성하는 도구
#
# 사용법:
#   python alpha_config.py validate baseline_48L
#   python alpha_config.py generate-train-args baseline_48L
#   python alpha_config.py generate-convert-args baseline_48L
#   python alpha_config.py generate-hf-config baseline_48L --output /path/to/config.json

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# ============================================================================
# Constants
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
CONFIGS_DIR = SCRIPT_DIR.parent / "configs" / "model"
CONVERT_SCRIPTS_DIR = (
    SCRIPT_DIR.parent.parent.parent
    / "toolkits"
    / "distributed_checkpoints_convertor"
    / "scripts"
    / "alpha"
    / "configs"
)

# Pattern characters
CHAR_MAMBA = "M"  # GatedDeltaNet (Mamba-like) layer
CHAR_ATTENTION = "*"  # Full attention layer
CHAR_MLP = "-"  # MoE MLP layer
CHAR_DENSE_MLP = "D"  # Dense MLP layer (standard FFN)

# Default token IDs (alpha v5 tokenizer post-2026-05 pre-flight)
# Alpha has no BOS — the tokenizer's bos_token is null in tokenizer_config.json.
# EOS/EOD = <|endoftext|> (id 0), separated from chat-turn-end <|im_end|> (id 3)
# per frontier convention (Qwen3 / Llama 3 / DSV3 all separate these).
# Note: BOS_TOKEN_ID = None is preferred but kept as 0 here only as a structural
# default for HF config emission; downstream tools should explicitly omit
# bos_token_id from the HF config when BOS is None.
DEFAULT_BOS_TOKEN_ID = None
DEFAULT_EOS_TOKEN_ID = 0


# ============================================================================
# Config Schema
# ============================================================================


@dataclass
class TokenConfig:
    """Token ID configuration for tokenizer."""

    bos_token_id: Optional[int] = DEFAULT_BOS_TOKEN_ID  # alpha has no BOS (None)
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID            # <|endoftext|> id 0
    pad_token_id: Optional[int] = 1                     # <|pad|> id 1


@dataclass
class HybridConfig:
    """Hybrid model (Mamba + Attention) configuration."""

    attention_ratio: float = 0.125  # Fraction of attention layers
    mlp_ratio: float = 0.5  # Fraction of MLP-only layers
    override_pattern: Optional[str] = None  # Explicit pattern (optional)
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    mamba_num_groups: int = 16
    mamba_num_heads: int = 32


@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""

    num_experts: int = 256
    moe_ffn_hidden_size: int = 768
    router_topk: int = 8
    router_load_balancing_type: str = "aux_loss"
    aux_loss_coeff: float = 0.001
    router_score_function: str = "sigmoid"  # alpha baseline is DSV3 sigmoid routing
    router_dtype: str = "fp32"
    grouped_gemm: bool = True
    permute_fusion: bool = True
    router_fusion: bool = True
    shared_expert_intermediate_size: int = 768
    # DSV3 group-limited routing + aux-loss-free bias (Optional → emitted only
    # when the source config actually carries them; checkpoint always does).
    shared_expert_gate: bool = False
    token_dispatcher_type: str = "alltoall"
    router_num_groups: Optional[int] = None
    router_group_topk: Optional[int] = None
    router_topk_scaling_factor: Optional[float] = None
    router_enable_expert_bias: bool = False
    router_bias_update_rate: Optional[float] = None


@dataclass
class ModelConfig:
    """Complete Alpha model configuration."""

    name: str = "alpha-baseline"
    architecture: str = "qwen3_next_mamba_hybrid"

    # Core architecture
    num_layers: int = 24  # Megatron layers
    hidden_size: int = 2048
    ffn_hidden_size: int = 5120

    # Attention
    num_attention_heads: int = 32
    kv_channels: int = 128
    group_query_attention: bool = True
    num_query_groups: int = 2

    # Hybrid & MoE
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)

    # Token IDs
    tokens: TokenConfig = field(default_factory=TokenConfig)

    # Normalization
    normalization: str = "RMSNorm"
    norm_epsilon: float = 1e-6
    qk_layernorm: bool = True
    apply_layernorm_1p: bool = True

    # Activation & Dropout
    activation: str = "swiglu"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    disable_bias_linear: bool = True

    # Positional Encoding (RoPE)
    position_embedding_type: str = "rope"
    use_rotary_position_embeddings: bool = True
    rotary_base: int = 10000000
    rotary_percent: float = 0.25

    # Tokenizer & Vocab (alpha v5: 163,860 effective tokens padded to 128-multiple)
    untie_embeddings_and_output_weights: bool = True
    padded_vocab_size: int = 163968
    tokenizer_type: str = "HuggingFaceTokenizer"
    tokenizer_path: str = ""

    # Sequence + parallelism (needed to build the Megatron skeleton for
    # conversion/validation; parallelism here is the *training* topology — the
    # convert/validate launchers override EP from the runtime GPU count).
    seq_length: int = 4096
    max_position_embeddings: int = 262144
    expert_model_parallel_size: int = 1

    # Cached pattern (computed lazily)
    _pattern_cache: Optional[str] = field(default=None, repr=False)

    def get_pattern(self) -> str:
        """Get the hybrid override pattern (cached)."""
        if self._pattern_cache is None:
            if self.hybrid.override_pattern:
                self._pattern_cache = self.hybrid.override_pattern
            else:
                self._pattern_cache = generate_pattern(
                    self.num_layers, self.hybrid.attention_ratio, self.hybrid.mlp_ratio
                )
        return self._pattern_cache


# ============================================================================
# Pattern Generation
# ============================================================================


def generate_pattern(
    num_layers: int,
    attention_ratio: float,
    mlp_ratio: float,
) -> str:
    """
    Generate hybrid override pattern from ratios.

    Args:
        num_layers: Number of Megatron layers
        attention_ratio: Fraction of attention layers (e.g., 0.125 = 12.5%)
        mlp_ratio: Fraction of MLP-only layers (e.g., 0.5 = 50%)

    Returns:
        Pattern string like "M-M-M-*-M-M-M-*-M-M-M-*-"

    Example:
        >>> generate_pattern(24, 0.125, 0.5)
        'M-M-M-*-M-M-M-*-M-M-M-*-'
    """
    # Calculate layer counts
    num_attention = max(1, round(num_layers * attention_ratio))
    num_mlp = round(num_layers * mlp_ratio)
    num_mamba = num_layers - num_attention - num_mlp

    if num_mamba < 0:
        raise ValueError(
            f"Invalid ratios: attention_ratio={attention_ratio}, mlp_ratio={mlp_ratio} "
            f"results in negative Mamba layers ({num_mamba}) for {num_layers} layers"
        )

    # Calculate group size (layers between attention layers)
    # Pattern: M-M-M-* repeats
    group_size = num_layers // num_attention if num_attention > 0 else num_layers

    pattern = []
    attention_positions = set()

    # Place attention layers evenly
    for i in range(num_attention):
        pos = (i + 1) * group_size - 1
        if pos < num_layers:
            attention_positions.add(pos)

    # Generate pattern
    mlp_counter = 0
    mlp_interval = num_layers // num_mlp if num_mlp > 0 else num_layers + 1

    for i in range(num_layers):
        if i in attention_positions:
            pattern.append(CHAR_ATTENTION)
        elif mlp_counter < num_mlp and (i + 1) % 2 == 0 and i not in attention_positions:
            # MLP on even positions (except attention positions)
            pattern.append(CHAR_MLP)
            mlp_counter += 1
        else:
            pattern.append(CHAR_MAMBA)

    # Ensure exact MLP count by adjusting
    current_mlp = pattern.count(CHAR_MLP)
    if current_mlp < num_mlp:
        # Add more MLP layers
        for i in range(num_layers):
            if pattern[i] == CHAR_MAMBA and current_mlp < num_mlp:
                pattern[i] = CHAR_MLP
                current_mlp += 1
    elif current_mlp > num_mlp:
        # Remove excess MLP layers
        for i in range(num_layers - 1, -1, -1):
            if pattern[i] == CHAR_MLP and current_mlp > num_mlp:
                pattern[i] = CHAR_MAMBA
                current_mlp -= 1

    return "".join(pattern)


def parse_pattern(pattern: str) -> Tuple[int, int, int, int]:
    """
    Parse pattern string to get layer counts.

    Returns:
        (num_mamba, num_attention, num_mlp, num_dense_mlp)
        Note: num_mlp is MoE MLP count, num_dense_mlp is Dense MLP count
    """
    num_mamba = pattern.count(CHAR_MAMBA)
    num_attention = pattern.count(CHAR_ATTENTION)
    num_mlp = pattern.count(CHAR_MLP)
    num_dense_mlp = pattern.count(CHAR_DENSE_MLP)
    return num_mamba, num_attention, num_mlp, num_dense_mlp


# ============================================================================
# Config Loading
# ============================================================================


def _flat_to_nested(flat: Dict) -> Dict:
    """Re-shape a flat YAML (Megatron-flag-named keys) into the legacy nested
    `{model: {moe: {...}, hybrid: {...}, ...}}` schema this loader was written
    around. The flat schema is canonical (it's what train.sh expands directly
    into argv); this adapter exists so the inspection/HF-config-export tooling
    in alpha_config.py keeps working without a deep rewrite.
    """
    # MoE flags: drop the `moe-` / `moe-router-` prefix
    moe_renames = {
        'num-experts': 'num_experts',
        'moe-ffn-hidden-size': 'moe_ffn_hidden_size',
        'moe-router-topk': 'router_topk',
        'moe-router-load-balancing-type': 'router_load_balancing_type',
        'moe-aux-loss-coeff': 'aux_loss_coeff',
        'moe-router-score-function': 'router_score_function',
        'moe-router-dtype': 'router_dtype',
        'moe-grouped-gemm': 'grouped_gemm',
        'moe-permute-fusion': 'permute_fusion',
        'moe-router-fusion': 'router_fusion',
        'moe-shared-expert-intermediate-size': 'shared_expert_intermediate_size',
        'moe-shared-expert-gate': 'shared_expert_gate',
        'moe-token-dispatcher-type': 'token_dispatcher_type',
        'moe-router-num-groups': 'router_num_groups',
        'moe-router-group-topk': 'router_group_topk',
        'moe-router-topk-scaling-factor': 'router_topk_scaling_factor',
        'moe-router-enable-expert-bias': 'router_enable_expert_bias',
        'moe-router-bias-update-rate': 'router_bias_update_rate',
    }
    # Hybrid flags
    hybrid_renames = {
        'hybrid-attention-ratio': 'attention_ratio',
        'hybrid-mlp-ratio': 'mlp_ratio',
        'hybrid-override-pattern': 'override_pattern',
        'mamba-state-dim': 'mamba_state_dim',
        'mamba-head-dim': 'mamba_head_dim',
        'mamba-num-groups': 'mamba_num_groups',
        'mamba-num-heads': 'mamba_num_heads',
    }

    moe = {tgt: flat[src] for src, tgt in moe_renames.items() if src in flat}
    hybrid = {tgt: flat[src] for src, tgt in hybrid_renames.items() if src in flat}
    consumed = set(moe_renames) | set(hybrid_renames)

    # Everything else stays top-level under model. Convert dashed keys to
    # underscored to match the dataclass field names.
    model = {'moe': moe, 'hybrid': hybrid}
    for key, value in flat.items():
        if key in consumed:
            continue
        model[key.replace('-', '_')] = value

    return {'model': model}


def load_config(config_name: str) -> ModelConfig:
    """Load model config from YAML file (flat or legacy nested schema)."""
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    # Flat schema: top-level keys are Megatron CLI flag names (`num-layers: 48`)
    # — no `model:` wrapper. Detect by absence of `model` and presence of any
    # known flat-only key.
    if 'model' not in data and ('num-layers' in data or 'hidden-size' in data):
        data = _flat_to_nested(data)

    model_data = data.get("model", data)

    # Parse hybrid config
    hybrid_data = model_data.get("hybrid", {})
    hybrid = HybridConfig(
        attention_ratio=hybrid_data.get("attention_ratio", 0.125),
        mlp_ratio=hybrid_data.get("mlp_ratio", 0.5),
        override_pattern=hybrid_data.get("override_pattern"),
        mamba_state_dim=hybrid_data.get("mamba_state_dim", 128),
        mamba_head_dim=hybrid_data.get("mamba_head_dim", 64),
        mamba_num_groups=hybrid_data.get("mamba_num_groups", 16),
        mamba_num_heads=hybrid_data.get("mamba_num_heads", 32),
    )

    # Parse MoE config
    moe_data = model_data.get("moe", {})
    moe = MoEConfig(
        num_experts=moe_data.get("num_experts", 256),
        moe_ffn_hidden_size=moe_data.get("moe_ffn_hidden_size", 768),
        router_topk=moe_data.get("router_topk", 8),
        router_load_balancing_type=moe_data.get("router_load_balancing_type", "aux_loss"),
        aux_loss_coeff=moe_data.get("aux_loss_coeff", 0.001),
        router_score_function=moe_data.get("router_score_function", "sigmoid"),
        router_dtype=moe_data.get("router_dtype", "fp32"),
        grouped_gemm=moe_data.get("grouped_gemm", True),
        permute_fusion=moe_data.get("permute_fusion", True),
        router_fusion=moe_data.get("router_fusion", True),
        shared_expert_intermediate_size=moe_data.get("shared_expert_intermediate_size", 768),
        shared_expert_gate=moe_data.get("shared_expert_gate", False),
        token_dispatcher_type=moe_data.get("token_dispatcher_type", "alltoall"),
        router_num_groups=moe_data.get("router_num_groups"),
        router_group_topk=moe_data.get("router_group_topk"),
        router_topk_scaling_factor=moe_data.get("router_topk_scaling_factor"),
        router_enable_expert_bias=moe_data.get("router_enable_expert_bias", False),
        router_bias_update_rate=moe_data.get("router_bias_update_rate"),
    )

    # Parse token config
    tokens_data = model_data.get("tokens", {})
    tokens = TokenConfig(
        bos_token_id=tokens_data.get("bos_token_id", DEFAULT_BOS_TOKEN_ID),
        eos_token_id=tokens_data.get("eos_token_id", DEFAULT_EOS_TOKEN_ID),
        pad_token_id=tokens_data.get("pad_token_id"),
    )

    return ModelConfig(
        name=model_data.get("name", "alpha"),
        architecture=model_data.get("architecture", "qwen3_next_mamba_hybrid"),
        num_layers=model_data.get("num_layers", 24),
        hidden_size=model_data.get("hidden_size", 2048),
        ffn_hidden_size=model_data.get("ffn_hidden_size", 5120),
        num_attention_heads=model_data.get("num_attention_heads", 32),
        kv_channels=model_data.get("kv_channels", 128),
        group_query_attention=model_data.get("group_query_attention", True),
        num_query_groups=model_data.get("num_query_groups", 2),
        hybrid=hybrid,
        moe=moe,
        tokens=tokens,
        normalization=model_data.get("normalization", "RMSNorm"),
        norm_epsilon=model_data.get("norm_epsilon", 1e-6),
        qk_layernorm=model_data.get("qk_layernorm", True),
        apply_layernorm_1p=model_data.get("apply_layernorm_1p", True),
        activation=model_data.get("activation", "swiglu"),
        attention_dropout=model_data.get("attention_dropout", 0.0),
        hidden_dropout=model_data.get("hidden_dropout", 0.0),
        disable_bias_linear=model_data.get("disable_bias_linear", True),
        position_embedding_type=model_data.get("position_embedding_type", "rope"),
        use_rotary_position_embeddings=model_data.get("use_rotary_position_embeddings", True),
        rotary_base=model_data.get("rotary_base", 10000000),
        rotary_percent=model_data.get("rotary_percent", 0.25),
        untie_embeddings_and_output_weights=model_data.get(
            "untie_embeddings_and_output_weights", True
        ),
        padded_vocab_size=model_data.get("padded_vocab_size", 163968),
        tokenizer_type=model_data.get("tokenizer_type", "HuggingFaceTokenizer"),
        tokenizer_path=model_data.get("tokenizer_model", model_data.get("tokenizer_path", "")),
        seq_length=model_data.get("seq_length", 4096),
        max_position_embeddings=model_data.get("max_position_embeddings", 262144),
        expert_model_parallel_size=model_data.get("expert_model_parallel_size", 1),
    )


# ============================================================================
# Config Loading — from a Megatron checkpoint (ground truth)
# ============================================================================


def _find_common_pt(ckpt_dir: str) -> Path:
    """Resolve the `common.pt` (Megatron args namespace) for a checkpoint.

    Accepts any of:
      - an `iter_NNNNNNN/` directory (contains common.pt directly)
      - a run dir or its `checkpoints/` dir (resolves latest via
        `latest_checkpointed_iteration.txt`)
    """
    p = Path(ckpt_dir)
    direct = p / "common.pt"
    if direct.exists():
        return direct

    # Walk to the checkpoints/ dir if a run dir was passed.
    ckpt_root = p / "checkpoints" if (p / "checkpoints").is_dir() else p
    latest = ckpt_root / "latest_checkpointed_iteration.txt"
    if latest.exists():
        iteration = int(latest.read_text().strip())
        cand = ckpt_root / f"iter_{iteration:07d}" / "common.pt"
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"Could not locate common.pt under {ckpt_dir!r}. Pass an iter_NNNNNNN "
        f"directory, a checkpoints/ directory, or a run output directory."
    )


def load_config_from_checkpoint(ckpt_dir: str) -> ModelConfig:
    """Build a ModelConfig from a Megatron checkpoint's `common.pt` args.

    This is the *ground-truth* source: it reflects exactly what the run was
    launched with, immune to post-hoc edits of the model/training YAMLs. The
    returned ModelConfig is shape-identical to `load_config(yaml)`, so all
    downstream generators (generate_hf_config / emit_megatron_flags / ...) work
    unchanged. Requires Megatron on PYTHONPATH to unpickle the args namespace.
    """
    import os as _os

    import torch  # local import: only needed for the checkpoint path

    _os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "true")
    common_pt = _find_common_pt(ckpt_dir)
    ckpt = torch.load(common_pt, map_location="cpu", weights_only=False)
    if "args" not in ckpt:
        raise KeyError(f"'args' namespace not found in {common_pt}")
    a = vars(ckpt["args"])

    def g(key, default=None):
        return a.get(key, default)

    hybrid = HybridConfig(
        attention_ratio=g("hybrid_attention_ratio", 0.125),
        mlp_ratio=g("hybrid_mlp_ratio", 0.5),
        override_pattern=g("hybrid_override_pattern"),
        mamba_state_dim=g("mamba_state_dim", 128),
        mamba_head_dim=g("mamba_head_dim", 128),
        mamba_num_groups=g("mamba_num_groups", 16),
        mamba_num_heads=g("mamba_num_heads", 32),
    )

    moe = MoEConfig(
        num_experts=g("num_experts", 0) or 0,
        moe_ffn_hidden_size=g("moe_ffn_hidden_size", 0) or 0,
        router_topk=g("moe_router_topk", 8),
        router_load_balancing_type=g("moe_router_load_balancing_type", "aux_loss"),
        aux_loss_coeff=g("moe_aux_loss_coeff", 0.0) or 0.0,
        router_score_function=g("moe_router_score_function", "softmax"),
        router_dtype=g("moe_router_dtype") or "fp32",
        grouped_gemm=bool(g("moe_grouped_gemm", True)),
        shared_expert_intermediate_size=g("moe_shared_expert_intermediate_size", 0) or 0,
        shared_expert_gate=bool(g("moe_shared_expert_gate", False)),
        token_dispatcher_type=g("moe_token_dispatcher_type", "alltoall"),
        router_num_groups=g("moe_router_num_groups"),
        router_group_topk=g("moe_router_group_topk"),
        router_topk_scaling_factor=g("moe_router_topk_scaling_factor"),
        router_enable_expert_bias=bool(g("moe_router_enable_expert_bias", False)),
        router_bias_update_rate=g("moe_router_bias_update_rate"),
    )

    # Token IDs are an alpha-v5 designation (EOS/EOD=0, no BOS, PAD=1); they are
    # not stored in the Megatron args namespace. The defaults encode that and are
    # guarded by tests/test_alpha_tokenizer_eod.py against drift.
    tokens = TokenConfig()

    # Megatron stores `--disable-bias-linear` as `add_bias_linear=False`.
    disable_bias = not g("add_bias_linear", True)

    return ModelConfig(
        name="alpha-baseline",
        num_layers=g("num_layers"),
        hidden_size=g("hidden_size"),
        ffn_hidden_size=g("ffn_hidden_size"),
        num_attention_heads=g("num_attention_heads"),
        kv_channels=g("kv_channels"),
        group_query_attention=bool(g("group_query_attention", True)),
        num_query_groups=g("num_query_groups", 1),
        hybrid=hybrid,
        moe=moe,
        tokens=tokens,
        normalization=g("normalization", "RMSNorm"),
        norm_epsilon=g("norm_epsilon", 1e-6),
        qk_layernorm=bool(g("qk_layernorm", False)),
        apply_layernorm_1p=bool(g("apply_layernorm_1p", False)),
        activation="swiglu" if g("swiglu", False) else "gelu",
        attention_dropout=g("attention_dropout", 0.0),
        hidden_dropout=g("hidden_dropout", 0.0),
        disable_bias_linear=disable_bias,
        position_embedding_type=g("position_embedding_type", "rope"),
        use_rotary_position_embeddings=g("position_embedding_type", "rope") == "rope",
        rotary_base=g("rotary_base", 10000000),
        rotary_percent=g("rotary_percent", 1.0),
        untie_embeddings_and_output_weights=bool(g("untie_embeddings_and_output_weights", True)),
        padded_vocab_size=g("padded_vocab_size"),
        tokenizer_type=g("tokenizer_type", "HuggingFaceTokenizer"),
        tokenizer_path=g("tokenizer_model", "") or "",
        seq_length=g("seq_length", 4096),
        max_position_embeddings=g("max_position_embeddings", 262144),
        expert_model_parallel_size=g("expert_model_parallel_size", 1),
    )


# ============================================================================
# Validation
# ============================================================================


def validate_config(config: ModelConfig) -> List[str]:
    """
    Validate model configuration.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Get pattern (uses cached version)
    pattern = config.get_pattern()

    # Validate pattern length
    if len(pattern) != config.num_layers:
        errors.append(
            f"Pattern length ({len(pattern)}) != num_layers ({config.num_layers})"
        )

    # Validate pattern characters
    valid_chars = {CHAR_MAMBA, CHAR_ATTENTION, CHAR_MLP, CHAR_DENSE_MLP}
    invalid_chars = set(pattern) - valid_chars
    if invalid_chars:
        errors.append(f"Invalid characters in pattern: {invalid_chars}")

    # Validate attention ratio
    num_mamba, num_attention, num_mlp, num_dense_mlp = parse_pattern(pattern)
    actual_attention_ratio = num_attention / config.num_layers
    expected_attention_ratio = config.hybrid.attention_ratio

    if abs(actual_attention_ratio - expected_attention_ratio) > 0.05:
        errors.append(
            f"Attention ratio mismatch: pattern has {actual_attention_ratio:.3f}, "
            f"config says {expected_attention_ratio:.3f}"
        )

    # Validate MLP ratio (MoE + Dense MLP combined)
    total_mlp = num_mlp + num_dense_mlp
    actual_mlp_ratio = total_mlp / config.num_layers
    expected_mlp_ratio = config.hybrid.mlp_ratio

    if abs(actual_mlp_ratio - expected_mlp_ratio) > 0.05:
        errors.append(
            f"MLP ratio mismatch: pattern has {actual_mlp_ratio:.3f} "
            f"({num_mlp} MoE + {num_dense_mlp} Dense), config says {expected_mlp_ratio:.3f}"
        )

    # Validate HF layer count (2:1 mapping)
    num_hf_layers = config.num_layers // 2
    if config.num_layers % 2 != 0:
        errors.append(f"num_layers ({config.num_layers}) must be even for 2:1 HF mapping")

    # Validate expert count
    if config.moe.num_experts > 0 and config.moe.num_experts % 8 != 0:
        errors.append(f"num_experts ({config.moe.num_experts}) should be divisible by 8 for EP=8")

    return errors


# ============================================================================
# Generators
# ============================================================================


def generate_train_args(config: ModelConfig) -> List[str]:
    """Generate Megatron training arguments."""
    pattern = config.get_pattern()

    args = [
        # Basic Architecture
        f"--num-layers {config.num_layers}",
        f"--hidden-size {config.hidden_size}",
        f"--ffn-hidden-size {config.ffn_hidden_size}",
        f"--num-attention-heads {config.num_attention_heads}",
        f"--kv-channels {config.kv_channels}",
        f"--num-query-groups {config.num_query_groups}",
        "",
        "# Hybrid Model Pattern",
        f"--hybrid-attention-ratio {config.hybrid.attention_ratio}",
        f"--hybrid-mlp-ratio {config.hybrid.mlp_ratio}",
        f"--hybrid-override-pattern {pattern}",
        "--is-hybrid-model",
        "",
        "# MoE Configuration",
        f"--num-experts {config.moe.num_experts}",
        f"--moe-router-topk {config.moe.router_topk}",
        f"--moe-ffn-hidden-size {config.moe.moe_ffn_hidden_size}",
        f"--moe-shared-expert-intermediate-size {config.moe.shared_expert_intermediate_size}",
        "--moe-shared-expert-gate",
        "--moe-grouped-gemm" if config.moe.grouped_gemm else "",
        f"--moe-router-score-function {config.moe.router_score_function}",
        "--moe-token-dispatcher-type alltoall",
        "",
        "# RoPE Settings",
        f"--rotary-base {config.rotary_base}",
        f"--rotary-percent {config.rotary_percent}",
        "",
        "# Output Layer",
        "--untie-embeddings-and-output-weights" if config.untie_embeddings_and_output_weights else "",
    ]

    return [a for a in args if a]  # Remove empty strings


def generate_convert_args(config: ModelConfig) -> Dict[str, str]:
    """Generate MG2HF conversion arguments."""
    pattern = config.get_pattern()

    return {
        "num_layers": str(config.num_layers),
        "hidden_size": str(config.hidden_size),
        "ffn_hidden_size": str(config.ffn_hidden_size),
        "num_attention_heads": str(config.num_attention_heads),
        "kv_channels": str(config.kv_channels),
        "num_query_groups": str(config.num_query_groups),
        "hybrid_attention_ratio": str(config.hybrid.attention_ratio),
        "hybrid_mlp_ratio": str(config.hybrid.mlp_ratio),
        "hybrid_override_pattern": pattern,
        "num_experts": str(config.moe.num_experts),
        "moe_router_topk": str(config.moe.router_topk),
        "moe_ffn_hidden_size": str(config.moe.moe_ffn_hidden_size),
        "moe_shared_expert_intermediate_size": str(config.moe.shared_expert_intermediate_size),
        "rotary_base": str(config.rotary_base),
        "rotary_percent": str(config.rotary_percent),
        "vocab_size": str(config.padded_vocab_size),
    }


def emit_megatron_flags(config: ModelConfig) -> List[str]:
    """Emit the complete Megatron CLI flags that define the model *skeleton*,
    one shell token per element, for building the model to load a checkpoint
    (conversion) or to compare weights (validation).

    Deliberately omits parallelism (TP/PP/EP): the convert/validate launchers
    inject those from the runtime GPU count (EP=#GPU), independent of the
    training topology — torch_dist reshards experts on load. Returned as
    individual tokens so the caller can `readarray -t` without word-splitting
    or glob-expanding the `*` in the hybrid pattern.
    """
    pattern = config.get_pattern()
    moe = config.moe

    flags: List[str] = [
        # Core architecture
        "--num-layers", str(config.num_layers),
        "--hidden-size", str(config.hidden_size),
        "--ffn-hidden-size", str(config.ffn_hidden_size),
        "--num-attention-heads", str(config.num_attention_heads),
        "--kv-channels", str(config.kv_channels),
        "--num-query-groups", str(config.num_query_groups),
        # Hybrid (Mamba + Attention)
        "--is-hybrid-model",
        "--hybrid-attention-ratio", str(config.hybrid.attention_ratio),
        "--hybrid-mlp-ratio", str(config.hybrid.mlp_ratio),
        "--hybrid-override-pattern", pattern,
        "--mamba-state-dim", str(config.hybrid.mamba_state_dim),
        "--mamba-head-dim", str(config.hybrid.mamba_head_dim),
        "--mamba-num-groups", str(config.hybrid.mamba_num_groups),
        "--mamba-num-heads", str(config.hybrid.mamba_num_heads),
        # MoE
        "--num-experts", str(moe.num_experts),
        "--moe-router-topk", str(moe.router_topk),
        "--moe-ffn-hidden-size", str(moe.moe_ffn_hidden_size),
        "--moe-shared-expert-intermediate-size", str(moe.shared_expert_intermediate_size),
        "--moe-router-score-function", str(moe.router_score_function),
        "--moe-token-dispatcher-type", str(moe.token_dispatcher_type),
        # RoPE / norm / activation
        "--position-embedding-type", config.position_embedding_type,
        "--rotary-base", str(config.rotary_base),
        "--rotary-percent", str(config.rotary_percent),
        "--normalization", config.normalization,
        "--norm-epsilon", str(config.norm_epsilon),
        # Sequence + vocab
        "--seq-length", str(config.seq_length),
        "--max-position-embeddings", str(config.max_position_embeddings),
        "--padded-vocab-size", str(config.padded_vocab_size),
        # Tokenizer
        "--tokenizer-type", config.tokenizer_type,
    ]

    if config.group_query_attention:
        flags += ["--group-query-attention"]
    if config.moe.grouped_gemm:
        flags += ["--moe-grouped-gemm"]
    if config.moe.shared_expert_gate:
        flags += ["--moe-shared-expert-gate"]
    if config.qk_layernorm:
        flags += ["--qk-layernorm"]
    if config.activation == "swiglu":
        flags += ["--swiglu"]
    if config.disable_bias_linear:
        flags += ["--disable-bias-linear"]
    if config.untie_embeddings_and_output_weights:
        flags += ["--untie-embeddings-and-output-weights"]

    # DSV3 group-limited routing + aux-loss-free expert bias: emit only when the
    # source actually carried them (checkpoint always does; model-only YAML may not).
    if moe.router_num_groups is not None:
        flags += ["--moe-router-num-groups", str(moe.router_num_groups)]
    if moe.router_group_topk is not None:
        flags += ["--moe-router-group-topk", str(moe.router_group_topk)]
    if moe.router_topk_scaling_factor is not None:
        flags += ["--moe-router-topk-scaling-factor", str(moe.router_topk_scaling_factor)]
    if moe.router_enable_expert_bias:
        flags += ["--moe-router-enable-expert-bias"]
        if moe.router_bias_update_rate is not None:
            flags += ["--moe-router-bias-update-rate", str(moe.router_bias_update_rate)]

    if config.tokenizer_path:
        flags += ["--tokenizer-model", config.tokenizer_path]

    return flags


def generate_hf_config(config: ModelConfig) -> Dict:
    """Generate HuggingFace config.json content."""
    num_hf_layers = config.num_layers // 2

    # Calculate full_attention_interval from pattern
    pattern = config.get_pattern()

    # Find attention positions in MG layer space
    attention_positions = [i for i, c in enumerate(pattern) if c == CHAR_ATTENTION]
    num_attention = len(attention_positions)

    # full_attention_interval is in HF layer space
    # HF layers = MG layers / 2
    # If we have 3 attention layers in 12 HF layers, interval = 12 / 3 = 4
    if num_attention > 0:
        full_attention_interval = num_hf_layers // num_attention
    else:
        full_attention_interval = num_hf_layers

    # mlp_only_layers: HF layers that use Dense MLP instead of MoE
    # In Qwen3-Next HF model, layers in mlp_only_layers use Qwen3NextMLP (Dense)
    # instead of Qwen3NextSparseMoeBlock (MoE)
    # 'D' in MG pattern = Dense MLP layer
    # MG layer index → HF layer index: hf_idx = mg_idx // 2
    dense_mlp_mg_indices = [i for i, c in enumerate(pattern) if c == CHAR_DENSE_MLP]
    mlp_only_layers = sorted(set(idx // 2 for idx in dense_mlp_mg_indices))

    hf: Dict = {
        "architectures": ["AlphaForCausalLM"],
        "attention_dropout": config.attention_dropout,
        "auto_map": {
            "AutoConfig": "configuration_alpha.AlphaConfig",
            "AutoModelForCausalLM": "modeling_alpha.AlphaForCausalLM"
        },
        "bos_token_id": config.tokens.bos_token_id,
        "decoder_sparse_step": 1,
        "eos_token_id": config.tokens.eos_token_id,
        "full_attention_interval": full_attention_interval,
        "head_dim": config.kv_channels,
        "hidden_act": "silu",
        "hidden_size": config.hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": config.ffn_hidden_size,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": config.hybrid.mamba_head_dim,
        "linear_num_key_heads": config.hybrid.mamba_num_groups,
        "linear_num_value_heads": config.hybrid.mamba_num_heads,
        "linear_value_head_dim": config.hybrid.mamba_head_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "mlp_only_layers": mlp_only_layers,
        "model_type": "alpha",
        "moe_intermediate_size": config.moe.moe_ffn_hidden_size,
        "norm_topk_prob": True,
        # DSV3 routing (must match training; aux-loss-free bias is in the weights).
        "scoring_func": config.moe.router_score_function,
        "n_group": config.moe.router_num_groups if config.moe.router_num_groups is not None else 8,
        "topk_group": config.moe.router_group_topk if config.moe.router_group_topk is not None else 4,
        "routed_scaling_factor": (
            config.moe.router_topk_scaling_factor
            if config.moe.router_topk_scaling_factor is not None
            else 2.5
        ),
        "num_attention_heads": config.num_attention_heads,
        "num_experts": config.moe.num_experts,
        "num_experts_per_tok": config.moe.router_topk,
        "num_hidden_layers": num_hf_layers,
        "num_key_value_heads": config.num_query_groups,
        "output_router_logits": False,
        "partial_rotary_factor": config.rotary_percent,
        "rms_norm_eps": config.norm_epsilon,
        "rope_scaling": None,
        "rope_theta": config.rotary_base,
        "router_aux_loss_coef": config.moe.aux_loss_coeff,
        "shared_expert_intermediate_size": config.moe.shared_expert_intermediate_size,
        "tie_word_embeddings": not config.untie_embeddings_and_output_weights,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.57.0.dev0",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": config.padded_vocab_size,
    }

    # Alpha has no BOS (bos_token is null in tokenizer_config.json). Omit the
    # key entirely rather than emitting `null`, so HF/SGLang/vLLM never treat a
    # stale id as a sentinel (see alpha_config.py token-default notes).
    if hf["bos_token_id"] is None:
        del hf["bos_token_id"]

    return hf


def generate_convert_script(config: ModelConfig) -> str:
    """Generate bash conversion config script."""
    pattern = config.get_pattern()

    return f'''#!/bin/bash
# Copyright (c) 2025 Alibaba PAI Team.
# Copyright (c) 2025 Alpha Project Team.
#
# Alpha {config.name} Configuration
# =================================
# {config.num_layers} Megatron layers -> {config.num_layers // 2} HF layers (2:1 mapping)
# Auto-generated from: examples/alpha/configs/model/{config.name.replace("alpha-", "")}.yaml

GPT_MODEL_ARGS+=(
    # Basic Architecture
    --num-layers {config.num_layers}
    --hidden-size {config.hidden_size}
    --ffn-hidden-size {config.ffn_hidden_size}
    --num-attention-heads {config.num_attention_heads}
    --kv-channels {config.kv_channels}
    --num-query-groups {config.num_query_groups}

    # Hybrid Model Pattern
    # Pattern: {pattern}
    # Attention ratio: {config.hybrid.attention_ratio} ({int(config.num_layers * config.hybrid.attention_ratio)} layers)
    # MLP ratio: {config.hybrid.mlp_ratio} ({int(config.num_layers * config.hybrid.mlp_ratio)} layers)
    --hybrid-attention-ratio {config.hybrid.attention_ratio}
    --hybrid-mlp-ratio {config.hybrid.mlp_ratio}
    --hybrid-override-pattern {pattern}
    --is-hybrid-model

    # MoE Configuration
    --num-experts {config.moe.num_experts}
    --moe-router-topk {config.moe.router_topk}
    --moe-ffn-hidden-size {config.moe.moe_ffn_hidden_size}
    --moe-shared-expert-intermediate-size {config.moe.shared_expert_intermediate_size}
    --moe-shared-expert-gate
    --moe-grouped-gemm
    --moe-router-score-function {config.moe.router_score_function}
    --moe-token-dispatcher-type alltoall

    # RoPE Settings
    --rotary-base {config.rotary_base}
    --rotary-percent {config.rotary_percent}

    # Output Layer
    --untie-embeddings-and-output-weights
)

# Parallelization Strategy
# NOTE: TP=1 required (Mamba layer constraint)
if [ -z "$MODEL_PARALLEL_ARGS" ]; then
    MODEL_PARALLEL_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 1
        --expert-model-parallel-size 8      # {config.moe.num_experts} experts / 8 GPUs = {config.moe.num_experts // 8} per GPU
    )
fi

# Vocabulary
VOCAB_SIZE={config.padded_vocab_size}
'''


# ============================================================================
# CLI Commands
# ============================================================================


def _resolve_config(args) -> ModelConfig:
    """Resolve a ModelConfig from either a checkpoint (ground truth, preferred
    for MG→HF and validation) or a named YAML (HF→MG fallback / inspection)."""
    from_ckpt = getattr(args, "from_checkpoint", None)
    if from_ckpt:
        return load_config_from_checkpoint(from_ckpt)
    config_name = getattr(args, "config_name", None)
    if not config_name:
        raise SystemExit(
            "error: provide a config name (e.g. baseline_48L) or --from-checkpoint <dir>"
        )
    return load_config(config_name)


def cmd_emit_megatron_flags(args):
    """Emit Megatron model-skeleton flags (one token per line) for convert/validate."""
    config = _resolve_config(args)
    flags = emit_megatron_flags(config)
    text = "\n".join(flags)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"✅ Megatron flags written to {args.output}")
    else:
        print(text)
    return 0


def cmd_validate(args):
    """Validate configuration."""
    config = _resolve_config(args)
    label = getattr(args, "from_checkpoint", None) or getattr(args, "config_name", None) or "config"
    errors = validate_config(config)

    if errors:
        print(f"❌ Validation FAILED for {label}:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print(f"✅ Validation PASSED for {label}")

        # Print summary
        pattern = config.get_pattern()
        num_mamba, num_attention, num_mlp, num_dense_mlp = parse_pattern(pattern)

        print(f"\n📊 Model Summary:")
        print(f"  - Megatron layers: {config.num_layers}")
        print(f"  - HuggingFace layers: {config.num_layers // 2}")
        print(f"  - Pattern: {pattern}")
        print(f"  - Mamba layers: {num_mamba} ({num_mamba/config.num_layers*100:.1f}%)")
        print(f"  - Attention layers: {num_attention} ({num_attention/config.num_layers*100:.1f}%)")
        print(f"  - MoE MLP layers: {num_mlp} ({num_mlp/config.num_layers*100:.1f}%)")
        print(f"  - Dense MLP layers: {num_dense_mlp} ({num_dense_mlp/config.num_layers*100:.1f}%)")
        print(f"  - Experts: {config.moe.num_experts} (top-{config.moe.router_topk})")
        return 0


def cmd_generate_train_args(args):
    """Generate training arguments."""
    config = _resolve_config(args)
    train_args = generate_train_args(config)

    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(train_args))
        print(f"✅ Training args written to {args.output}")
    else:
        print("\n".join(train_args))
    return 0


def cmd_generate_convert_args(args):
    """Generate conversion arguments."""
    config = _resolve_config(args)
    convert_args = generate_convert_args(config)

    if args.output:
        with open(args.output, "w") as f:
            for key, value in convert_args.items():
                f.write(f"{key}={value}\n")
        print(f"✅ Convert args written to {args.output}")
    else:
        for key, value in convert_args.items():
            print(f"{key}={value}")
    return 0


def cmd_generate_hf_config(args):
    """Generate HuggingFace config.json."""
    config = _resolve_config(args)
    hf_config = generate_hf_config(config)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(hf_config, f, indent=2)
        print(f"✅ HF config written to {args.output}")
    else:
        print(json.dumps(hf_config, indent=2))
    return 0


def cmd_generate_convert_script(args):
    """Generate conversion bash script."""
    config = _resolve_config(args)
    script = generate_convert_script(config)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(script)
        os.chmod(output_path, 0o755)
        print(f"✅ Convert script written to {args.output}")
    else:
        print(script)
    return 0


def cmd_sync(args):
    """Sync all generated files from unified config."""
    config = load_config(args.config_name)
    errors = validate_config(config)

    if errors:
        print(f"❌ Cannot sync - validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    # Generate convert script
    script_path = CONVERT_SCRIPTS_DIR / f"{args.config_name}.sh"
    script = generate_convert_script(config)
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    print(f"✅ Synced: {script_path}")

    print(f"\n🎉 All files synced for {args.config_name}")
    return 0


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Alpha Config Generator - Generate train/convert/HF configs from unified YAML"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    def add_source_args(p, *, with_output=True):
        """Common source selectors: a named YAML or a checkpoint (ground truth)."""
        p.add_argument(
            "config_name",
            nargs="?",
            help="Config name (e.g., baseline_48L). Omit when using --from-checkpoint.",
        )
        p.add_argument(
            "--from-checkpoint",
            help="Derive config from a Megatron checkpoint's common.pt (iter dir, "
            "checkpoints/ dir, or run output dir). Ground truth; preferred for MG→HF.",
        )
        if with_output:
            p.add_argument("--output", "-o", help="Output file path")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate configuration")
    add_source_args(p_validate, with_output=False)
    p_validate.set_defaults(func=cmd_validate)

    # emit-megatron-flags (skeleton flags for convert/validate; one token/line)
    p_emit = subparsers.add_parser(
        "emit-megatron-flags",
        help="Emit Megatron model-skeleton flags (one token/line) for convert/validate",
    )
    add_source_args(p_emit)
    p_emit.set_defaults(func=cmd_emit_megatron_flags)

    # generate-train-args
    p_train = subparsers.add_parser("generate-train-args", help="Generate training arguments")
    add_source_args(p_train)
    p_train.set_defaults(func=cmd_generate_train_args)

    # generate-convert-args
    p_convert = subparsers.add_parser("generate-convert-args", help="Generate conversion arguments")
    add_source_args(p_convert)
    p_convert.set_defaults(func=cmd_generate_convert_args)

    # generate-hf-config
    p_hf = subparsers.add_parser("generate-hf-config", help="Generate HuggingFace config.json")
    add_source_args(p_hf)
    p_hf.set_defaults(func=cmd_generate_hf_config)

    # generate-convert-script
    p_script = subparsers.add_parser(
        "generate-convert-script", help="Generate conversion bash script"
    )
    add_source_args(p_script)
    p_script.set_defaults(func=cmd_generate_convert_script)

    # sync
    p_sync = subparsers.add_parser("sync", help="Sync all generated files from unified config")
    p_sync.add_argument("config_name", help="Config name (e.g., baseline_48L)")
    p_sync.set_defaults(func=cmd_sync)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
