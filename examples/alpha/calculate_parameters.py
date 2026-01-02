#!/usr/bin/env python3
# Copyright (c) 2025 Alpha Project Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Alpha Model Parameter Calculator
=================================
MoE 모델의 전체 파라미터와 활성화 파라미터를 계산합니다.

Usage:
    python calculate_parameters.py --config configs/model/baseline_48L.yaml
    python calculate_parameters.py --config configs/model/baseline_48L.yaml --detailed
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Tuple


class AlphaParameterCalculator:
    """Alpha MoE 모델 파라미터 계산기"""

    def __init__(self, config: Dict):
        """
        Args:
            config: YAML 설정 딕셔너리
        """
        self.config = config['model']
        self.moe = config['model']['moe']
        self.hybrid = config['model']['hybrid']

        # 기본 파라미터
        self.num_layers = self.config['num_layers']
        self.hidden_size = self.config['hidden_size']
        self.ffn_hidden_size = self.config['ffn_hidden_size']
        self.vocab_size = self.config['padded_vocab_size']

        # Attention 파라미터
        self.num_attention_heads = self.config['num_attention_heads']
        self.kv_channels = self.config['kv_channels']
        self.num_query_groups = self.config['num_query_groups']

        # MoE 파라미터
        self.num_experts = self.moe['num_experts']
        self.moe_ffn_hidden_size = self.moe['moe_ffn_hidden_size']
        self.router_topk = self.moe['router_topk']
        self.shared_expert_size = self.moe['shared_expert_intermediate_size']

        # Hybrid 파라미터
        self.attention_ratio = self.hybrid['attention_ratio']
        self.mlp_ratio = self.hybrid['mlp_ratio']
        self.mamba_state_dim = self.hybrid['mamba_state_dim']
        self.mamba_head_dim = self.hybrid['mamba_head_dim']
        self.mamba_num_groups = self.hybrid['mamba_num_groups']
        self.mamba_num_heads = self.hybrid['mamba_num_heads']

        # 패턴 파싱
        self.pattern = self.hybrid['override_pattern']
        self.parse_pattern()

    def parse_pattern(self):
        """하이브리드 패턴 파싱

        패턴 토큰:
        - M: Mamba 레이어 (Linear Attention SSM)
        - *: Full Attention 레이어 (Multi-Head Attention)
        - -: MoE MLP 레이어 (Mixture of Experts)
        - D: Dense MLP 레이어 (Standard FFN with SwiGLU)
        """
        # "MDM-M-*-M-M-M-*-..." 에서 모든 토큰 추출
        # 각 문자가 하나의 레이어를 의미
        tokens = [c for c in self.pattern if c in 'M*-D']

        self.num_mamba_layers = tokens.count('M')
        self.num_attention_layers = tokens.count('*')
        self.num_moe_mlp_layers = tokens.count('-')  # MoE MLP
        self.num_dense_mlp_layers = tokens.count('D')  # Dense MLP
        self.num_mlp_layers = self.num_moe_mlp_layers + self.num_dense_mlp_layers  # 총 MLP

        total_tokens = len(tokens)
        assert total_tokens == self.num_layers, \
            f"Pattern length ({total_tokens}) != num_layers ({self.num_layers})\n" \
            f"Pattern: {self.pattern}\n" \
            f"Tokens: {tokens}\n" \
            f"M: {self.num_mamba_layers}, *: {self.num_attention_layers}, -: {self.num_mlp_layers}"

    def calc_embedding_params(self) -> Tuple[int, int]:
        """임베딩 레이어 파라미터"""
        # Input embedding: vocab_size × hidden_size
        input_embed = self.vocab_size * self.hidden_size

        # Output embedding (unshared)
        if self.config['untie_embeddings_and_output_weights']:
            output_embed = self.vocab_size * self.hidden_size
        else:
            output_embed = 0

        total = input_embed + output_embed
        return total, total  # 임베딩은 항상 활성화

    def calc_mamba_layer_params(self) -> Tuple[int, int]:
        """단일 Mamba 레이어 파라미터 (Linear Attention SSM)"""
        h = self.hidden_size
        state_dim = self.mamba_state_dim
        head_dim = self.mamba_head_dim
        num_heads = self.mamba_num_heads
        num_groups = self.mamba_num_groups

        # Mamba SSM core parameters
        # Based on Qwen3-Next Mamba implementation

        # 1. Input projection: h -> (state_dim + 2 * num_groups * head_dim)
        in_proj_dim = state_dim + 2 * num_groups * head_dim
        in_proj = h * in_proj_dim

        # 2. Conv1D: state_dim channels, kernel_size=4
        conv1d = state_dim * 4

        # 3. SSM parameters (A, B, C, D, dt)
        # A: (num_heads, head_dim, state_dim)
        # B: (num_groups, state_dim)
        # C: (num_groups, state_dim)
        # D: (num_heads, head_dim)
        # dt: (num_heads, head_dim)
        ssm_a = num_heads * head_dim * state_dim
        ssm_b = num_groups * state_dim
        ssm_c = num_groups * state_dim
        ssm_d = num_heads * head_dim
        ssm_dt = num_heads * head_dim

        # 4. Output projection: num_heads * head_dim -> h
        out_proj = (num_heads * head_dim) * h

        # 5. Layer normalization
        norm = h * 2  # scale + bias (RMSNorm only scale)

        total = in_proj + conv1d + ssm_a + ssm_b + ssm_c + ssm_d + ssm_dt + out_proj + norm
        return total, total

    def calc_attention_layer_params(self) -> Tuple[int, int]:
        """단일 Attention 레이어 파라미터 (Multi-Head Attention)"""
        h = self.hidden_size
        num_heads = self.num_attention_heads
        kv_channels = self.kv_channels
        num_kv_heads = self.num_query_groups

        # 1. QKV projection
        # Q: h -> num_heads * kv_channels
        # K, V: h -> num_kv_heads * kv_channels (GQA)
        q_proj = h * (num_heads * kv_channels)
        k_proj = h * (num_kv_heads * kv_channels)
        v_proj = h * (num_kv_heads * kv_channels)

        # 2. Output projection
        out_proj = (num_heads * kv_channels) * h

        # 3. QK LayerNorm (if enabled)
        qk_norm = 0
        if self.config.get('qk_layernorm', False):
            qk_norm = 2 * kv_channels * 2  # Q, K each

        # 4. Layer normalization
        norm = h * 2

        total = q_proj + k_proj + v_proj + out_proj + qk_norm + norm
        return total, total

    def calc_mlp_layer_params(self) -> Tuple[int, int]:
        """단일 MLP 레이어 파라미터 (Feed-Forward)"""
        h = self.hidden_size
        ffn = self.ffn_hidden_size

        # SwiGLU: W_gate, W_up, W_down
        gate = h * ffn
        up = h * ffn
        down = ffn * h

        # Layer normalization
        norm = h * 2

        total = gate + up + down + norm
        return total, total

    def calc_moe_layer_params(self) -> Tuple[int, int]:
        """MoE 레이어 파라미터 (Expert 선택 포함)"""
        h = self.hidden_size
        expert_ffn = self.moe_ffn_hidden_size
        num_experts = self.num_experts
        topk = self.router_topk

        # 1. Router: h -> num_experts
        router = h * num_experts

        # 2. Expert FFN (SwiGLU per expert)
        expert_gate = h * expert_ffn
        expert_up = h * expert_ffn
        expert_down = expert_ffn * h
        expert_params = (expert_gate + expert_up + expert_down) * num_experts

        # 3. Shared Expert (always active)
        shared_ffn = self.shared_expert_size
        shared_gate = h * shared_ffn
        shared_up = h * shared_ffn
        shared_down = shared_ffn * h
        shared_params = shared_gate + shared_up + shared_down

        # 4. Layer normalization
        norm = h * 2

        # 전체 파라미터
        total_params = router + expert_params + shared_params + norm

        # 활성화 파라미터 (Top-K experts + Shared expert)
        active_expert_params = (expert_gate + expert_up + expert_down) * topk
        active_params = router + active_expert_params + shared_params + norm

        return total_params, active_params

    def calculate(self) -> Dict[str, int]:
        """전체 파라미터 계산

        패턴 구조:
        - M: Mamba 레이어 (GatedDeltaNet)
        - *: Full Attention 레이어
        - D: Dense MLP 레이어 (Standard SwiGLU FFN)
        - -: MoE MLP 레이어 (Mixture of Experts)

        예시 패턴: "MDM-M-*-M-M-M-*-M-M-M-*-"
        - M × 9: Mamba 레이어
        - D × 1: Dense MLP (첫 번째 MLP만)
        - * × 3: Full Attention
        - - × 11: MoE MLP
        """
        results = {}

        # 1. Embedding
        embed_total, embed_active = self.calc_embedding_params()
        results['embedding_total'] = embed_total
        results['embedding_active'] = embed_active

        # 2. Mamba layers (M)
        mamba_total, mamba_active = self.calc_mamba_layer_params()
        results['mamba_per_layer'] = mamba_total
        results['mamba_total'] = mamba_total * self.num_mamba_layers
        results['mamba_active'] = mamba_active * self.num_mamba_layers

        # 3. Attention layers (*)
        attn_total, attn_active = self.calc_attention_layer_params()
        results['attention_per_layer'] = attn_total
        results['attention_total'] = attn_total * self.num_attention_layers
        results['attention_active'] = attn_active * self.num_attention_layers

        # 4. Dense MLP layers (D) - Standard SwiGLU FFN
        dense_mlp_total, dense_mlp_active = self.calc_mlp_layer_params()
        results['dense_mlp_per_layer'] = dense_mlp_total
        results['dense_mlp_total'] = dense_mlp_total * self.num_dense_mlp_layers
        results['dense_mlp_active'] = dense_mlp_active * self.num_dense_mlp_layers

        # 5. MoE MLP layers (-) - Mixture of Experts
        moe_total, moe_active = self.calc_moe_layer_params()
        results['moe_per_layer'] = moe_total
        results['moe_total'] = moe_total * self.num_moe_mlp_layers
        results['moe_active'] = moe_active * self.num_moe_mlp_layers

        # 6. 총합
        results['total_params'] = (
            results['embedding_total'] +
            results['mamba_total'] +
            results['attention_total'] +
            results['dense_mlp_total'] +
            results['moe_total']
        )

        results['active_params'] = (
            results['embedding_active'] +
            results['mamba_active'] +
            results['attention_active'] +
            results['dense_mlp_active'] +
            results['moe_active']
        )

        # 7. 메타 정보
        results['num_layers'] = self.num_layers
        results['num_mamba_layers'] = self.num_mamba_layers
        results['num_attention_layers'] = self.num_attention_layers
        results['num_dense_mlp_layers'] = self.num_dense_mlp_layers
        results['num_moe_mlp_layers'] = self.num_moe_mlp_layers
        results['num_experts'] = self.num_experts
        results['router_topk'] = self.router_topk

        return results

    def format_number(self, num: int) -> str:
        """숫자를 읽기 쉽게 포맷 (B, M, K)"""
        if num >= 1e9:
            return f"{num / 1e9:.2f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K"
        else:
            return str(num)

    def print_summary(self, results: Dict[str, int], detailed: bool = False):
        """결과 출력"""
        print("\n" + "="*80)
        print(f"Alpha Model Parameter Summary: {self.config['name']}")
        print("="*80)

        print(f"\nModel Configuration:")
        print(f"  Total Layers: {results['num_layers']}")
        print(f"    - Mamba Layers (M): {results['num_mamba_layers']}")
        print(f"    - Attention Layers (*): {results['num_attention_layers']}")
        print(f"    - Dense MLP Layers (D): {results['num_dense_mlp_layers']}")
        print(f"    - MoE MLP Layers (-): {results['num_moe_mlp_layers']}")
        print(f"  Hidden Size: {self.hidden_size}")
        print(f"  Vocab Size: {self.vocab_size:,}")

        print(f"\nMoE Configuration:")
        print(f"  Total Experts: {results['num_experts']}")
        print(f"  Active Experts (Top-K): {results['router_topk']}")
        print(f"  Expert Utilization: {results['router_topk']/results['num_experts']*100:.1f}%")

        print(f"\n" + "-"*80)
        print(f"Parameter Counts:")
        print(f"-"*80)

        # 주요 결과
        total_params = results['total_params']
        active_params = results['active_params']

        print(f"\n{'Total Parameters:':<30} {self.format_number(total_params):>15} ({total_params:,})")
        print(f"{'Active Parameters:':<30} {self.format_number(active_params):>15} ({active_params:,})")
        print(f"{'Activation Ratio:':<30} {active_params/total_params*100:>14.1f}%")

        if detailed:
            print(f"\n" + "-"*80)
            print(f"Detailed Breakdown:")
            print(f"-"*80)

            # Embedding
            print(f"\n1. Embedding Layers:")
            print(f"   Total:  {self.format_number(results['embedding_total']):>15} ({results['embedding_total']:,})")
            print(f"   Active: {self.format_number(results['embedding_active']):>15} ({results['embedding_active']:,})")

            # Mamba
            print(f"\n2. Mamba Layers ({results['num_mamba_layers']} layers):")
            print(f"   Per Layer: {self.format_number(results['mamba_per_layer']):>12} ({results['mamba_per_layer']:,})")
            print(f"   Total:     {self.format_number(results['mamba_total']):>12} ({results['mamba_total']:,})")

            # Attention
            print(f"\n3. Attention Layers ({results['num_attention_layers']} layers):")
            print(f"   Per Layer: {self.format_number(results['attention_per_layer']):>12} ({results['attention_per_layer']:,})")
            print(f"   Total:     {self.format_number(results['attention_total']):>12} ({results['attention_total']:,})")

            # Dense MLP
            print(f"\n4. Dense MLP Layers ({results['num_dense_mlp_layers']} layers):")
            print(f"   Per Layer: {self.format_number(results['dense_mlp_per_layer']):>12} ({results['dense_mlp_per_layer']:,})")
            print(f"   Total:     {self.format_number(results['dense_mlp_total']):>12} ({results['dense_mlp_total']:,})")

            # MoE
            num_moe = results['num_moe_mlp_layers']
            print(f"\n5. MoE MLP Layers ({num_moe} layers):")
            print(f"   Per Layer:")
            print(f"     Total:  {self.format_number(results['moe_per_layer']):>15} ({results['moe_per_layer']:,})")
            if num_moe > 0:
                print(f"     Active: {self.format_number(results['moe_active'] // num_moe):>15}")
            print(f"   All Layers:")
            print(f"     Total:  {self.format_number(results['moe_total']):>15} ({results['moe_total']:,})")
            print(f"     Active: {self.format_number(results['moe_active']):>15} ({results['moe_active']:,})")

            # 비율
            print(f"\n" + "-"*80)
            print(f"Component Ratios (Total):")
            print(f"-"*80)
            print(f"  Embedding:  {results['embedding_total']/total_params*100:5.1f}%")
            print(f"  Mamba:      {results['mamba_total']/total_params*100:5.1f}%")
            print(f"  Attention:  {results['attention_total']/total_params*100:5.1f}%")
            print(f"  Dense MLP:  {results['dense_mlp_total']/total_params*100:5.1f}%")
            print(f"  MoE:        {results['moe_total']/total_params*100:5.1f}%")

        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Alpha Model Parameter Calculator")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model/baseline_48L.yaml',
        help='Path to model config YAML file'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed parameter breakdown'
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Relative to examples/alpha/
        config_path = Path(__file__).parent / config_path

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Calculate
    calculator = AlphaParameterCalculator(config)
    results = calculator.calculate()
    calculator.print_summary(results, detailed=args.detailed)

    # Save to file (optional)
    output_file = config_path.parent.parent / 'outputs' / f"{config['model']['name']}_params.txt"
    output_file.parent.mkdir(exist_ok=True, parents=True)

    import sys
    from io import StringIO

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = output_buffer = StringIO()
    calculator.print_summary(results, detailed=True)
    sys.stdout = old_stdout

    # Save
    with open(output_file, 'w') as f:
        f.write(output_buffer.getvalue())

    print(f"Detailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
