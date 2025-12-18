# Copyright 2025 The HuggingFace Team. All rights reserved.
# Copyright 2025 Alpha Project Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Alpha Model - HuggingFace Compatible Implementation

Alpha is based on Qwen3-Next Mamba Hybrid architecture (GatedDeltaNet + Full Attention).
This module provides local version control and customization independence from the
transformers library.

Usage:
    from transformers import AutoModelForCausalLM, AutoConfig

    # Load with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        "path/to/alpha/hfmodel",
        trust_remote_code=True
    )
"""

from .configuration_alpha import AlphaConfig
from .modeling_alpha import (
    AlphaForCausalLM,
    AlphaModel,
    AlphaPreTrainedModel,
)

__all__ = [
    "AlphaConfig",
    "AlphaForCausalLM",
    "AlphaModel",
    "AlphaPreTrainedModel",
]
