# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
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
"""Pretrain GPT."""

import os
import torch
import inspect

from functools import partial
from megatron.core import mpu

from megatron.training import get_args, get_timers
# NOTE: megatron.training's tokenizer — alpha initializes the core tokenizer
# (megatron_patch.tokenizer.build_tokenizer is a separate registry that is NOT
# built on the alpha pretrain path and would raise NotImplementedError).
from megatron.training import get_tokenizer as get_core_tokenizer
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)

from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_thd_batch_on_this_cp_rank
from megatron_patch.data.utils import (
    get_batch_on_this_tp_rank_original,
    get_batch_on_this_tp_rank_idxmap_sft,
    get_position_id_on_this_tp_rank_idxmap_sft_packing,
    cu_seqlens_from_position_ids,
    merge_eod_pad_segments,
)

def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        packed_seq_params = None
        if args.dataset == 'MMAP' and args.train_mode == "finetune" and args.reset_position_ids:
            assert args.context_parallel_size == 1, (
                "packed SFT on a middle pipeline stage has no THD+CP wiring "
                "(cu_seqlens here is unmerged/global); use PP<=2 or CP=1"
            )
            position_ids = get_position_id_on_this_tp_rank_idxmap_sft_packing(data_iterator)
            position_ids = position_ids[0] # shape: [seq_length]
            # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
            cu_seqlens, max_seqlen = cu_seqlens_from_position_ids(position_ids)
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                qkv_format='thd',
                max_seqlen_q = max_seqlen,
                max_seqlen_kv = max_seqlen,
            )

        return None, None, None, None, None, None, packed_seq_params

    if args.dataset == 'JSON-SFT':
        if args.train_mode == "pretrain":
            raise ValueError('The JSON-SFT dataset should only be used for finetuning!')
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank_original(data_iterator, per_seq_average=True)
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs')
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            None
        )
    elif args.dataset == 'MMAP':
        # get batches based on the TP rank you are on
        if args.train_mode == "pretrain":
            batch = get_batch_on_this_tp_rank(data_iterator)
        else:
            batch = get_batch_on_this_tp_rank_idxmap_sft(data_iterator, per_seq_average=True)
        
        packed_seq_params = None
        cu_seqlens = None
        if args.reset_position_ids:
            # sequence-packing, build cu_seqlens
            position_ids = batch.get('position_ids', None)
            if position_ids is not None:
                assert position_ids.shape[0] == 1, (
                    "sequence packing (--reset-position-ids) derives cu_seqlens from "
                    "position_ids[0] and therefore requires micro-batch-size 1, got "
                    f"micro batch {position_ids.shape[0]}"
                )
                # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
                cu_seqlens, max_seqlen = cu_seqlens_from_position_ids(position_ids[0])
                # Per-document EOD pad runs (bestfit_pack --pad-doc-multiple) show
                # up as length-1 all-EOD segments because the position reset fires
                # after EVERY EOD. Absorb them into the preceding document so each
                # segment is "document + its pad run" — required for the THD+CP
                # per-segment % (2*cp) divisibility, and a no-op on pad-free data.
                cu_seqlens = merge_eod_pad_segments(
                    cu_seqlens, batch['tokens'][0], get_core_tokenizer().eod
                )
                seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
                max_seqlen = seg_lens.max().to(torch.int32)

        num_seqs = batch.pop('num_seqs', None)
        if cu_seqlens is not None and args.context_parallel_size > 1:
            # THD + CP: every tensor is partitioned PER SEGMENT (each CP rank
            # gets the load-balanced chunk pair (r, 2cp-1-r) of every packed
            # segment via TE's thd_get_partitioned_indices). The GDN mixer's
            # THD a2a permutation and TE's thd attention both assume exactly
            # this layout. Pads are inline EODs, so padded == actual cu_seqlens.
            batch, packed_seq_params = get_thd_batch_on_this_cp_rank(
                batch, cu_seqlens, cu_seqlens, max_seqlen.reshape(1)
            )
        else:
            if cu_seqlens is not None:
                packed_seq_params = PackedSeqParams(
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    qkv_format='thd',
                    max_seqlen_q=int(max_seqlen),
                    max_seqlen_kv=int(max_seqlen),
                )
            # slice batch along sequence dimension for context parallelism
            batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            packed_seq_params
        )
    else:
        raise ValueError("please set correct --dataset ")


def loss_func(loss_mask: torch.Tensor, num_seqs: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    # NOTE: for each seq, sum(loss_mask) == 1 if num_seqs is not None, 
    # otherwise sum(loss_mask) == n_tokens
    loss = torch.stack([torch.sum(losses.view(-1) * loss_mask), loss_mask.sum()])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]

    # NOTE: The grad will be scaled down by CP size later, should not remove this multilication factor
    # LINK: https://github.com/NVIDIA/Megatron-LM/issues/906
    # The issue is solved since 0926

    if num_seqs is None:
        # average on token-level
        return loss[0] / loss[1] * args.context_parallel_size, {"lm loss": averaged_loss}
    return loss[0] * args.context_parallel_size, num_seqs.sum(), {"lm loss": averaged_loss}

def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    args = get_args()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params = get_batch(data_iterator)
    timers("batch-generator").stop()

    input_kwargs = {
        'input_ids': tokens,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

    # Signature checks must run on the UNWRAPPED model: at train time `model` is
    # DDP(Float16Module(<real model>)) and wrapper forwards are generic
    # (*inputs, **kwargs), so inspecting the wrapper never finds these params
    # even when the real model supports them. Kwargs still flow through the
    # wrappers to the real forward, so only the check needs unwrapping.
    unwrapped = model
    while hasattr(unwrapped, 'module'):
        unwrapped = unwrapped.module
    forward_params = inspect.signature(unwrapped.forward).parameters

    if 'loss_mask' in forward_params:
        # NOTE: MTP-head (since 0328) requires loss_mask to compute correct loss scale.
        input_kwargs['loss_mask'] = loss_mask

    if 'packed_seq_params' in forward_params:
        input_kwargs['packed_seq_params'] = packed_seq_params
    else:
        assert packed_seq_params is None, f"Sequence Packing is not supported for {type(unwrapped).__name__}"

    output_tensor = model(**input_kwargs)

    return output_tensor, partial(loss_func, loss_mask, num_seqs)
