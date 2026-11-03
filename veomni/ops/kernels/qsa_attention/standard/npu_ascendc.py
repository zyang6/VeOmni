# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Qwen4-Exp QSA on AscendC GBSA / GBSAG (``npu_ascendc``).

The CANN call sequence matches MindSpeed-MM
``mindspeed_mm/fsdp/ops/qsa/ascendc/generic_sparse_attention.py``:

* forward: ``generic_block_sparse_attention`` (GBSA), TND, ``layout_sparse_pattern=4``,
  ``residual_block_mode=0``, causal ``mask_mode=1``. FP16 uses ``softmax_precision=1``;
  BF16 on 910B must use ``softmax_precision=0``.
* backward: ``generic_block_sparse_attention_grad`` (GBSAG). The forward pattern
  keeps the incomplete tail block; GBSAG drops that tail and sets
  ``residual_block_mode=1`` so the kernel adds it back.

VeOmni's indexer still emits compact token ids ``[B, S, K]``. This row turns
those tokens into unique block ids with ``block_size`` (the indexer
``compress_ratio``) and then calls the same adapter. Q and KV sequence lengths
must match, so Ulysses/CP is unsupported. Dropout and a KV cache are unsupported.
"""

from __future__ import annotations

import importlib
import logging
import math
from functools import lru_cache

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


logger = logging.getLogger(__name__)


def validate_prefill_mask(attention_mask, position_ids, batch, sequence):
    """Check sequence boundaries once per model call, before building 4D masks."""
    valid = None
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != (batch, sequence):
            raise NotImplementedError(
                "AscendC QSA accepts only a 2D padding mask or None; custom/packed masks are unsupported"
            )
        valid = attention_mask != 0
        invalid = ((attention_mask != 0) & (attention_mask != 1)).any()
        invalid = invalid | (valid[:, 1:] & ~valid[:, :-1]).any()
        if invalid.item():
            raise NotImplementedError(
                "AscendC QSA supports binary right-padding masks only; left padding/packing is unsupported"
            )
    if position_ids is not None:
        if position_ids.shape != (batch, sequence):
            raise ValueError("QSA text position_ids must have shape [B,S]")
        reset = position_ids[:, 1:] <= position_ids[:, :-1]
        if valid is not None:
            reset = reset & valid[:, 1:] & valid[:, :-1]
        if reset.any().item():
            raise NotImplementedError("AscendC QSA does not support packed sequences with reset text positions")


def _qsag_pattern(ids, counts, batch, sequence, block_size):
    """Convert forward [Hkv,B*S,K] to complete-block-only [B,Hkv,S,K]."""
    heads, _, width = ids.shape
    ids = ids.reshape(heads, batch, sequence, width).permute(1, 0, 2, 3)
    counts = counts.reshape(heads, batch, sequence).permute(1, 0, 2)
    slots = torch.arange(width, device=ids.device)
    complete = (torch.arange(sequence, device=ids.device) + 1) // block_size
    valid = (slots < counts[..., None]) & (ids >= 0) & (ids < complete[None, None, :, None])
    sentinel = (sequence + block_size - 1) // block_size
    indices = torch.where(valid, ids, sentinel).sort(dim=-1).values
    indices = torch.where(indices < sentinel, indices, 0).to(torch.int32).contiguous()
    return indices, valid.sum(-1).to(torch.int32).contiguous()


@lru_cache(maxsize=1)
def _load_qsag_backward():
    import cann_ops_transformer.ops as ops
    from cann_ops_transformer.ops import generic_block_sparse_attention_grad

    # Some CANN wheels have the QSAG-capable native ABI but retain a Python
    # validator restricted to layout=1/residual=0. Use that wheel's existing
    # binding without changing global registrations or files in site-packages.
    legacy_module = (
        "cann_ops_transformer.ops.attention.generic_block_sparse_attention_grad."
        "generic_block_sparse_attention_grad"
    )
    if isinstance(generic_block_sparse_attention_grad, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
        if "generic_block_sparse_attention_grad" in getattr(ops, "_ep_overridden_ops", ()):
            return generic_block_sparse_attention_grad, False
        module = importlib.import_module(legacy_module)
    else:
        module = importlib.import_module(generic_block_sparse_attention_grad.__module__)
    if module.__name__ == legacy_module and not hasattr(module, "_check_qsag"):
        native = module.generic_block_sparse_attention_grad_op_builder.load()
        logger.warning(
            "CANN grad Python wrapper predates QSAG; using its native "
            "GenericBlockSparseAttentionGrad binding (layout=0, residual=1). "
            "The loaded custom operator must support QSAG."
        )
        return native.generic_block_sparse_attention_grad, True
    return generic_block_sparse_attention_grad, False


def _qsag_backward(q, k, v, dout, out, lse, ids, counts, cu, block_size, scale):
    op, native_binding = _load_qsag_backward()
    if native_binding:
        # Positional ABI from CANN's csrc/attention/generic_block_sparse_attention_grad.cpp.
        # metadata/mask=None, INT64 cumulative lengths, no seqused, TND Q/KV,
        # layout=0, causal, precision=0, no window, implicit incomplete tail.
        return op(
            q, k, v, dout, out, lse, ids, counts,
            None, None, cu, cu, None, None, [1, block_size],
            "TND", "TND", 0, scale, 1, 0, -1, -1, 1, False,
        )
    return op(
        q, k, v, dout, out, lse, ids, counts, [1, block_size],
        metadata=None, attn_mask=None,
        cu_seqlens_q=cu, cu_seqlens_kv=cu, seqused_q=None, seqused_kv=None,
        layout_q="TND", layout_kv="TND", layout_sparse_pattern=0,
        softmax_scale=scale, mask_mode=1, softmax_precision=0,
        win_left=-1, win_right=-1, residual_block_mode=1,
        is_consistent_topk=False,
    )


class _GenericSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ids, counts, cu, metadata, block_size, scale, softmax_precision, needs_backward):
        from cann_ops_transformer.ops import generic_block_sparse_attention

        out, lse = generic_block_sparse_attention(
            q, k, v, ids, counts, [1, block_size], metadata=metadata,
            attn_mask=None, block_table=None, cu_seqlens_q=cu, cu_seqlens_kv=cu,
            layout_q="TND", layout_kv="TND", layout_sparse_pattern=4,
            residual_block_mode=0, is_consistent_topk=False,
            softmax_scale=scale, mask_mode=1, quant_mode=0, softmax_precision=softmax_precision,
            return_softmax_lse=needs_backward,
        )
        if needs_backward:
            ctx.save_for_backward(q, k, v, out, lse, ids, counts, cu)
            ctx.block_size = block_size
            ctx.scale = scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, out, lse, ids, counts, cu = ctx.saved_tensors
        batch = cu.numel() - 1
        sequence = q.shape[0] // batch
        grad_ids, grad_counts = _qsag_pattern(ids, counts, batch, sequence, ctx.block_size)
        dq, dk, dv = _qsag_backward(
            q, k, v, dout.contiguous(), out, lse,
            grad_ids, grad_counts, cu, ctx.block_size, ctx.scale,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def complete_block_ids(compact, sequence, token_budget, compress_ratio):
    """Restore dense-prefix rows omitted by the Lightning indexer."""
    if compress_ratio <= 0 or token_budget <= 0 or token_budget % compress_ratio:
        raise ValueError("QSA token_budget must be positive and divisible by compress_ratio")
    batch, sparse_rows, width = compact.shape
    prefix = min(sequence, token_budget + compress_ratio - 1)
    if sparse_rows != sequence - prefix or width != token_budget // compress_ratio + 1:
        raise ValueError("Compact QSA routing shape does not match sequence/budget/block size")
    rows = torch.arange(prefix, device=compact.device)
    ids = torch.arange(width, dtype=torch.int32, device=compact.device)
    dense = torch.where(ids[None, :] <= rows[:, None] // compress_ratio, ids[None, :], -1)
    return torch.cat((dense[None].expand(batch, -1, -1), compact), dim=1)


def generic_sparse_attention(query, key, value, block_ids, *, block_size, scale, dropout=0.0):
    """BNSD -> contiguous TND -> BSH; single-backward causal prefill only."""
    if dropout != 0.0:
        raise NotImplementedError("CANN QSA does not support attention dropout")
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("QSA requires Q/K/V in BNSD layout and matching K/V shapes")
    b, hq, s, d = query.shape
    hkv = key.shape[1]
    if min(b, hq, hkv, s, d) <= 0 or hq % hkv or hq // hkv > 256:
        raise ValueError("QSA requires positive dimensions, Hq divisible by Hkv and GQA<=256")
    if key.shape != (b, hkv, s, d) or not 1 <= d <= 256:
        raise ValueError("QSA requires equal Q/KV sequence lengths and head_dim<=256")
    max_block_size = 128 if hq // hkv <= 128 else 64
    if not isinstance(block_size, int) or not 1 <= block_size <= max_block_size or not math.isfinite(scale):
        raise ValueError("Unsupported QSA block size or softmax scale")
    if query.device.type != "npu" or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CANN QSA requires FP16/BF16 NPU tensors")
    if any(x.device != query.device or x.dtype != query.dtype for x in (key, value)):
        raise ValueError("QSA Q/K/V must have matching dtype and device")
    if (
        block_ids.ndim != 3
        or block_ids.shape[:2] != (b, s)
        or block_ids.shape[2] < 1
        or block_ids.dtype != torch.int32
        or block_ids.device != query.device
    ):
        raise ValueError("QSA block_ids must be INT32 [B,S,K] on the Q/K/V device")

    from cann_ops_transformer.ops import generic_block_sparse_attention_metadata

    needs_backward = torch.is_grad_enabled() and any(x.requires_grad for x in (query, key, value))
    # Kernel rejects softmax_precision=1 for BF16 on 910/910B. FP16 keeps the MindSpeed value 1.
    softmax_precision = 0 if query.dtype == torch.bfloat16 else 1
    limit = (s + block_size - 1) // block_size
    valid = (block_ids >= 0) & (block_ids < limit)
    ids = torch.where(valid, block_ids, limit).sort(dim=-1).values
    count = valid.sum(-1).to(torch.int32)
    ids = torch.where(ids < limit, ids, 0).to(torch.int32)
    ids = ids.reshape(1, b * s, -1).expand(hkv, -1, -1).contiguous()
    count = count.reshape(1, b * s).expand(hkv, -1).contiguous()
    cu = torch.arange(b + 1, dtype=torch.int64, device=query.device) * s
    metadata = generic_block_sparse_attention_metadata(
        ids, count, hq, hkv, d, [1, block_size],
        cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=s, max_seqlen_kv=s,
        layout_q="TND", layout_kv="TND", layout_sparse_pattern=4,
        residual_block_mode=0, is_consistent_topk=False,
        mask_mode=1, quant_mode=0, softmax_precision=softmax_precision,
    )
    q, k, v = [x.transpose(1, 2).reshape(b * s, x.shape[1], d).contiguous() for x in (query, key, value)]
    out = _GenericSparseAttention.apply(
        q, k, v, ids, count, cu, metadata, block_size, scale, softmax_precision, needs_backward,
    )
    return out.reshape(b, s, hq * d)


def token_indices_to_block_ids(indices: Tensor, block_size: int, seq_len: int) -> Tensor:
    """Collapse compact token ids ``[B, S, K]`` into unique causal block ids.

    ``-1`` slots stay padding. Tokens that share a block become one id. The
    result is INT32 ``[B, S, W]`` with valid ids packed at the front and
    ``W = min(K, ceil(S / block_size))``.
    """
    if indices.ndim != 3:
        raise ValueError(f"QSA selected indices must be rank-3 [B,S,K], got {tuple(indices.shape)}")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"QSA block_size must be a positive int, got {block_size!r}")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"QSA selected indices must be int32 or int64, got {indices.dtype}")
    if indices.shape[-1] == 0:
        return torch.full((*indices.shape[:2], 1), -1, dtype=torch.int32, device=indices.device)
    limit = (seq_len + block_size - 1) // block_size
    valid = indices >= 0
    blocks = torch.div(indices, block_size, rounding_mode="floor").to(torch.int32)
    sentinel = torch.iinfo(torch.int32).max
    tagged = torch.where(valid, blocks, torch.full((), sentinel, dtype=torch.int32, device=indices.device))
    ordered = tagged.sort(dim=-1).values
    previous = torch.cat((torch.full_like(ordered[..., :1], -1), ordered[..., :-1]), dim=-1)
    unique = (ordered != previous) & (ordered != sentinel) & (ordered < limit)
    deduped = torch.where(unique, ordered, torch.full((), sentinel, dtype=torch.int32, device=indices.device))
    packed = deduped.sort(dim=-1).values
    width = min(packed.shape[-1], max(limit, 1))
    packed = packed[..., :width]
    return torch.where(
        packed == sentinel,
        torch.full((), -1, dtype=torch.int32, device=indices.device),
        packed,
    ).contiguous()


def wrapper(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selected_indices: Tensor,
    *,
    scaling: float,
    dropout: float = 0.0,
    attention_mask: Tensor | None = None,
    block_size: int | None = None,
) -> tuple[Tensor, None]:
    """Run GBSA/GBSAG over the compact QSA token selection.

    *query*, *key*, and *value* are ``[B, H, S, D]``. *selected_indices* is
    ``[B, S, K]`` of global token ids (``-1`` padding), the same contract as
    ``eager`` / ``torch_npu``. *block_size* is the indexer compress ratio.
    The returned attention is ``[B, S, H, D]`` and weights are ``None``.
    """
    if block_size is None:
        raise ValueError(
            "npu_ascendc QSA requires block_size (Qwen4ExpTextQSAIndexer.compress_ratio); "
            "the compact selection stores token ids, and GBSA consumes block ids."
        )
    if query.ndim != 4:
        raise ValueError(f"npu_ascendc QSA query must be [B,H,S,D], got {tuple(query.shape)}")
    if attention_mask is not None:
        validate_prefill_mask(attention_mask, None, query.shape[0], query.shape[2])
    if selected_indices.shape[:2] != (query.shape[0], query.shape[2]):
        raise ValueError(
            "npu_ascendc QSA selection must be [B,S,K] aligned with query; "
            f"got query={tuple(query.shape)}, selected_indices={tuple(selected_indices.shape)}. "
            "Ulysses/CP (local Q, gathered KV) is unsupported."
        )
    block_ids = token_indices_to_block_ids(selected_indices, int(block_size), query.shape[2])
    flat = generic_sparse_attention(
        query, key, value, block_ids, block_size=int(block_size), scale=scaling, dropout=dropout,
    )
    batch, heads, seq_len, dim = query.shape
    return flat.view(batch, seq_len, heads, dim), None
