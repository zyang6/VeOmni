# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Crop and restore Qwen4-Exp PLE rows around a DCP save.

Runtime parameters stay padded so ``ple`` can split them evenly. A DCP
checkpoint stores only the logical rows. ``2_500_012`` is not divisible by
the ``ple`` mesh, so the saved tensor is redistributed one shard at a time
onto the logical shape. Loading pads the zero tail back onto the parameter.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from veomni.models.checkpoint.convert import get_checkpoint_tensor_converter
from veomni.models.transformers.qwen4_exp.checkpoint_tensor_converter import _PLE_SHARD_PATTERN
from veomni.utils import logging


logger = logging.get_logger(__name__)


def ple_row_counts(model: nn.Module) -> Optional[Tuple[int, int]]:
    """Return ``(logical_rows, padded_rows)`` for a Qwen4-Exp PLE model."""
    converter = get_checkpoint_tensor_converter(model) if model is not None else None
    logical = getattr(converter, "logical_rows", None)
    if not isinstance(logical, int):
        return None

    padded = None
    for module in model.modules():
        rows = getattr(module, "rows_per_checkpoint_shard", None)
        shard_padded = getattr(module, "padded_rows_per_shard", None)
        if rows != logical or not isinstance(shard_padded, int):
            continue
        if padded is None:
            padded = shard_padded
        elif padded != shard_padded:
            raise RuntimeError(f"PLE modules disagree on padded_rows_per_shard: {padded} vs {shard_padded}.")
    if padded is None:
        raise RuntimeError("Qwen4-Exp model has logical PLE rows but no padded_rows_per_shard.")
    if padded < logical:
        raise RuntimeError(f"PLE padded rows {padded} are smaller than logical rows {logical}.")
    return logical, padded


def _is_ple_weight(name: str) -> bool:
    if _PLE_SHARD_PATTERN.match(name):
        return True
    # DCP metadata keys are ``model.`` plus the state-dict name.
    if name.startswith("model.") and _PLE_SHARD_PATTERN.match(name[len("model.") :]):
        return True
    return False


def _shards_dim0(tensor: DTensor) -> bool:
    return any(isinstance(placement, Shard) and placement.dim == 0 for placement in tensor.placements)


def _checkpoint_name(name: str) -> str:
    if _PLE_SHARD_PATTERN.match(name):
        return name
    if name.startswith("model.") and _PLE_SHARD_PATTERN.match(name[len("model.") :]):
        return name[len("model.") :]
    return name


def _project_one(tensor: torch.Tensor, crop, logical_rows: int, name: str) -> torch.Tensor:
    """Project one tensor onto the checkpoint row count.

    The row cut itself is ``converter.crop_hf_export_tensor``. A dim-0 DTensor
    is gathered first because that cut is only valid on the full matrix, and
    the logical row count is not divisible by the ``ple`` mesh.
    """
    if tensor.shape[0] == logical_rows:
        return tensor
    checkpoint_name = _checkpoint_name(name)
    if isinstance(tensor, DTensor) and _shards_dim0(tensor):
        # Collective. Every rank must reach this for the same name. One PLE
        # shard is gathered, cropped, and released before the next one.
        full = tensor.full_tensor().detach()
        cropped = crop(checkpoint_name, full)
        del full
        return distribute_tensor(cropped, tensor.device_mesh, tensor.placements)
    if isinstance(tensor, DTensor):
        local = crop(checkpoint_name, tensor.to_local().detach())
        shape = torch.Size([logical_rows, *tensor.shape[1:]])
        return DTensor.from_local(local, tensor.device_mesh, tensor.placements, shape=shape)
    return crop(checkpoint_name, tensor.detach())


def _pad_one(tensor: torch.Tensor, padded_rows: int, name: str) -> torch.Tensor:
    rows = tensor.shape[0]
    if rows == padded_rows:
        return tensor
    if rows > padded_rows:
        raise RuntimeError(f"{name}: DCP rows {rows} are longer than padded rows {padded_rows}.")
    if isinstance(tensor, DTensor) and _shards_dim0(tensor):
        full = tensor.full_tensor().detach()
        tail = full.new_zeros((padded_rows - full.shape[0], *full.shape[1:]))
        padded = torch.cat((full, tail), dim=0)
        del full
        return distribute_tensor(padded, tensor.device_mesh, tensor.placements)
    if isinstance(tensor, DTensor):
        local = tensor.to_local().detach()
        tail = local.new_zeros((padded_rows - local.shape[0], *local.shape[1:]))
        local = torch.cat((local, tail), dim=0)
        shape = torch.Size([padded_rows, *tensor.shape[1:]])
        return DTensor.from_local(local, tensor.device_mesh, tensor.placements, shape=shape)
    tail = tensor.new_zeros((padded_rows - tensor.shape[0], *tensor.shape[1:]))
    return torch.cat((tensor.detach(), tail), dim=0)


def project_state_dict_to_checkpoint(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Store PLE weights at ``rows_per_checkpoint_shard``.

    DCP save and Hugging Face export both call this. The row cut is
    ``Qwen4ExpCheckpointTensorConverter.crop_hf_export_tensor``. Models without
    that converter are returned unchanged.
    """
    counts = ple_row_counts(model)
    if counts is None:
        return state_dict
    logical, _padded = counts
    converter = get_checkpoint_tensor_converter(model)
    crop = getattr(converter, "crop_hf_export_tensor", None)
    if crop is None:
        return state_dict
    projected = dict(state_dict)
    changed = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor) or not _is_ple_weight(name):
            continue
        updated = _project_one(tensor, crop, logical, name)
        if updated is not tensor:
            changed += 1
        projected[name] = updated
    if changed:
        logger.info_rank0(f"Projected {changed} PLE tensor(s) to {logical} checkpoint rows.")
    return projected if changed else state_dict


def crop_ple_state_dict_for_dcp(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """DCP entry point. Same projection as Hugging Face export."""
    return project_state_dict_to_checkpoint(model, state_dict)


def pad_ple_state_dict_from_dcp(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pad cropped PLE weights back to the runtime ``padded_rows_per_shard``."""
    counts = ple_row_counts(model)
    if counts is None:
        return state_dict
    _logical, padded = counts
    restored = dict(state_dict)
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor) or not _is_ple_weight(name):
            continue
        restored[name] = _pad_one(tensor, padded, name)
    return restored


def ple_checkpoint_is_cropped(checkpoint_dir: str, model: nn.Module) -> bool:
    """Whether ``checkpoint_dir`` stored PLE weights at the logical row count.

    Checkpoints written before this crop still contain the padded runtime shape
    and load without a second pad.
    """
    counts = ple_row_counts(model)
    if counts is None:
        return False
    logical, padded = counts
    metadata = FileSystemReader(checkpoint_dir).read_metadata()
    saved = None
    for key, tensor_meta in metadata.state_dict_metadata.items():
        if not hasattr(tensor_meta, "size") or not _is_ple_weight(key):
            continue
        rows = int(tensor_meta.size[0])
        if saved is None:
            saved = rows
        elif saved != rows:
            raise RuntimeError(f"PLE checkpoint rows disagree: {saved} vs {rows} ({key}).")
    if saved is None:
        return False
    if saved == logical:
        return True
    if saved == padded:
        return False
    raise RuntimeError(
        f"PLE checkpoint rows {saved} are neither the logical row count {logical} "
        f"nor the padded runtime row count {padded}."
    )
