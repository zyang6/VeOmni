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
"""Checkpoint conversion for the Qwen4-Exp VLM-SFT integration.

The runtime deliberately preserves the released checkpoint's
``ngram_embedding.shard_N.weight`` layout. Each source shard is padded only on
dim 0 so ExtraParallel can stream its local row slice without ever
materializing or concatenating the complete ~95 GiB PLE table.

MTP remains outside the supported VLM-SFT model. Every weight loader uses the
optional ``should_skip_without_loading`` capability to pass over ``mtp.*``
tensors without reading them; :meth:`convert` still consumes any that reach it.
"""

import math
import re
from typing import Dict, List, Optional

import torch

from veomni.models.checkpoint.convert import ConvertedCheckpointTensor
from veomni.utils import logging


logger = logging.get_logger(__name__)

_PLE_SHARD_PATTERN = re.compile(r"^(?P<prefix>.+\.ngram_embedding)\.shard_(?P<shard>\d+)\.weight$")
_MTP_PATTERN = re.compile(r"(?:^|\.)mtp(?:\.|$)")


def _logical_rows_from_model(model) -> int:
    """Read the unpadded per-shard row count from a constructed PLE module."""
    found = None
    for module in model.modules():
        rows = getattr(module, "rows_per_checkpoint_shard", None)
        padded = getattr(module, "padded_rows_per_shard", None)
        if not isinstance(rows, int) or not isinstance(padded, int):
            continue
        if padded < rows:
            raise RuntimeError(f"PLE padded rows {padded} are smaller than logical rows {rows}.")
        if found is None:
            found = rows
        elif found != rows:
            raise RuntimeError(f"PLE modules disagree on rows_per_checkpoint_shard: {found} vs {rows}.")
    if found is None:
        raise RuntimeError("Qwen4-Exp model has no PLE rows_per_checkpoint_shard.")
    return found


class Qwen4ExpCheckpointTensorConverter:
    """Pad independent PLE shards and discard unsupported MTP tensors."""

    def __init__(self, split_ngram_parts: int, shard_row_divisor: int, logical_rows: int):
        self.split_ngram_parts = split_ngram_parts
        self.shard_row_divisor = shard_row_divisor
        self.logical_rows = logical_rows
        self.ignored_mtp_tensors = 0

    def can_handle(self, name: str) -> bool:
        return _MTP_PATTERN.search(name) is not None or _PLE_SHARD_PATTERN.match(name) is not None

    def convert(self, name: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        if _MTP_PATTERN.search(name) is not None:
            self.ignored_mtp_tensors += 1
            return None

        match = _PLE_SHARD_PATTERN.match(name)
        if match is None:
            return None

        shard_idx = int(match.group("shard"))
        if shard_idx >= self.split_ngram_parts:
            raise RuntimeError(
                f"Qwen4-Exp PLE shard index {shard_idx} is outside configured split_ngram_parts="
                f"{self.split_ngram_parts}: {name}"
            )
        padded_rows = math.ceil(tensor.shape[0] / self.shard_row_divisor) * self.shard_row_divisor
        if padded_rows != tensor.shape[0]:
            padding = tensor.new_zeros((padded_rows - tensor.shape[0], *tensor.shape[1:]))
            tensor = torch.cat((tensor, padding), dim=0)
        return ConvertedCheckpointTensor(name=name, tensor=tensor)

    def is_dim0_zero_pad(self, name: str) -> bool:
        """PLE shard conversion is only trailing dim-0 zero-padding."""
        return _PLE_SHARD_PATTERN.match(name) is not None

    def crop_hf_export_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Drop the runtime dim-0 pad from a full PLE matrix.

        Both DCP save and Hugging Face export use this. ``tensor`` must already
        be the full matrix, not one ``ple`` rank's local shard. Rows
        ``[logical_rows:)`` are the zeros added so the table can be split
        evenly. Lookup indexes only ``[0, logical_rows)``.
        """
        if _PLE_SHARD_PATTERN.match(name) is None:
            return tensor
        rows = tensor.shape[0]
        if rows == self.logical_rows:
            return tensor
        if rows < self.logical_rows:
            raise RuntimeError(f"{name}: export rows {rows} are shorter than logical rows {self.logical_rows}.")
        return tensor.narrow(0, 0, self.logical_rows).contiguous()

    def should_skip_without_loading(self, name: str) -> bool:
        """The streaming loader can discard unsupported MTP tensors by key."""
        return _MTP_PATTERN.search(name) is not None

    def record_skip_without_loading(self, name: str) -> None:
        if not self.should_skip_without_loading(name):
            raise ValueError(f"Qwen4-Exp cannot skip checkpoint tensor without loading: {name}")
        self.ignored_mtp_tensors += 1

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        if self.ignored_mtp_tensors:
            logger.warning_rank0(
                "Ignored %d Qwen4-Exp MTP checkpoint tensors because this integration supports VLM SFT only "
                "and does not construct an MTP module or compute MTP loss.",
                self.ignored_mtp_tensors,
            )
        return []


def create_qwen4_exp_checkpoint_tensor_converter(model):
    """Create the converter for a top-level or standalone Qwen4-Exp model."""
    config = getattr(model.config, "text_config", model.config)
    return Qwen4ExpCheckpointTensorConverter(
        split_ngram_parts=config.split_ngram_parts,
        shard_row_divisor=config.make_ngram_vocab_size_divisible_by,
        logical_rows=_logical_rows_from_model(model),
    )


def convert_qwen4_exp_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Map sharded PLE index entries to the runtime key and omit MTP entries."""
    converted: Dict[str, int] = {}
    for name, shard_index in fqn_to_index_mapping.items():
        if _MTP_PATTERN.search(name) is not None:
            continue
        converted[name] = shard_index
    return converted
