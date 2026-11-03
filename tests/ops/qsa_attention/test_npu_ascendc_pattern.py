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

"""CPU checks for the npu_ascendc block-id adapter. NPU numerics live under docs/gbsa_veomni."""

import torch

from veomni.ops.kernels.qsa_attention.standard import npu_ascendc as qsa


def test_token_ids_collapse_to_unique_blocks():
    indices = torch.tensor([[[0, 1, 4, 5, -1], [2, 2, 6, -1, 7]]], dtype=torch.int32)
    blocks = qsa.token_indices_to_block_ids(indices, block_size=4, seq_len=8)
    assert blocks.dtype == torch.int32
    assert blocks.shape == (1, 2, 2)
    torch.testing.assert_close(blocks[0, 0], torch.tensor([0, 1], dtype=torch.int32))
    torch.testing.assert_close(blocks[0, 1], torch.tensor([0, 1], dtype=torch.int32))


def test_forward_and_backward_patterns_select_the_same_tokens():
    batch, sequence, hkv, block_size = 2, 11, 2, 4
    routing = torch.full((batch, sequence, 3), -1, dtype=torch.int32)
    for row in range(sequence):
        complete, tail = divmod(row + 1, block_size)
        chosen = list(range(min(complete, 2)))
        if tail:
            chosen.append(complete)
        routing[:, row, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
    valid = routing >= 0
    ids = torch.where(valid, routing, sequence).sort(-1).values
    ids = torch.where(ids < sequence, ids, 0).reshape(1, batch * sequence, -1).expand(hkv, -1, -1).contiguous()
    counts = valid.sum(-1).int().reshape(1, -1).expand(hkv, -1).contiguous()
    grad_ids, grad_counts = qsa._qsag_pattern(ids, counts, batch, sequence, block_size)
    for b in range(batch):
        for h in range(hkv):
            for row in range(sequence):
                forward = routing[b, row][routing[b, row] >= 0].tolist()
                backward = grad_ids[b, h, row, : grad_counts[b, h, row]].tolist()
                assert all(block < (row + 1) // block_size for block in backward)
                if (row + 1) % block_size:
                    backward.append((row + 1) // block_size)
                assert set(forward) == set(backward)
