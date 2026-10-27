# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DCP save stores logical PLE rows and reloads them without changing a value."""

import shutil
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.checkpoint.layout import step_dir, weights_dir
from veomni.checkpoint.ple_rows import ple_checkpoint_is_cropped
from veomni.models.transformers.qwen4_exp.checkpoint_tensor_converter import (
    create_qwen4_exp_checkpoint_tensor_converter,
)
from veomni.utils.save_safetensor_utils import crop_state_dict_for_hf_export, get_model_save_state


PADDED = 16
LOGICAL = 11
COLS = 8
SENTINEL = -123.0


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.ngram_embedding = nn.ModuleDict({"shard_0": nn.Embedding(PADDED, COLS)})
        self.other = nn.Linear(4, 4, bias=False)
        self.rows_per_checkpoint_shard = LOGICAL
        self.padded_rows_per_shard = PADDED
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(split_ngram_parts=4, make_ngram_vocab_size_divisible_by=8)
        )
        self._create_checkpoint_tensor_converter = create_qwen4_exp_checkpoint_tensor_converter


def _parallel_state():
    return SimpleNamespace(dp_mode="fsdp2")


def _reference():
    rows = torch.arange(PADDED * COLS, dtype=torch.float32).reshape(PADDED, COLS)
    rows[LOGICAL:] = SENTINEL
    other = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 0.5
    return rows, other


def _install(model, mesh, rows, other):
    weight = distribute_tensor(rows, mesh, [Shard(1), Shard(0)])
    model.block.ngram_embedding["shard_0"].weight = nn.Parameter(weight)
    with torch.no_grad():
        model.other.weight.copy_(other)


def _full(weight):
    tensor = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
    return tensor.detach().cpu()


def _train_ple(model):
    """Move only the logical rows, leaving the runtime pad untouched."""
    weight = model.block.ngram_embedding["shard_0"].weight
    full = weight.full_tensor().detach().clone()
    full[:LOGICAL] += 1
    updated = distribute_tensor(full, weight.device_mesh, weight.placements)
    with torch.no_grad():
        weight.to_local().copy_(updated.to_local())
    return full


def _freeze_ple(model):
    for parameter in model.block.parameters():
        parameter.requires_grad_(False)


def _max_abs(left, right):
    return (left.detach().cpu() - right.detach().cpu()).abs().max().item()


def _check_resume(model, trained_rows, trained_other, source_opt, opt):
    loaded = _full(model.block.ngram_embedding["shard_0"].weight)
    logical_diff = _max_abs(loaded[:LOGICAL], trained_rows[:LOGICAL])
    tail_diff = _max_abs(loaded[LOGICAL:], torch.zeros_like(loaded[LOGICAL:]))
    other_diff = _max_abs(model.other.weight, trained_other)
    assert tuple(loaded.shape) == (PADDED, COLS)
    assert logical_diff == 0.0, logical_diff
    assert tail_diff == 0.0, tail_diff
    assert SENTINEL not in set(loaded.reshape(-1).tolist())
    assert other_diff == 0.0, other_diff
    source_param = source_opt.param_groups[0]["params"][0]
    restored_param = opt.param_groups[0]["params"][0]
    for key in ("exp_avg", "exp_avg_sq"):
        moment_diff = _max_abs(opt.state[restored_param][key], source_opt.state[source_param][key])
        assert moment_diff == 0.0, (key, moment_diff)
    assert opt.state[restored_param]["step"].item() == source_opt.state[source_param]["step"].item()
    return logical_diff, tail_diff, other_diff


def main():
    # The container's training backend is NPU, but these ranks use gloo so the
    # checkpointer's agreement collectives must stay on CPU.
    import veomni.utils.dist_utils as dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ple_fsdp", "ple"))
    rows, other = _reference()
    source = _Model()
    _install(source, mesh, rows.clone(), other.clone())
    trained_rows = _train_ple(source)
    _freeze_ple(source)
    source_opt = torch.optim.AdamW(source.other.parameters(), lr=0.01)
    source.other.weight.sum().backward()
    source_opt.step()
    trained_other = source.other.weight.detach().cpu().clone()

    directory = None
    if rank == 0:
        directory = tempfile.mkdtemp(prefix="ple-dcp-")
    holder = [directory]
    dist.broadcast_object_list(holder, src=0)
    directory = holder[0]
    parallel_state = _parallel_state()

    try:
        DistributedCheckpointer.save(
            directory,
            {"model": source, "optimizer": source_opt, "extra_state": {"lr_scheduler": None}},
            global_steps=1,
            parallel_state=parallel_state,
        )
        checkpoint = step_dir(directory, 1)
        weight_dir = weights_dir(checkpoint)
        assert ple_checkpoint_is_cropped(weight_dir, source)
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(weight_dir).read_metadata()
        ple_key = next(key for key in metadata.state_dict_metadata if "ngram_embedding.shard_0.weight" in key)
        saved_rows = int(metadata.state_dict_metadata[ple_key].size[0])
        assert saved_rows == LOGICAL, saved_rows

        dense = {
            "block.ngram_embedding.shard_0.weight": trained_rows,
            "other.weight": trained_other,
        }
        exported = crop_state_dict_for_hf_export(source, dense)
        hf_ple = exported["block.ngram_embedding.shard_0.weight"]
        assert tuple(hf_ple.shape) == (LOGICAL, COLS)
        assert _max_abs(hf_ple, trained_rows[:LOGICAL]) == 0.0
        assert torch.equal(exported["other.weight"], trained_other)

        hf_state = get_model_save_state(source, None, parallel_state=parallel_state)
        hf_saved = _full(hf_state["block.ngram_embedding.shard_0.weight"])
        assert tuple(hf_saved.shape) == (LOGICAL, COLS)
        assert hf_saved.dtype == torch.bfloat16
        assert _max_abs(hf_saved, trained_rows[:LOGICAL].to(torch.bfloat16)) == 0.0
        assert _max_abs(hf_state["other.weight"], trained_other.to(torch.bfloat16)) == 0.0

        restored = _Model()
        _install(restored, mesh, torch.zeros(PADDED, COLS), torch.zeros(4, 4))
        _freeze_ple(restored)
        restored_opt = torch.optim.AdamW(restored.other.parameters(), lr=0.01)
        DistributedCheckpointer.load(
            checkpoint,
            {"model": restored, "optimizer": restored_opt, "extra_state": {}},
            parallel_state=parallel_state,
        )
        logical_diff, tail_diff, other_diff = _check_resume(
            restored, trained_rows, trained_other, source_opt, restored_opt
        )

        DistributedCheckpointer.save(
            directory,
            {"model": restored, "optimizer": restored_opt, "extra_state": {"lr_scheduler": None}},
            global_steps=2,
            parallel_state=parallel_state,
        )
        second = _Model()
        _install(second, mesh, torch.full((PADDED, COLS), 9.0), torch.full((4, 4), 9.0))
        _freeze_ple(second)
        second_opt = torch.optim.AdamW(second.other.parameters(), lr=0.01)
        DistributedCheckpointer.load(
            step_dir(directory, 2),
            {"model": second, "optimizer": second_opt, "extra_state": {}},
            parallel_state=parallel_state,
        )
        _check_resume(second, trained_rows, trained_other, source_opt, second_opt)
        assert ple_checkpoint_is_cropped(weights_dir(step_dir(directory, 2)), second)

        if rank == 0:
            print(
                f"PASS saved_rows={saved_rows} logical_max_abs={logical_diff} "
                f"tail_max_abs={tail_diff} other_max_abs={other_diff} "
                f"hf_bf16_max_abs=0.0 resume_cycles=2",
                flush=True,
            )
    finally:
        if rank == 0 and directory:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    import os
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
