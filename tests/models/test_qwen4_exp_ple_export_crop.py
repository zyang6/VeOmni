# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""HF export crops only the Qwen4-Exp PLE runtime pad."""

import json
import os
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from veomni.models.transformers.qwen4_exp.checkpoint_tensor_converter import (
    create_qwen4_exp_checkpoint_tensor_converter,
)
from veomni.utils.save_safetensor_utils import (
    crop_exported_hf_ple_files,
    crop_state_dict_for_hf_export,
)


_PLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight"


class _Ple(nn.Module):
    def __init__(self, logical_rows, padded_rows):
        super().__init__()
        self.rows_per_checkpoint_shard = logical_rows
        self.padded_rows_per_shard = padded_rows


class _Model(nn.Module):
    def __init__(self, logical_rows=10, padded_rows=16):
        super().__init__()
        self.ple = _Ple(logical_rows, padded_rows)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                split_ngram_parts=4,
                make_ngram_vocab_size_divisible_by=8,
            )
        )
        self._create_checkpoint_tensor_converter = create_qwen4_exp_checkpoint_tensor_converter


def _padded_weight(logical_rows=10, padded_rows=16, cols=4):
    weight = torch.arange(padded_rows * cols, dtype=torch.float32).reshape(padded_rows, cols)
    weight[logical_rows:] = 0
    return weight


def test_crop_keeps_logical_prefix_and_drops_zero_tail():
    model = _Model()
    converter = create_qwen4_exp_checkpoint_tensor_converter(model)
    weight = _padded_weight()
    cropped = converter.crop_hf_export_tensor(_PLE, weight)

    assert cropped.shape == (10, 4)
    assert torch.equal(cropped, weight[:10])


def test_crop_leaves_unpadded_and_non_ple_tensors():
    converter = create_qwen4_exp_checkpoint_tensor_converter(_Model())
    logical = _padded_weight()[:10]
    assert converter.crop_hf_export_tensor(_PLE, logical) is logical

    other = torch.zeros(16, 4)
    assert converter.crop_hf_export_tensor("model.embed_tokens.weight", other) is other


def test_crop_rejects_a_short_ple_shard():
    converter = create_qwen4_exp_checkpoint_tensor_converter(_Model())
    try:
        converter.crop_hf_export_tensor(_PLE, torch.zeros(9, 4))
    except RuntimeError as exc:
        assert "shorter than logical rows" in str(exc)
    else:
        raise AssertionError("expected a short PLE shard to be rejected")


def test_state_dict_helper_skips_models_without_the_crop():
    weight = torch.zeros(3, 2)
    state = {"model.embed_tokens.weight": weight}
    assert crop_state_dict_for_hf_export(nn.Linear(2, 2), state) is state


def test_state_dict_helper_crops_a_dense_ple_shard():
    model = _Model()
    weight = _padded_weight()
    state = crop_state_dict_for_hf_export(model, {_PLE: weight, "model.embed_tokens.weight": torch.zeros(2)})

    assert state[_PLE].shape == (10, 4)
    assert state["model.embed_tokens.weight"].shape == (2,)


def test_exported_file_crop_updates_shape_and_total_size(tmp_path):
    model = _Model()
    weight = _padded_weight()
    other = torch.ones(2, 2)
    save_file({_PLE: weight, "model.embed_tokens.weight": other}, str(tmp_path / "model-00001-of-00001.safetensors"))
    total_size = weight.numel() * weight.element_size() + other.numel() * other.element_size()
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {
            _PLE: "model-00001-of-00001.safetensors",
            "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
        },
    }
    with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
        json.dump(index, handle)

    crop_exported_hf_ple_files(model, str(tmp_path))

    loaded = load_file(str(tmp_path / "model-00001-of-00001.safetensors"))
    assert loaded[_PLE].shape == (10, 4)
    assert torch.equal(loaded[_PLE], weight[:10])
    assert torch.equal(loaded["model.embed_tokens.weight"], other)
    with open(tmp_path / "model.safetensors.index.json", encoding="utf-8") as handle:
        updated = json.load(handle)
    dropped = 6 * 4 * weight.element_size()
    assert updated["metadata"]["total_size"] == total_size - dropped
    assert not os.path.exists(str(tmp_path / "model-00001-of-00001.safetensors.crop.tmp"))
