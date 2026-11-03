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
"""Patch configuration for the Qwen4-Exp NPU correctness path.

Regen command:
patchgen veomni.models.transformers.qwen4_exp.qwen4_exp_npu_patch_gen_config -o veomni/models/transformers/qwen4_exp/generated --diff

Qwen4-Exp uses partial interleaved mRoPE and experimental QSA/PLE code paths,
so this NPU build inherits the device-agnostic GPU patch set. The only
attention-path change is the AscendC GBSA/GBSAG call, and it stays in this
file so the GPU patch config and GPU generated modeling stay untouched.
"""

from copy import deepcopy

from veomni.models.transformers.qwen4_exp.qwen4_exp_gpu_patch_gen_config import config as gpu_config


config = deepcopy(gpu_config)
config.target_file = "patched_modeling_qwen4_exp_npu.py"
config.description = "Qwen4-Exp NPU VLM-SFT correctness integration with PLE sharding"

# Replace the GPU QSA indexer mask value in-place so NPU masks invisible block
# scores with -1.0 instead of -inf. ReLU block scores are non-negative, and
# CANN TopKV2 faults (AICore error 507015) on sparse rows dominated by -inf.
for _patch in config.patches:
    if _patch.target == "Qwen4ExpTextQSAIndexer.forward":
        _patch.name_map = {
            'scores = scores.masked_fill(~visible, float("-inf"))': (
                "scores = scores.masked_fill(~visible, -1.0)  # NPU TopKV2 faults on sparse -inf rows (ReLU scores >= 0)"
            ),
        }
        break
else:
    raise RuntimeError("Expected Qwen4ExpTextQSAIndexer.forward in qwen4_exp GPU patch config")

# GBSA/GBSAG is NPU-only. Rewrite the inherited attention forward in place:
# reject Ulysses/CP for npu_ascendc, and pass block_size only for that impl so
# eager / torch_npu / tilelang keep their original signatures.
for _patch in config.patches:
    if _patch.target == "Qwen4ExpTextAttention.forward":
        _patch.name_map = {
            (
                '    if sp_enabled and past_key_values is not None:\n'
                '        raise NotImplementedError("Qwen4-Exp QSA does not support a KV cache under Ulysses/CP.")\n'
                "    if sp_enabled:"
            ): (
                '    if sp_enabled and past_key_values is not None:\n'
                '        raise NotImplementedError("Qwen4-Exp QSA does not support a KV cache under Ulysses/CP.")\n'
                '    if sp_enabled and self.veomni_qsa_attn.impl == "npu_ascendc":\n'
                "        raise NotImplementedError(\n"
                '            "npu_ascendc QSA (GBSA/GBSAG) requires equal query and key sequence lengths. "\n'
                '            "Ulysses/CP gathers KV and is unsupported; use qsa_attention_implementation=\'torch_npu\'."\n'
                "        )\n"
                "    if sp_enabled:"
            ),
            (
                "    attn_output, attn_weights = self.veomni_qsa_attn(\n"
                "        query_states,\n"
                "        key_states,\n"
                "        value_states,\n"
                "        selection,\n"
                "        scaling=self.scaling,\n"
                "        dropout=0.0 if not self.training else self.attention_dropout,\n"
                "        attention_mask=attention_mask,\n"
                "    )"
            ): (
                "    qsa_kwargs = {}\n"
                '    if self.veomni_qsa_attn.impl == "npu_ascendc":\n'
                '        qsa_kwargs["block_size"] = self.indexer.compress_ratio\n'
                "    attn_output, attn_weights = self.veomni_qsa_attn(\n"
                "        query_states,\n"
                "        key_states,\n"
                "        value_states,\n"
                "        selection,\n"
                "        scaling=self.scaling,\n"
                "        dropout=0.0 if not self.training else self.attention_dropout,\n"
                "        attention_mask=attention_mask,\n"
                "        **qsa_kwargs,\n"
                "    )"
            ),
        }
        break
else:
    raise RuntimeError("Expected Qwen4ExpTextAttention.forward in qwen4_exp GPU patch config")
