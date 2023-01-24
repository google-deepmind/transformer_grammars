# Copyright 2021-2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Common code to training and evaluation."""

import haiku as hk
from transformer_grammars.models import lm
from transformer_grammars.models.masking import utils as masking_utils


def build_forward(model_cfg, maskrules, token_type_ranges, *, is_training):
  """Builds a forward function for the model."""

  model_kwargs = dict(**model_cfg)
  model_kwargs.pop("extra_attention_mask_name", None)
  model_kwargs.pop("extra_attention_mask_kwargs", None)
  model_kwargs["num_attns"] = maskrules.num_attention_functions
  model_kwargs["vocab_size"] = token_type_ranges.vocab_size

  @hk.transform_with_state
  def forward(
      *,
      inputs,
      inputs_ttypes,
      attn_mask,
      attn_relpos,
      attn_indicator,
      memory_attn_mask,
      memory_padding_mask,
      smartmem_mem_from_seq,
      smartmem_mem_from_mem,
      beginning_of_seq,
  ):
    model = lm.GeneralizedTXLLanguageModel(**model_kwargs)
    output, unused_layers_outputs = model(
        inputs,
        beginning_of_seq=beginning_of_seq,
        token_type=inputs_ttypes,
        attn_mask=attn_mask,
        attn_relpos=attn_relpos,
        attn_indicator=attn_indicator,
        memory_attn_mask=memory_attn_mask,
        memory_padding_mask=memory_padding_mask,
        smartmem_mem_from_mem=smartmem_mem_from_mem,
        smartmem_mem_from_seq=smartmem_mem_from_seq,
        is_training=is_training,
    )
    return output

  return forward


def build_maskrules(model_cfg):
  maskrules_name = model_cfg.get("extra_attention_mask_name", "txl")
  maskrules_kwargs = model_cfg.get("extra_attention_mask_kwargs", {})
  return masking_utils.get_masking_rules(
      maskrules_name,
      sequence_length=model_cfg["sequence_length"],
      memory_length=model_cfg["memory_length"],
      **maskrules_kwargs,
  )


def model_input_from_chunk(chunk, maskrules):
  """Returns model input from masking rules chunk."""
  d = chunk._asdict()
  for key in [
      "memory_pos",
      "depth",
      "end_of_seq",
      "labels",
      "labels_ttypes",
      "seq_idx",
  ]:
    del d[key]
  if not maskrules.use_relative_positions:
    d["attn_relpos"] = None
  return d
