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

"""Tests for transformer_grammars.models.lm."""

import unittest
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from transformer_grammars.models import lm


class AbstractArray(object):
  """Abstract JAX array."""

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = jnp.dtype(dtype)


def get_inputs():
  batch_size = 64
  sequence_length = 256
  inputs = AbstractArray((batch_size, sequence_length), jnp.int32)
  inputs_ttypes = AbstractArray((batch_size, sequence_length), jnp.int32)
  attn_mask = AbstractArray(
      (batch_size, sequence_length, sequence_length), jnp.int32
  )
  attn_relpos = AbstractArray(
      (batch_size, sequence_length, 2 * sequence_length), jnp.int32
  )
  attn_indicator = AbstractArray((batch_size, sequence_length), jnp.int32)
  memory_attn_mask = AbstractArray(
      (batch_size, sequence_length, sequence_length), jnp.int32
  )
  memory_padding_mask = AbstractArray((batch_size, sequence_length), jnp.int32)
  smartmem_mem_from_seq = AbstractArray(
      (batch_size, sequence_length, sequence_length), jnp.int32
  )
  smartmem_mem_from_mem = AbstractArray(
      (batch_size, sequence_length, sequence_length), jnp.int32
  )
  beginning_of_seq = AbstractArray((batch_size,), jnp.int32)
  return dict(
      seq=inputs,
      token_type=inputs_ttypes,
      beginning_of_seq=beginning_of_seq,
      attn_mask=attn_mask,
      attn_relpos=attn_relpos,
      attn_indicator=attn_indicator,
      memory_attn_mask=memory_attn_mask,
      memory_padding_mask=memory_padding_mask,
      smartmem_mem_from_mem=smartmem_mem_from_mem,
      smartmem_mem_from_seq=smartmem_mem_from_seq,
  )


def apply_fn(**kwargs):
  model = lm.GeneralizedTXLLanguageModel(
      vocab_size=32768,
      d_model=1024,
      num_layers=16,
      num_heads=8,
      ffw_hidden_size=4096,
      embedding_dropout=0.1,
      core_dropout=0.1,
      core_output_dropout=0.1,
      sequence_length=256,
      memory_length=256,
      tied_input_output_embeddings=True,
      relative_position_embeddings=1,
      tied_layer_weights=0,
  )
  output, _ = model(**kwargs, is_training=True)
  return output


class CoreTest(unittest.TestCase):

  def test_expected_num_params(self):
    inputs_dict = get_inputs()
    transformed = hk.transform_with_state(apply_fn)
    params, _ = jax.eval_shape(
        transformed.init, jax.random.PRNGKey(0), **inputs_dict
    )
    num_params = sum([np.product(x.shape) for x in jax.tree_flatten(params)[0]])
    self.assertEqual(num_params, 251887616)


if __name__ == "__main__":
  unittest.main()
