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

"""Embedding layer."""

import haiku as hk
import jax.numpy as jnp


class EmbeddingLayer(hk.Module):
  """Embedding layer that allows weight sharing."""

  def __init__(
      self,
      embedding_size: int,
      vocab_size: int,
      dtype: jnp.dtype = jnp.float32,
      share_weights: bool = True,
      output_bias: bool = True,
      name: str = "embedding_layer",
  ):
    """Initialises the module."""

    super().__init__(name=name)
    self._embedding_size = embedding_size
    self._vocab_size = vocab_size
    self._dtype = dtype
    self._output_bias = output_bias

    self._embedding_weights = hk.get_parameter(
        name="input_embedding",
        shape=[self._vocab_size, self._embedding_size],
        dtype=self._dtype,
        init=hk.initializers.VarianceScaling(
            distribution="uniform", mode="fan_out"
        ),
    )

    if share_weights:
      self._output_weights = jnp.transpose(self._embedding_weights)
    else:
      self._output_weights = hk.get_parameter(
          name="output_weights",
          shape=[self._embedding_size, self._vocab_size],
          dtype=self._dtype,
          init=hk.initializers.VarianceScaling(
              distribution="uniform", mode="fan_out", scale=1.0
          ),
      )

  def encode(self, input_tokens: jnp.ndarray) -> jnp.ndarray:
    """Map tokens to embeddings."""
    assert jnp.issubdtype(input_tokens.dtype, jnp.integer)
    # If you don't wrap ids in a singleton tuple then JAX will try to unpack
    # it along the row dimension and treat each row as a separate index into
    # one of the dimensions of the array. The error only surfaces when
    # indexing with DeviceArray, while indexing with numpy.ndarray works fine.
    # See https://github.com/google/jax/issues/620 for more details.
    # Cast to a jnp array in case `ids` is a tracer (eg a dynamic_unroll).
    return jnp.asarray(self._embedding_weights)[(input_tokens,)]

  def decode(self, embeddings: jnp.ndarray) -> jnp.ndarray:
    """Decode embeddings to token logits."""
    out = jnp.matmul(embeddings, self._output_weights)
    if self._output_bias:
      bias = hk.get_parameter(
          "bias",
          shape=[self._vocab_size],
          dtype=self._dtype,
          init=jnp.zeros,
      )
      out += bias
    return out
