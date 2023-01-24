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

"""Generalized Transformer-XL-based language model.

It is generalized to relative positions different from difference in linear
order, custom attention masks, and memory update.

Embeds, applies the core, projects to get logits.

See the documentation for the core about TG specific changes.
"""

from typing import Callable, Optional
from einshape import jax_einshape as einshape
import haiku as hk
import jax
import jax.numpy as jnp
from transformer_grammars.models import core
from transformer_grammars.models import embedding_layer


def _apply_mask(m, a, b, *, axis):
  """Returns an array composed of elements from a or b depending on the mask.

  Args:
    m: Mask used to select elements from `a` when equal to 1, from `b`
      otherwise, of shape [m]
    a: Array of arbitrary shape [d_0, d_1, ..., d_{n-1}]
    b: Array of same shape as `a`
    axis: Axis, in `a` and `b`, such that d_{axis} == m.

  Returns:
    Array `c` such that slices, on axis `axis`, for which m == 1, are
    extracted from `a, and for which m == 0, are extracted from `b`.
  """
  assert a.shape == b.shape, (a.shape, b.shape)
  assert m.shape[0] == a.shape[axis], (m.shape, a.shape, axis)
  new_shape = [1 if i != axis else -1 for (i, _) in enumerate(a.shape)]
  m = m.reshape(new_shape)
  return a * m + b * (1 - m)


class GeneralizedTXLLanguageModel(hk.Module):
  """Generalized Transformer-XL-based language model."""

  def __init__(
      self,
      *,
      vocab_size: int,
      d_model: int,
      num_layers: int,
      num_heads: int,
      ffw_hidden_size: int,
      core_dropout: float,
      core_output_dropout: float,
      embedding_dropout: float,
      sequence_length: int,
      memory_length: int,
      tied_input_output_embeddings: bool,
      use_output_bias: bool = True,
      key_size: Optional[int] = None,
      value_size: Optional[int] = None,
      relative_position_embeddings: bool = True,
      use_attn_bias: bool = False,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
      num_attns: int = 1,
      tied_layer_weights: bool = False,
      num_unrestricted_layers: int = 0,
      min_relative_position: Optional[int] = None,
      max_relative_position: Optional[int] = None,
      num_unrestricted_heads: Optional[int] = None,
      name: str = "lm",
  ):
    """Initialises the module.

    Args:
      vocab_size: Vocabulary size.
      d_model: Size of the embeddings.
      num_layers: Number of transformer block layers.
      num_heads: Number of attention heads to use.
      ffw_hidden_size: Hidden size for MLP that follows attention.
      core_dropout: Dropout rate for the TXL core, applied to the embeddings,
        attention and MLP modules.
      core_output_dropout: Dropout rate applied to the output of the core,
        before projection.
      embedding_dropout: Dropout rate for the token embeddings, before them
        being passed to the core.
      sequence_length: (Unused) Input sequence length.
      memory_length: How many tokens to hold in Core memory.
      tied_input_output_embeddings: Use the same embedding matrix for the input
        and for the output.
      use_output_bias: Apply a learned bias to the output logits.
      key_size: Size of key (and query) embedding for attention. If not passed,
        defaults to d_model / num_heads.
      value_size: Size of value embedding for attention. If not passed, defaults
        to d_model / num_heads.
      relative_position_embeddings: Whether to use relative position embeddings.
      use_attn_bias: Whether or not to use biases in attention linear layers.
      activation: The nonlinearity to use in the DenseBlocks.
      num_attns: (TG only) Number of attention functions (e.g. 2 if different
        attns for STACK and COMPOSE)
      tied_layer_weights: If True, all the layers share the same weights.
      num_unrestricted_layers: (TG only) Number of regular TXL (no Transformer
        Grammar restriction) layers (0 means no regular TXL layer, n > 0 means
        n at the top of the stack, n < 0 means -n at the bottom of the stack)
      min_relative_position: (TG only) Minimum value for the relative positions
        that can be passed to the model.
      max_relative_position: (TG only) Maximum value for the relative positions
        that can be passed to the model.
      num_unrestricted_heads: (TG only) For TG layers, number of unrestricted
        (TXL) heads.
      name: The Haiku name of the module.
    """
    super().__init__(name=name)
    del sequence_length
    if key_size is None:
      key_size = d_model // num_heads
    if value_size is None:
      value_size = d_model // num_heads
    if ffw_hidden_size < 0:
      # Negative ffw_hidden_size is used to express a ratio between d_model
      # and FFW d_hidden. Useful for sweeps.
      ffw_hidden_size = d_model * (-ffw_hidden_size)
    self._core = core.Core(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        key_size=key_size,
        value_size=value_size,
        ffw_hidden_size=ffw_hidden_size,
        dropout_rate=core_dropout,
        memory_length=memory_length,
        relative_position_embeddings=relative_position_embeddings,
        use_attn_bias=use_attn_bias,
        activation=activation,
        tied_layer_weights=tied_layer_weights,
        num_attns=num_attns,
        num_unrestricted_layers=num_unrestricted_layers,
        min_relative_position=min_relative_position,
        max_relative_position=max_relative_position,
        num_unrestricted_heads=num_unrestricted_heads,
        name="core",
    )
    self._num_layers = num_layers
    self._d_model = d_model
    self._memory_length = memory_length
    self._embed = embedding_layer.EmbeddingLayer(
        embedding_size=d_model,
        vocab_size=vocab_size,
        share_weights=tied_input_output_embeddings,
        output_bias=use_output_bias,
    )
    self._embedding_dropout = embedding_dropout
    self._core_output_dropout = core_output_dropout

  def __call__(
      self,
      seq,
      beginning_of_seq,
      *,
      attn_mask,
      attn_relpos,
      attn_indicator,
      memory_attn_mask,
      memory_padding_mask,
      smartmem_mem_from_seq,
      smartmem_mem_from_mem,
      is_training: bool,
      **kwargs,
  ):
    """Applies the model to a sequence."""
    bs, seqlen = seq.shape
    use_memory = self._memory_length > 0
    mask = jnp.greater(seq, 0)
    emb = self._embed.encode(seq)
    if is_training:
      # WARNING: The core applies dropout to the token embeddings as well, so
      # the effective dropout rate for the embeddings is
      # embedding_dropout + core_dropout - embedding_dropout * core_dropout.
      emb = hk.dropout(hk.next_rng_key(), self._embedding_dropout, emb)

    # Get the memory from Haiku state
    if use_memory:
      # `memory` is the saved activations from each layer
      memory = hk.get_state(
          "memory",
          shape=[
              self._num_layers,
              bs,
              self._memory_length,
              self._d_model,
          ],
          dtype=emb.dtype,
          init=jnp.zeros,
      )
      empty_memory = jnp.zeros_like(memory)
      memory = _apply_mask(beginning_of_seq, empty_memory, memory, axis=1)
      memory_padding_mask = jnp.tile(
          einshape("bt->b1t", memory_padding_mask), (1, seqlen, 1)
      )
    else:
      memory = None
      memory_padding_mask = None

    core_output, new_memory, layers_outputs = self._core(
        input_embeddings=emb,
        input_mask=mask,
        memory=memory,
        memory_mask=memory_padding_mask,
        is_training=is_training,
        extra_attention_mask=attn_mask,
        extra_memory_attention_mask=memory_attn_mask,
        relative_positions=attn_relpos,
        attn_indicator=attn_indicator,
        smartmem_mem_from_seq=smartmem_mem_from_seq,
        smartmem_mem_from_mem=smartmem_mem_from_mem,
    )

    # Set the memory into Haiku state.
    if memory is not None:
      hk.set_state("memory", jax.lax.stop_gradient(new_memory))

    if is_training:
      core_output = hk.dropout(
          hk.next_rng_key(), self._core_output_dropout, core_output
      )
    output = self._embed.decode(core_output)
    return output, layers_outputs
