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

"""Generalized Transformer-XL implementation.

Extended for TG:
- to accept any mask, relative positions
- to possibly apply different attention functions at different positions
- to tie layers, or not
- to have some layers be restricted, or not
- to have some heads be restricted, or not
"""

# pylint: disable=g-complex-comprehension

from typing import Any, Callable, Mapping, Optional, Tuple

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

# The memory consists of the inputs to every transformer layer, and has
# shape [num_layers, batch_size, memory_length, d_model]
TransformerMemory = jnp.ndarray  # pylint: disable=invalid-name


def layer_norm(
    name: Optional[str] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)


def broadcast_batch(x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
  return jnp.broadcast_to(x, (batch_size,) + x.shape)


def make_attention_mask(
    input_mask: jnp.ndarray,
    memory_mask: Optional[jnp.ndarray] = None,
    extra_attention_mask: Optional[jnp.ndarray] = None,
    extra_memory_attention_mask: Optional[jnp.ndarray] = None,
    memory_len: Optional[int] = None,
    dtype: jnp.dtype = jnp.float32,
    causal: Optional[bool] = True,
) -> jnp.ndarray:
  """Creates the attention mask out of component masks.

  Args:
    input_mask: Array of shape [B, T].
    memory_mask: Optional array of shape [B, T, M].
    extra_attention_mask: Optional array of shape [B, T, T].
    extra_memory_attention_mask: Optional array of shape [B, T, M].
    memory_len: M.
    dtype: Return dtype.
    causal: Whether the mask is causal.

  Returns:
    Array of shape [B, T, T + M].
  """
  batch_size, seq_len = input_mask.shape

  if causal:
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.bool_))
    # Ensure that the baked-in constant is only of size (seq_len, seq_len).
    causal_mask = lax.tie_in(input_mask, causal_mask)
    attention_mask = input_mask[:, None, :] * causal_mask[None, :, :]
  else:
    attention_mask = jnp.broadcast_to(
        input_mask[:, np.newaxis, :], [batch_size, seq_len, seq_len]
    )

  attention_mask = attention_mask.astype(dtype)
  unrestricted_attention_mask = attention_mask
  if extra_attention_mask is not None:
    attention_mask *= extra_attention_mask

  # Prepend memory_mask to attention_mask if needed.
  if memory_len:
    if memory_mask is None:
      mask_shape = (seq_len, memory_len)
      memory_mask = np.ones(mask_shape, dtype=np.bool_)
      # Ensure that the baked-in constant is only of size (seq_len, mem_len).
      memory_mask = lax.tie_in(input_mask, memory_mask)
      memory_mask = broadcast_batch(memory_mask.astype(dtype), batch_size)
    unrestricted_memory_mask = memory_mask
    if extra_memory_attention_mask is not None:
      memory_mask *= extra_memory_attention_mask
    attention_mask = jnp.concatenate((memory_mask, attention_mask), axis=-1)
    unrestricted_attention_mask = jnp.concatenate(
        (unrestricted_memory_mask, unrestricted_attention_mask), axis=-1
    )

  # Verify we did it right.
  assert attention_mask.dtype == dtype
  assert attention_mask.shape == (batch_size, seq_len, seq_len + memory_len)
  assert unrestricted_attention_mask.dtype == attention_mask.dtype
  assert unrestricted_attention_mask.shape == attention_mask.shape
  return attention_mask, unrestricted_attention_mask


def _apply_mask_to_tensor(x, mask):
  num_mask_dims = len(mask.shape)
  num_x_dims = len(x.shape)
  assert num_x_dims >= num_mask_dims
  assert x.shape[:num_mask_dims] == mask.shape
  for _ in range(num_x_dims - num_mask_dims):
    mask = jnp.expand_dims(mask, -1)
  return x * mask


def _apply_mask_to_pytree(x, mask):
  f = lambda y: _apply_mask_to_tensor(y, mask)
  return jax.tree_map(f, x)


def switch(funcs, indicator, *args, **kwargs):
  """Applies one of the functions, depending on the value of the indicator."""
  num_classes = len(funcs)
  assert indicator is not None or len(funcs) == 1
  if len(funcs) == 1:
    f = funcs[0]
    return f(*args, **kwargs)
  outs = [f(*args, **kwargs) for f in funcs]
  masks = [jnp.equal(indicator, i) for i in range(num_classes)]
  masked_outs = [
      _apply_mask_to_pytree(x, mask) for (x, mask) in zip(outs, masks)
  ]
  return jax.tree_map(sum, *masked_outs)


def _rel_shift_inner(logits: jnp.ndarray, attention_length: int) -> jnp.ndarray:
  """Shifts the relative logits.

  This is a more general than the original Transformer-XL implementation as
  inputs may also see the future. (The implementation does not rely on a
  causal mask removing the upper-right triangle.)

  Given attention length 3 and inputs:
      [[-3, -2, -1, 0, 1, 2],
       [-3, -2, -1, 0, 1, 2],
       [-3, -2, -1, 0, 1, 2]]

  The shifted output is:
      [[0, 1, 2],
       [-1, 0, 1],
       [-2, -1, 0]]

  Args:
    logits: input tensor of shape [T_q, T_v + T_q]
    attention_length: T_v `int` length of the attention, should be equal to
      memory size + sequence length.

  Returns:
    A shifted version of the input of size [T_q, T_v]. In each row, a window
    of size T_v elements is kept. The window starts at the rightmost end, for
    the first row. It then shifts left by 1 for each subsequent row.
  """
  if logits.ndim != 2:
    raise ValueError("`logits` needs to be an array of dimension 2.")
  tq, total_len = logits.shape
  assert total_len == tq + attention_length
  logits = jnp.reshape(logits, [total_len, tq])
  logits = lax.slice(logits, (1, 0), logits.shape)  # logits[1:]
  logits = jnp.reshape(logits, [tq, total_len - 1])
  # Equiv to logits[:, :attention_length].
  logits = lax.slice(logits, (0, 0), (tq, attention_length))
  return logits


def relative_shift(logits: jnp.ndarray, attention_length: int) -> jnp.ndarray:
  fn = lambda t: _rel_shift_inner(t, attention_length)
  return jax.vmap(jax.vmap(fn))(logits)


class MultiHeadAttention(hk.Module):
  """Multihead attention module with relative position encodings and memory.

  With TG changes to accept any mask/relative position.
  """

  def __init__(
      self,
      *,
      value_size: int,
      key_size: int,
      num_heads: int,
      init_scale: float,
      dropout_rate: float,
      use_bias: bool,
      use_final_bias: bool,
      final_init_scale_multiplier: float,
      min_relative_position: Optional[int] = None,
      max_relative_position: Optional[int] = None,
      apply_final_linear: bool = True,
      name: str = "multihead_attention",
  ):
    """Initialises the MultiHeadAttention module."""
    super().__init__(name=name)
    self._value_size = value_size
    self._key_size = key_size
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate
    self._init_scale = init_scale
    self._final_init_scale = final_init_scale_multiplier * init_scale
    self._use_bias = use_bias
    self._use_final_bias = use_final_bias
    self._min_relative_position = min_relative_position
    self._max_relative_position = max_relative_position
    self._apply_final_linear = apply_final_linear

  @hk.transparent
  def _multihead_linear(self, inputs: jnp.ndarray, hidden_size: int, name: str):
    linear = hk.Linear(
        self._num_heads * hidden_size,
        with_bias=self._use_bias,
        w_init=hk.initializers.VarianceScaling(scale=self._init_scale),
        name=name,
    )
    out = linear(inputs)
    return jnp.reshape(out, inputs.shape[:-1] + (self._num_heads, hidden_size))

  def __call__(
      self,
      inputs: jnp.ndarray,
      attention_mask: jnp.ndarray,
      positional_encodings: Optional[jnp.ndarray],
      memory: Optional[jnp.ndarray],
      relative_positions: Optional[jnp.ndarray],
      is_training: bool,
  ) -> jnp.ndarray:
    """Computes the attention values.

    We use the following shape conventions: `B` for batch size, `T` for
    chunk
    size, `M` for memory length and `D` for the embedding dimension.

    Args:
      inputs: Array of shape [B, T, D]
      attention_mask: Array of shape [B, T, M + T] indicating which attention
        pairs are valid.
      positional_encodings: Optional array of shape [B, R, D] where R is the
        number of possible relative positions.
      memory: Optional array of extra attention values of shape [B, M, D]
      relative_positions: Optional relative position indication, of shape [B, T,
        T], taking values in the interval [-T, T+M], indicating the relative
        position of query[b, i] vs. key[b, j] in relative_positions[b, i, j]. In
        a usual TXL, this is i - j.
      is_training: Whether to apply dropout

    Returns:
      An array of shape [B, T, D] the result of applying self-attention to
      inputs, unless apply_final_linear is False, in which case the returned
      value is an array of shape [B, T, H, V].
    """
    batch_size, seq_len, embedding_size = inputs.shape

    queries = inputs
    if memory is None:
      values = inputs
    else:
      values = jnp.concatenate([memory, inputs], axis=1)

    query_heads = self._multihead_linear(queries, self._key_size, "query")
    # query_heads has shape [B, T, H, d_keys]
    key_heads = self._multihead_linear(values, self._key_size, "key")
    # key_heads has shape [B, T + M, H, d_keys]
    value_heads = self._multihead_linear(values, self._value_size, "value")
    # value_heads has shape [B, T + M, H, d_value]

    if positional_encodings is not None:
      logits = self._relative_position_embeddings(
          query_heads,
          key_heads,
          positional_encodings,
          relative_positions,
          is_training,
      )
    else:
      logits = jnp.einsum("bthd,bThd->bhtT", query_heads, key_heads)

    scaled_logits = logits * self._key_size ** (-0.5)
    # Mask logits by subtracting a large number (1e30). These become 0 after
    # the exponentiation in the softmax.
    assert attention_mask.dtype == scaled_logits.dtype
    masked_logits = scaled_logits - (1 - attention_mask[:, None, :, :]) * 1e30
    assert masked_logits.dtype == scaled_logits.dtype
    weights = jax.nn.softmax(masked_logits)

    if is_training:
      weights = hk.dropout(hk.next_rng_key(), self._dropout_rate, weights)

    attn_vec = jnp.einsum("bhtT,bThd->bthd", weights, value_heads)
    if self._apply_final_linear:
      attn_vec = jnp.reshape(
          attn_vec,
          [batch_size, seq_len, self._num_heads * self._value_size],
      )

      final_linear = hk.Linear(
          embedding_size,
          w_init=hk.initializers.VarianceScaling(scale=self._final_init_scale),
          with_bias=self._use_final_bias,
      )
      outputs = final_linear(attn_vec)
    else:
      outputs = attn_vec  # [B, T, H, V]

    return outputs

  @hk.transparent
  def _relative_position_embeddings(
      self,
      query_heads: jnp.ndarray,
      key_heads: jnp.ndarray,
      positional_encodings: jnp.ndarray,
      relative_positions: Optional[jnp.ndarray],
      is_training: bool,
  ) -> jnp.ndarray:
    """Compute attention using the given encodings."""
    r_w_bias = hk.get_parameter(
        "r_w_bias",
        [self._num_heads * self._key_size],
        init=hk.initializers.VarianceScaling(),
    )
    r_w_bias = jnp.reshape(r_w_bias, [self._num_heads, self._key_size])

    content_logits = jnp.einsum(
        "bthd,bThd->bhtT",
        query_heads + r_w_bias,
        key_heads,
    )

    batch_size = query_heads.shape[0]
    if is_training:
      positional_encodings = broadcast_batch(positional_encodings, batch_size)
      positional_encodings = hk.dropout(
          hk.next_rng_key(), self._dropout_rate, positional_encodings
      )
    relative_keys = self._multihead_linear(
        positional_encodings, self._key_size, "relative_keys"
    )
    # relative_keys has shape [B, R, H, d_keys]

    # Since we didn't do this before.
    if not is_training:
      relative_keys = broadcast_batch(relative_keys, batch_size)

    r_r_bias = hk.get_parameter(
        "r_r_bias",
        [self._num_heads * self._key_size],
        init=hk.initializers.VarianceScaling(),
    )  # i.e. v in TXL paper
    r_r_bias = jnp.reshape(r_r_bias, [self._num_heads, self._key_size])

    # Reminder: query_heads has shape [B, T, H, d_keys]
    relative_logits = jnp.einsum(
        "bthd,bThd->bhtT", query_heads + r_r_bias, relative_keys
    )
    # relative_logits has shape [B, H, T, R]
    if relative_positions is None:
      relative_logits = relative_shift(
          relative_logits, attention_length=key_heads.shape[1]
      )
    else:
      assert self._max_relative_position is not None
      assert self._min_relative_position is not None
      relative_positions = jnp.clip(
          relative_positions,
          self._min_relative_position,
          self._max_relative_position,
      )
      # Here, instead of doing the relative shift, which is justified because
      # when we go one token to the right in the queries, the keys all become
      # further away by one position, we need to do a gather to pick, given the
      # relative positions matrix, the right relative logits.
      relative_positions_ = (
          self._max_relative_position - relative_positions
      ).astype(jnp.int32)
      # We index in TPU friendly way:
      relative_positions_one_hot = jax.nn.one_hot(
          relative_positions_, num_classes=relative_logits.shape[-1]
      )
      relative_logits = jnp.einsum(
          "bhtT,btsT->bhts", relative_logits, relative_positions_one_hot
      )
      # Instead of doing nested vmaps:
      # def h(x, idx):
      #   return x[idx]
      # def g(x, idx):
      #   return jax.vmap(h, (0, None), 0)(x, idx)
      # def f(x, idx):
      #   return jax.vmap(g, (1, 0), 1)(x, idx)
      # relative_logits = jax.vmap(f, (0, 0), 0)(
      #     relative_logits, relative_positions_)
    assert content_logits.shape == relative_logits.shape
    return content_logits + relative_logits


class DenseBlock(hk.Module):
  """Dense block."""

  def __init__(
      self,
      *,
      ffw_hidden_size: int,
      dropout_rate: float,
      init_scale: float,
      final_init_scale_multiplier: float,
      use_final_bias: bool,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._ffw_hidden_size = ffw_hidden_size
    self._dropout_rate = dropout_rate
    self._init_scale = init_scale
    self._final_init_scale = init_scale * final_init_scale_multiplier
    self._use_final_bias = use_final_bias
    self._activation = activation

  def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    d_model = x.shape[-1]
    x = hk.Linear(
        self._ffw_hidden_size,
        w_init=hk.initializers.VarianceScaling(self._init_scale),
    )(x)

    x = self._activation(x)
    if is_training:
      x = hk.dropout(hk.next_rng_key(), self._dropout_rate, x)
    return hk.Linear(
        d_model,
        w_init=hk.initializers.VarianceScaling(self._final_init_scale),
        with_bias=self._use_final_bias,
    )(x)


def _suffixes(n):
  if n == 1:
    yield (0, "")
  else:
    for i in range(n):
      yield (i, str(i))


def _make_block_head_hybrid(
    *,
    layer: int,
    mha_kwargs: Mapping[str, Any],
    ffw_kwargs: Mapping[str, Any],
    dropout_rate: float,
    num_heads: int,
    num_unrestricted_heads: int,
    embedding_size: int,
    num_attns: int = 1,
):
  """Generalized Transformer-XL block, with restricted/unrestricted heads."""
  num_restricted_heads = num_heads - num_unrestricted_heads
  restricted_attns = [
      MultiHeadAttention(
          name=f"h{layer}_attn{suffix}",
          num_heads=num_restricted_heads,
          apply_final_linear=False,
          **mha_kwargs,
      )
      for i, suffix in _suffixes(num_attns)
  ]
  unrestricted_attn = MultiHeadAttention(
      name=f"h{layer}_unrestricted_attn",
      num_heads=num_unrestricted_heads,
      apply_final_linear=False,
      **mha_kwargs,
  )
  # pylint: disable=protected-access
  post_attn_linears = [
      hk.Linear(
          embedding_size,
          w_init=hk.initializers.VarianceScaling(
              scale=unrestricted_attn._final_init_scale
          ),
          with_bias=unrestricted_attn._use_final_bias,
          name=f"h{layer}_attn{suffix}_linear",
      )
      for i, suffix in _suffixes(num_attns)
  ]
  # pylint: enable=protected-access

  dense_block = DenseBlock(name=f"h{layer}_mlp", **ffw_kwargs)
  ln1 = layer_norm(name=f"h{layer}_ln1")
  ln2 = layer_norm(name=f"h{layer}_ln2")

  def f(
      *,
      is_training: bool,
      inputs: jnp.ndarray,
      attention_mask: jnp.ndarray,
      unrestricted_attention_mask: jnp.ndarray,
      positional_encodings: jnp.ndarray,
      txl_positional_encodings: jnp.ndarray,
      memory: jnp.ndarray,
      relative_positions: Optional[jnp.ndarray],
      attn_indicator: Optional[jnp.ndarray],
  ):
    if attn_indicator is None:
      assert num_attns == 1
    batch_size, seq_len, _ = inputs.shape
    attn_vecs = []
    if num_restricted_heads > 0:
      restricted_attn_vec = switch(
          restricted_attns,
          attn_indicator,
          inputs=inputs,
          attention_mask=attention_mask,
          positional_encodings=positional_encodings,
          memory=memory,
          is_training=is_training,
          relative_positions=relative_positions,
      )
      attn_vecs.append(restricted_attn_vec)
    if num_unrestricted_heads > 0:
      unrestricted_attn_vec = unrestricted_attn(
          inputs=inputs,
          attention_mask=unrestricted_attention_mask,
          positional_encodings=txl_positional_encodings,
          memory=memory,
          relative_positions=None,
          is_training=is_training,
      )
      attn_vecs.append(unrestricted_attn_vec)
    attn_vec = jnp.concatenate(attn_vecs, axis=2)
    # pylint: disable=protected-access
    attn_vec = jnp.reshape(
        attn_vec,
        [batch_size, seq_len, num_heads * unrestricted_attn._value_size],
    )
    # pylint: enable=protected-access
    h_attention = switch(post_attn_linears, attn_indicator, attn_vec)
    if is_training:
      h_attention = hk.dropout(hk.next_rng_key(), dropout_rate, h_attention)

    h = ln1(inputs + h_attention)
    h_ffw = dense_block(h, is_training)
    if is_training:
      h_ffw = hk.dropout(hk.next_rng_key(), dropout_rate, h_ffw)

    h = ln2(h + h_ffw)
    return h

  return f


def _make_block(
    *,
    layer: int,
    mha_kwargs: Mapping[str, Any],
    ffw_kwargs: Mapping[str, Any],
    dropout_rate: float,
    num_heads: int,
    embedding_size: int,
    num_attns: int = 1,
):
  """Generalized Transformer-XL block."""
  del embedding_size
  attns = [
      MultiHeadAttention(
          name=f"h{layer}_attn{suffix}", num_heads=num_heads, **mha_kwargs
      )
      for i, suffix in _suffixes(num_attns)
  ]
  dense_block = DenseBlock(name=f"h{layer}_mlp", **ffw_kwargs)
  ln1 = layer_norm(name=f"h{layer}_ln1")
  ln2 = layer_norm(name=f"h{layer}_ln2")

  def f(
      *,
      is_training: bool,
      inputs: jnp.ndarray,
      attention_mask: jnp.ndarray,
      unrestricted_attention_mask: jnp.ndarray,
      positional_encodings: jnp.ndarray,
      txl_positional_encodings: jnp.ndarray,
      memory: jnp.ndarray,
      relative_positions: Optional[jnp.ndarray],
      attn_indicator: Optional[jnp.ndarray],
  ):
    # Delete unused, received for compatibility with head-hybrid.
    del unrestricted_attention_mask, txl_positional_encodings
    if attn_indicator is None:
      assert num_attns == 1
    h_attention = switch(
        attns,
        attn_indicator,
        inputs=inputs,
        attention_mask=attention_mask,
        positional_encodings=positional_encodings,
        memory=memory,
        is_training=is_training,
        relative_positions=relative_positions,
    )
    if is_training:
      h_attention = hk.dropout(hk.next_rng_key(), dropout_rate, h_attention)

    h = ln1(inputs + h_attention)
    h_ffw = dense_block(h, is_training)
    if is_training:
      h_ffw = hk.dropout(hk.next_rng_key(), dropout_rate, h_ffw)

    h = ln2(h + h_ffw)
    return h

  return f


def _extract_for_layer(x, layer_idx, unrestricted_layer, y=None):
  if unrestricted_layer:
    return y
  if not isinstance(x, tuple):
    return x
  idx = layer_idx % len(x)
  return x[idx]


def _sinusoid_position_encoding(
    max_value: int,
    min_value: int,
    hidden_size: int,
    max_timescale: float = 1e4,
    min_timescale: float = 2.0,
):
  """Creates sinusoidal encodings.

  The time dimension is larger than sequence_length as we need to cover all
  cases of looking in either the future or past.

  Args:
    max_value: `int` max position M (appearing in the first row of the output)
    min_value: `int` min position m (appearing in the last row of the output)
    hidden_size: `int` dimension of the positional encoding vectors, D
    max_timescale: `int` maximum timescale for the frequency
    min_timescale: `int` minimum timescale for the frequency

  Returns:
    An array of shape [M - m, D]
  """
  assert min_value <= max_value
  freqs = np.arange(0, hidden_size, min_timescale)
  inv_freq = max_timescale ** (-freqs / hidden_size)
  # Since inputs can look into the past and into the future, depending on the
  # permutation mask, we need to have relative encodings for both. The furthest
  # back an input can see is the final token, up to sequence_length +
  # memory_length - 1. The furthest ahead an input can see is for token 0 where
  # it can see up to sequence_length - 1 future tokens.
  pos_seq = np.arange(max_value, min_value, -1.0)
  sinusoid_inp = np.einsum("i,j->ij", pos_seq, inv_freq)
  pos_emb = np.concatenate(
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
  )
  return pos_emb


class Core(hk.Module):
  """Generalized Transformer-XL-based core."""

  def __init__(
      self,
      d_model: int,
      num_layers: int,
      num_heads: int,
      key_size: int,
      value_size: int,
      ffw_hidden_size: int,
      dropout_rate: float,
      memory_length: int,
      relative_position_embeddings: bool = True,
      use_attn_bias: bool = False,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
      tied_layer_weights: bool = False,
      num_attns: int = 1,
      num_unrestricted_layers: int = 0,
      min_relative_position: Optional[int] = None,
      max_relative_position: Optional[int] = None,
      num_unrestricted_heads: Optional[int] = None,
      name: str = "core",
  ):
    """Initialises the module.

    Args:
      d_model: Size of the embeddings.
      num_layers: Number of transformer block layers.
      num_heads: Number of attention heads to use.
      key_size: Size of key (and query) embedding for attention.
      value_size: Size of value embedding for attention.
      ffw_hidden_size: Hidden size for MLP that follows attention.
      dropout_rate: How much dropout to apply to attention and MLP modules.
      memory_length: How many tokens to hold in memory.
      relative_position_embeddings: Whether to use relative position embeddings.
      use_attn_bias: Whether or not to use biases in attention linear layers.
      activation: The nonlinearity to use in the DenseBlocks.
      tied_layer_weights: If True, all the layers share the same weights.
      num_attns: Number of attention functions. 1 by default, can be 2 to use
        different weights for stack and compose attention.
      num_unrestricted_layers: Number of regular TXL (no Transformer Grammar
        tweak) layers (0 means no regular TXL layer, n > 0 means n at the top of
        the stack, n < 0 means -n at the bottom of the stack)
      min_relative_position: (TG only) Minimum value for the relative positions
        that can be passed to the model.
      max_relative_position: (TG only) Maximum value for the relative positions
        that can be passed to the model.
      num_unrestricted_heads: (TG only) For TG layers, number of unrestricted
        (TXL) heads.
      name: The Haiku name of the module.
    """
    super().__init__(name=name)
    if tied_layer_weights and num_unrestricted_layers:
      raise ValueError(
          "tied_layer_weights and num_unrestricted_layers are incompatible."
      )
    if (
        num_unrestricted_heads is not None
    ) and not 0 <= num_unrestricted_heads <= num_heads:
      raise ValueError(
          f"The number of unrestricted heads must be less than the "
          f"number of heads: {num_unrestricted_heads} vs. {num_heads}."
      )
    self._d_model = d_model
    self._num_layers = num_layers
    self._dropout_rate = dropout_rate
    self._memory_length = memory_length
    self._relative_position_embeddings = relative_position_embeddings
    self._tied_layer_weights = tied_layer_weights
    self._num_attns = num_attns
    self._num_unrestricted_layers = num_unrestricted_layers
    self._min_relative_position = min_relative_position
    self._max_relative_position = max_relative_position
    self._num_heads = num_heads
    self._num_unrestricted_heads = num_unrestricted_heads
    self._mha_kwargs = dict(
        value_size=value_size,
        key_size=key_size,
        init_scale=2.0 / self._num_layers,
        dropout_rate=self._dropout_rate,
        use_bias=use_attn_bias,
        use_final_bias=True,
        final_init_scale_multiplier=1.0,
    )
    self._ffw_kwargs = dict(
        ffw_hidden_size=ffw_hidden_size,
        dropout_rate=self._dropout_rate,
        init_scale=2.0 / self._num_layers,
        final_init_scale_multiplier=1.0,
        use_final_bias=True,
        activation=activation,
    )

  def __call__(
      self,
      input_embeddings: jnp.ndarray,
      input_mask: jnp.ndarray,
      memory: Optional[TransformerMemory] = None,
      memory_mask: Optional[jnp.ndarray] = None,
      extra_attention_mask: Optional[jnp.ndarray] = None,
      extra_memory_attention_mask: Optional[jnp.ndarray] = None,
      relative_positions: Optional[jnp.ndarray] = None,
      attn_indicator: Optional[jnp.ndarray] = None,
      smartmem_mem_from_seq: Optional[jnp.ndarray] = None,
      smartmem_mem_from_mem: Optional[jnp.ndarray] = None,
      is_training: bool = True,
  ) -> Tuple[jnp.ndarray, Optional[TransformerMemory], jnp.ndarray]:
    """Computes the logits and next memory.

    Args:
      input_embeddings: array of shape [B, T, d_model]
      input_mask: Padding mask of shape [B, T].
      memory: Optional memory of shape [N_layers, B, M, d_model]
      memory_mask: Optional memory mask of shape [B, T, M].
      extra_attention_mask: Optional attention mask of shape [B, T, T]. This
        mask will be used in addition to the input_mask and the causal_mask.
      extra_memory_attention_mask: Optional attention mask of shape [B, T, M].
        This mask will be used in addition to the memory_mask.
      relative_positions: Optional relative position indication, of shape [B, T,
        T+M], taking values in the interval [-T, T+M], indicating the relative
        position of query[b, i] vs. key[b, j] in relative_positions[b, i, j],
        with key computed from the concatenated [memory, inputs]. In a usual
        TXL, this is i - j + M.
      attn_indicator: Optional attention function indicator, i.e. which
        attention function to use for each position. Shape [B, T].
      smartmem_mem_from_seq: Smart memory -- indicator array of shape [B, T, M]
        such that embeddings at index i in the current sequence should be put in
        position j in the memory for the next chunk.
      smartmem_mem_from_mem: Smart memory -- indicator array of shape [B, M, M]
        such that embeddings at index i in the memory should be put in position
        j in the memory for the next chunk.
      is_training: Whether to use dropout.

    Returns:
      A tuple containing:
        - The final layer embeddings
        - The new memory if `memory` is given else the constant 0
    """
    assert len(input_embeddings.shape) == 3
    assert self._num_attns == 1 or attn_indicator is not None

    batch_size = input_embeddings.shape[0]
    seq_len = input_embeddings.shape[1]
    memory_and_seq_len = seq_len + self._memory_length
    expected_shape = (batch_size, seq_len, memory_and_seq_len)
    if relative_positions is not None and (
        relative_positions.shape != expected_shape
    ):
      raise ValueError(
          "Invalid input shape for relative_positions: "
          f"{relative_positions.shape!r} vs {expected_shape!r} expected."
      )
    if (smartmem_mem_from_mem is None) != (smartmem_mem_from_seq is None):
      raise ValueError(
          "smartmem_mem_from_mem and smartmem_mem_from_seq must be"
          " either both None, or none of them should be None."
      )
    use_smart_memory = smartmem_mem_from_seq is not None

    h = input_embeddings
    if is_training:
      h = hk.dropout(hk.next_rng_key(), self._dropout_rate, h)

    # Generate positional encodings.
    _, seq_len, embedding_size = input_embeddings.shape
    if not self._relative_position_embeddings:
      positional_encodings = None
      txl_positional_encodings = None
      mha_kwargs = self._mha_kwargs
    else:
      if self._max_relative_position is not None:
        max_relative_position = self._max_relative_position
      else:
        max_relative_position = seq_len + self._memory_length
      if self._min_relative_position is not None:
        min_relative_position = self._min_relative_position - 1
      else:
        min_relative_position = -seq_len
      positional_encodings = _sinusoid_position_encoding(
          max_value=max_relative_position,
          min_value=min_relative_position,
          hidden_size=embedding_size,
      )
      positional_encodings = positional_encodings.astype(input_embeddings.dtype)
      # Ensure that the baked-in constant is size (2 * seq_len, seq_len).
      positional_encodings = lax.tie_in(input_embeddings, positional_encodings)

      # TXL-style positional encodings
      txl_positional_encodings = _sinusoid_position_encoding(
          max_value=seq_len + self._memory_length,
          min_value=-seq_len,
          hidden_size=embedding_size,
      )
      txl_positional_encodings = txl_positional_encodings.astype(
          input_embeddings.dtype
      )
      txl_positional_encodings = lax.tie_in(
          input_embeddings, txl_positional_encodings
      )

      mha_kwargs = self._mha_kwargs.copy()
      mha_kwargs["max_relative_position"] = max_relative_position
      mha_kwargs["min_relative_position"] = min_relative_position

    if self._num_unrestricted_heads is not None:
      block_fn = _make_block_head_hybrid
      block_kwargs = dict(num_unrestricted_heads=self._num_unrestricted_heads)
    else:
      block_fn = _make_block
      block_kwargs = dict()

    if self._tied_layer_weights:
      assert self._num_unrestricted_layers == 0
      # Parameterize function on options.
      block = block_fn(
          layer=0,
          mha_kwargs=mha_kwargs,
          ffw_kwargs=self._ffw_kwargs,
          dropout_rate=self._dropout_rate,
          num_heads=self._num_heads,
          num_attns=self._num_attns,
          embedding_size=embedding_size,
          **block_kwargs,
      )
      blocks = [(block, False)] * self._num_layers
    else:
      blocks = []
      for i in range(self._num_layers):
        if self._num_unrestricted_layers > 0 and (
            i >= self._num_layers - self._num_unrestricted_layers
        ):
          unrestricted_layer = True
        elif self._num_unrestricted_layers < 0 and (
            i < -self._num_unrestricted_layers
        ):
          unrestricted_layer = True
        else:
          unrestricted_layer = False

        if unrestricted_layer:
          actual_block_fn = _make_block
          actual_block_kwargs = dict()
        else:
          actual_block_fn, actual_block_kwargs = (
              block_fn,
              block_kwargs,
          )

        blocks.append((
            # Parameterize function on options.
            actual_block_fn(
                layer=i,
                # We don't need to specify special MHA args in unrestricted
                # mode, because the min/max relative positions will be ignored
                # in that case.
                mha_kwargs=mha_kwargs,
                ffw_kwargs=self._ffw_kwargs,
                dropout_rate=self._dropout_rate,
                num_heads=self._num_heads,
                num_attns=self._num_attns if not unrestricted_layer else 1,
                embedding_size=embedding_size,
                **actual_block_kwargs,
            ),
            unrestricted_layer,
        ))

    new_memory = None if memory is None else []
    layers_outputs = [h]
    for i, (block, unrestricted_layer) in enumerate(blocks):
      # Add embeddings to memory before we go through the layer
      if new_memory is not None:
        if not use_smart_memory:
          new_mem = jnp.concatenate((memory[i], h), axis=1)
          new_mem = lax.slice(
              new_mem,
              start_indices=(
                  0,
                  new_mem.shape[1] - self._memory_length,
                  0,
              ),
              limit_indices=new_mem.shape,
          )
        else:
          # memory has shape [N_layers, B, M, d_model]
          old_mem = memory[i]  #  [B, M, d_model]
          new_mem_from_mem = jnp.einsum(
              "bmd,bmn->bnd", old_mem, smartmem_mem_from_mem
          )
          new_mem_from_seq = jnp.einsum(
              "btd,btn->bnd", h, smartmem_mem_from_seq
          )
          new_mem = new_mem_from_mem + new_mem_from_seq
        new_memory.append(new_mem)

      memory_i = memory[i] if memory is not None else None

      # Generate attention mask.
      attention_mask, unrestricted_attention_mask = make_attention_mask(
          input_mask=input_mask,
          memory_mask=memory_mask,
          extra_attention_mask=_extract_for_layer(
              extra_attention_mask, i, unrestricted_layer
          ),
          extra_memory_attention_mask=_extract_for_layer(
              extra_memory_attention_mask, i, unrestricted_layer
          ),
          memory_len=self._memory_length,
          dtype=input_embeddings.dtype,
      )

      h = block(  # pylint: disable=missing-kwoa
          is_training=is_training,
          inputs=h,
          attention_mask=attention_mask,
          unrestricted_attention_mask=unrestricted_attention_mask,
          positional_encodings=_extract_for_layer(
              positional_encodings,
              i,
              unrestricted_layer,
              y=txl_positional_encodings,
          ),
          txl_positional_encodings=txl_positional_encodings,
          memory=memory_i,
          relative_positions=_extract_for_layer(
              relative_positions, i, unrestricted_layer
          ),
          attn_indicator=attn_indicator if not unrestricted_layer else None,
      )
      layers_outputs.append(h)

    if new_memory is not None:
      new_memory = jnp.stack(new_memory)

    return (
        h,  # [B, T, H]
        new_memory,
        jnp.stack(layers_outputs, axis=2),  # [B, T, L, H]
    )

  def initial_memory(
      self,
      batch_size: int,
      dtype=jnp.float32,
  ) -> Optional[TransformerMemory]:
    """Creates the initial memory array, filled with 0s."""
    if not self._memory_length:
      return
    memory_shape = (
        self._num_layers,
        batch_size,
        self._memory_length,
        self._d_model,
    )
    return jnp.zeros(shape=memory_shape, dtype=dtype)
