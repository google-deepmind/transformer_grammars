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

"""Tests for transformer_grammars.models.core."""

import functools
import operator
import unittest
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from transformer_grammars.models import core
import tree

TEST_MODEL_PARAMS = dict(
    d_model=32,
    num_layers=2,
    num_heads=4,
    key_size=8,
    value_size=8,
    ffw_hidden_size=128,
    dropout_rate=0.1,
)
SMALL_MODEL_PARAMS = dict(TEST_MODEL_PARAMS, num_layers=1)


def _product(l):
  return functools.reduce(operator.mul, l, 1)


def get_input_data(key, batch_size, seq_len, mem_len, d_model, num_layers):
  key, key_ = jax.random.split(key)
  input_embeddings = jax.random.normal(
      key_, shape=(batch_size, seq_len, d_model), dtype=jnp.float32
  )

  key, key_ = jax.random.split(key)

  ## Only the first 6 inputs are valid
  input_mask = jnp.array([[1] * 6 + [0] * 4] * batch_size)

  key, key_ = jax.random.split(key)

  memory_keys = jax.random.split(key, num=num_layers)
  memory = jnp.stack(
      [
          jax.random.normal(memory_keys[i], (batch_size, mem_len, d_model))
          for i in range(num_layers)
      ]
  )

  return dict(
      input_embeddings=input_embeddings, input_mask=input_mask, memory=memory
  )


class CoreTest(unittest.TestCase):

  def test_rel_shift(self):
    logits = jnp.array([
        [-3, -2, -1, 0, 1, 2],
        [-3, -2, -1, 0, 1, 2],
        [-3, -2, -1, 0, 1, 2],
    ])

    logits = jnp.broadcast_to(logits, (4, 8) + logits.shape)
    shifted_logits = core.relative_shift(logits, attention_length=3)

    expected_output = jnp.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    expected_output = jnp.broadcast_to(
        expected_output, (4, 8) + expected_output.shape
    )

    np.testing.assert_array_equal(shifted_logits, expected_output)

  def test_output_and_memory_shape(self):
    key = jax.random.PRNGKey(42)

    batch_size = 2
    seq_len = 10
    mem_len = 5
    d_model = TEST_MODEL_PARAMS["d_model"]
    num_layers = TEST_MODEL_PARAMS["num_layers"]

    model_inputs = get_input_data(
        key, batch_size, seq_len, mem_len, d_model, num_layers
    )

    def forward(model_inputs, is_training=True):
      return core.Core(memory_length=mem_len, **TEST_MODEL_PARAMS)(
          is_training=is_training, **model_inputs
      )

    init_fn, apply_fn = hk.transform(forward)

    key, key_ = jax.random.split(key)
    params = init_fn(key_, model_inputs)

    key, key_ = jax.random.split(key)
    outputs, next_memory, _ = apply_fn(params, key_, model_inputs)

    self.assertEqual(outputs.shape, (batch_size, seq_len, d_model))
    self.assertEqual(len(next_memory), num_layers)
    for i in range(num_layers):
      self.assertEqual(next_memory[i].shape, (batch_size, mem_len, d_model))

  def test_masked_inputs_are_unused(self):
    """Output position i depends exactly on i and non-masked inputs <= i."""
    seq_len = 10
    mem_len = 5

    key_ = jax.random.PRNGKey(42)
    key, key_ = jax.random.split(key_)
    data = get_input_data(
        key=key,
        batch_size=1,
        seq_len=seq_len,
        mem_len=mem_len,
        d_model=SMALL_MODEL_PARAMS["d_model"],
        num_layers=SMALL_MODEL_PARAMS["num_layers"],
    )

    def logits(input_embeddings):
      model = core.Core(memory_length=mem_len, **SMALL_MODEL_PARAMS)
      logits, _, _ = model(
          input_embeddings=input_embeddings,
          input_mask=data["input_mask"],
          memory=data["memory"],
          is_training=False,
      )
      # Use batch 0 only
      return logits[0]

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(logits))

    key, key_ = jax.random.split(key_)
    params = init_fn(key, data["input_embeddings"])

    inputs_jacobian_fn = jax.jacrev(lambda inputs: apply_fn(params, inputs))

    # Jacobian has shape [T, D, B, T, D]
    inputs_jacobian = inputs_jacobian_fn(data["input_embeddings"])
    # Use input batch 0 only
    inputs_jacobian = inputs_jacobian[:, :, 0]
    # Sum over input channnel dimension
    inputs_jacobian = jnp.sum(inputs_jacobian, axis=-1)
    # Take max gradient over output channel dimension
    inputs_jacobian = jnp.max(jnp.abs(inputs_jacobian), axis=1)

    used_inputs = inputs_jacobian != 0.0

    allowed_inputs = jnp.logical_and(
        data["input_mask"], jnp.tril(jnp.ones((seq_len, seq_len)))
    )
    # Each masked input can see itself due to residual connections
    allowed_inputs = jnp.logical_or(allowed_inputs, jnp.eye(seq_len))
    np.testing.assert_array_equal(used_inputs, allowed_inputs)

  def test_memory_mask(self):
    """Tests that masked memory doesn't change the output logits."""
    key = jax.random.PRNGKey(42)

    batch_size = 1
    seq_len = 10
    mem_len = 5
    d_model = TEST_MODEL_PARAMS["d_model"]
    num_layers = TEST_MODEL_PARAMS["num_layers"]

    model_inputs = get_input_data(
        key, batch_size, seq_len, mem_len, d_model, num_layers
    )
    input_embeddings = model_inputs["input_embeddings"]
    initial_memory = model_inputs["memory"]
    input_mask = np.ones((batch_size, seq_len))  # all inputs are valid

    key, key_ = jax.random.split(key)

    def forward_small_mem(input_embeddings, input_mask, memory, memory_mask):
      model = core.Core(memory_length=mem_len, **TEST_MODEL_PARAMS)
      return model(
          input_embeddings,
          input_mask,
          memory,
          memory_mask,
          is_training=False,
      )

    init_fn, apply_small_fn = hk.without_apply_rng(
        hk.transform(forward_small_mem)
    )

    params = init_fn(
        key_, input_embeddings, input_mask, initial_memory, memory_mask=None
    )

    chunk_small_outputs, _, _ = apply_small_fn(
        params, input_embeddings, input_mask, initial_memory, None
    )

    def forward_large_mem(input_embeddings, input_mask, memory, memory_mask):
      model = core.Core(memory_length=mem_len * 2, **TEST_MODEL_PARAMS)
      return model(
          input_embeddings,
          input_mask,
          memory,
          memory_mask,
          is_training=False,
      )

    _, apply_large_fn = hk.without_apply_rng(hk.transform(forward_large_mem))

    # Memory is twice as large, but we only attend to the second half.
    # Outputs should be the same if the memory mask works as expected.
    large_initial_memory = jnp.concatenate(
        (initial_memory, initial_memory), axis=2
    )
    large_memory_mask = jnp.arange(mem_len * 2) >= mem_len
    large_memory_mask = jnp.broadcast_to(
        large_memory_mask, (batch_size, seq_len, mem_len * 2)
    )
    chunk_large_outputs, _, _ = apply_large_fn(
        params,
        input_embeddings,
        input_mask,
        large_initial_memory,
        large_memory_mask,
    )

    np.testing.assert_array_almost_equal(
        chunk_small_outputs, chunk_large_outputs, decimal=5
    )

  def test_memory_shifts_over_multiple_steps(self):
    key = jax.random.PRNGKey(42)
    batch_size = 2
    seq_len = 5
    mem_len_factor = 4
    mem_len = seq_len * mem_len_factor
    d_model = TEST_MODEL_PARAMS["d_model"]
    num_layers = TEST_MODEL_PARAMS["num_layers"]

    def forward(input_embeddings, input_mask, memory):
      model = core.Core(memory_length=mem_len, **TEST_MODEL_PARAMS)
      return model(input_embeddings, input_mask, memory, is_training=False)

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(forward))
    apply_fn = jax.jit(apply_fn)

    key, key_ = jax.random.split(key)
    model_inputs = get_input_data(
        key_, batch_size, seq_len, mem_len, d_model, num_layers
    )

    initial_memory = model_inputs["memory"]
    input_mask = jnp.ones((batch_size, seq_len))  # all inputs are valid

    key, key_ = jax.random.split(key)
    params = init_fn(
        key_, model_inputs["input_embeddings"], input_mask, initial_memory
    )

    # Compute outputs by feeding one token at a time.
    memory = initial_memory
    for i in range(1, mem_len_factor):
      key, key_ = jax.random.split(key)
      input_embeddings = jax.random.normal(
          key_, shape=(batch_size, seq_len, d_model), dtype=jnp.float32
      )
      _, new_memory, _ = apply_fn(params, input_embeddings, input_mask, memory)
      memory = new_memory

      # Memory is shifted i times by seq_len after i steps.
      np.testing.assert_array_equal(
          initial_memory[:, :, -seq_len:],
          memory[:, :, -(i + 1) * seq_len : -i * seq_len],
      )

  def test_rel_shift_and_explicit_relative_position_give_same_result(self):
    key = jax.random.PRNGKey(42)

    batch_size = 2
    seq_len = 10
    mem_len = 5
    d_model = TEST_MODEL_PARAMS["d_model"]
    num_layers = TEST_MODEL_PARAMS["num_layers"]

    model_inputs = get_input_data(
        key, batch_size, seq_len, mem_len, d_model, num_layers
    )

    def forward(model_inputs, is_training=True):
      return core.Core(memory_length=mem_len, **TEST_MODEL_PARAMS)(
          is_training=is_training, **model_inputs
      )

    def forward_with_relative_positions(
        model_inputs, relative_positions, is_training=True
    ):
      return core.Core(memory_length=mem_len, **TEST_MODEL_PARAMS)(
          is_training=is_training,
          relative_positions=relative_positions,
          **model_inputs,
      )

    init_fn, apply_fn = hk.transform(forward)
    _, apply_with_relative_pos_fn = hk.transform(
        forward_with_relative_positions
    )

    key, key_ = jax.random.split(key)
    params = init_fn(key_, model_inputs)

    _, key_ = jax.random.split(key)
    outputs, next_memory, _ = apply_fn(params, key_, model_inputs)

    relative_positions = (
        mem_len
        + np.arange(seq_len).reshape((-1, 1))
        - np.arange(seq_len + mem_len).reshape((1, -1))
    )
    # relative_positions[i, j] = i - j + mem_len
    # i.e. how many tokens ahead query i compared to key j
    relative_positions = np.broadcast_to(
        relative_positions, (batch_size, seq_len, seq_len + mem_len)
    )
    outputs2, next_memory2, _ = apply_with_relative_pos_fn(
        params, key_, model_inputs, relative_positions
    )

    np.testing.assert_array_equal(outputs, outputs2)
    np.testing.assert_array_equal(next_memory, next_memory2)

  def test_shift_and_smartmem_indicator_give_same_result(self):
    key = jax.random.PRNGKey(42)

    batch_size = 2
    seq_len = 10
    mem_len = 5
    d_model = TEST_MODEL_PARAMS["d_model"]
    num_layers = TEST_MODEL_PARAMS["num_layers"]

    model_inputs0 = get_input_data(
        key, batch_size, seq_len, mem_len, d_model, num_layers
    )
    model_inputs1 = get_input_data(
        key, batch_size, seq_len, mem_len, d_model, num_layers
    )
    del model_inputs1["memory"]

    smartmem_mem_from_mem = jnp.zeros(
        (batch_size, mem_len, mem_len), dtype=jnp.float32
    )
    smartmem_mem_from_seq = jnp.zeros(
        (batch_size, seq_len, mem_len), dtype=jnp.float32
    )
    for i in range(min(mem_len, seq_len)):
      smartmem_mem_from_seq = smartmem_mem_from_seq.at[:, -1 - i, -1 - i].set(1)

    def forward(model_inputs, is_training=True):
      return core.Core(memory_length=mem_len, **TEST_MODEL_PARAMS)(
          is_training=is_training, **model_inputs
      )

    init_fn, apply_fn = hk.transform(forward)

    key, key_ = jax.random.split(key)
    params = init_fn(key_, model_inputs0)

    _, key_ = jax.random.split(key)

    # Apply the TXL core as usual, i.e. shifting the current embeddings into the
    # memory.
    outputs0, next_memory0, _ = apply_fn(params, key_, model_inputs0)
    model_inputs1["memory"] = next_memory0
    outputs1, next_memory1, _ = apply_fn(params, key_, model_inputs1)

    model_inputs0_sm = model_inputs0.copy()
    model_inputs0_sm["smartmem_mem_from_mem"] = smartmem_mem_from_mem
    model_inputs0_sm["smartmem_mem_from_seq"] = smartmem_mem_from_seq
    outputs0_sm, next_memory0_sm, _ = apply_fn(params, key_, model_inputs0_sm)
    model_inputs1_sm = model_inputs1.copy()
    model_inputs1_sm["smartmem_mem_from_mem"] = smartmem_mem_from_mem
    model_inputs1_sm["smartmem_mem_from_seq"] = smartmem_mem_from_seq
    model_inputs1_sm["memory"] = next_memory0_sm
    outputs1_sm, next_memory1_sm, _ = apply_fn(params, key_, model_inputs1_sm)

    np.testing.assert_array_equal(outputs0, outputs0_sm)
    np.testing.assert_array_equal(outputs1, outputs1_sm)
    np.testing.assert_array_equal(next_memory0, next_memory0_sm)
    np.testing.assert_array_equal(next_memory1, next_memory1_sm)

  def test_fewer_params_with_tied_layers(self):
    key = jax.random.PRNGKey(42)

    batch_size = 2
    seq_len = 10
    d_model = TEST_MODEL_PARAMS["d_model"]
    num_layers = TEST_MODEL_PARAMS["num_layers"]

    model_inputs = get_input_data(
        key, batch_size, seq_len, 0, d_model, num_layers
    )
    del model_inputs["memory"]

    def forward(model_inputs, is_training=True):
      return core.Core(
          memory_length=0, tied_layer_weights=False, **TEST_MODEL_PARAMS
      )(is_training=is_training, **model_inputs)

    def forward_tied(model_inputs, is_training=True):
      return core.Core(
          memory_length=0, tied_layer_weights=True, **TEST_MODEL_PARAMS
      )(is_training=is_training, **model_inputs)

    init_fn, _ = hk.transform(forward)
    params = init_fn(key, model_inputs)

    init_tied_fn, _ = hk.transform(forward_tied)
    params_tied = init_tied_fn(key, model_inputs)

    num_params = sum(map(lambda x: _product(x.shape), tree.flatten(params)))
    num_params_tied = sum(
        map(lambda x: _product(x.shape), tree.flatten(params_tied))
    )
    self.assertLess(num_params_tied, num_params)

  def test_more_params_with_several_attn_functions(self):
    key = jax.random.PRNGKey(42)

    batch_size = 2
    seq_len = 10
    d_model = TEST_MODEL_PARAMS["d_model"]
    num_layers = TEST_MODEL_PARAMS["num_layers"]

    model_inputs = get_input_data(
        key, batch_size, seq_len, 0, d_model, num_layers
    )
    model_inputs["attn_indicator"] = model_inputs["input_mask"] * 0
    del model_inputs["memory"]

    def forward(model_inputs, is_training=True):
      return core.Core(memory_length=0, num_attns=1, **TEST_MODEL_PARAMS)(
          is_training=is_training, **model_inputs
      )

    def forward_multiple_attns(model_inputs, is_training=True):
      return core.Core(memory_length=0, num_attns=2, **TEST_MODEL_PARAMS)(
          is_training=is_training, **model_inputs
      )

    init_fn, _ = hk.transform(forward)
    params = init_fn(key, model_inputs)

    init_multiple_attns_fn, _ = hk.transform(forward_multiple_attns)
    params_multiple_attns = init_multiple_attns_fn(key, model_inputs)

    num_params = sum(map(lambda x: _product(x.shape), tree.flatten(params)))
    num_params_multiple_attns = sum(
        map(
            lambda x: _product(x.shape),
            tree.flatten(params_multiple_attns),
        )
    )
    self.assertGreater(num_params_multiple_attns, num_params)

  def test_hybrid_with_only_txl_layers_is_a_txl(self):
    key = jax.random.PRNGKey(42)

    batch_size = 2
    seq_len = 10
    mem_len = 5
    d_model = TEST_MODEL_PARAMS["d_model"]
    num_layers = TEST_MODEL_PARAMS["num_layers"]
    keys = hk.PRNGSequence(42)

    model_inputs = get_input_data(
        key, batch_size, seq_len, mem_len, d_model, num_layers
    )

    hybrid_model_inputs = model_inputs.copy()
    hybrid_model_inputs["extra_attention_mask"] = np.broadcast_to(
        np.eye(seq_len).reshape((1, seq_len, seq_len)),
        (batch_size, seq_len, seq_len),
    )
    hybrid_model_inputs["relative_positions"] = np.random.randint(
        -4, 4, (batch_size, seq_len, seq_len + mem_len)
    )
    hybrid_model_inputs["attn_indicator"] = model_inputs["input_mask"] * 0

    def forward_txl(model_inputs, is_training=False):
      model = core.Core(memory_length=mem_len, **TEST_MODEL_PARAMS)
      return model(is_training=is_training, **model_inputs)

    def forward_hybrid_tg_with_only_txl_layers(model_inputs, is_training=False):
      model = core.Core(
          num_unrestricted_layers=num_layers,
          memory_length=mem_len,
          num_attns=2,
          min_relative_position=-1,
          max_relative_position=1,
          **TEST_MODEL_PARAMS,
      )
      return model(is_training=is_training, **model_inputs)

    init_fn, apply_fn = hk.transform(forward_txl)
    params = init_fn(next(keys), model_inputs)

    _, apply_hybrid_fn = hk.transform(forward_hybrid_tg_with_only_txl_layers)

    key = next(keys)
    outputs, next_memory, _ = apply_fn(params, key, model_inputs)
    outputs_hybrid, next_memory_hybrid, _ = apply_hybrid_fn(
        params, key, hybrid_model_inputs
    )

    np.testing.assert_array_equal(outputs, outputs_hybrid)
    np.testing.assert_array_equal(next_memory, next_memory_hybrid)
