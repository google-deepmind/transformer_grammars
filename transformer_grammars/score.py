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

"""Score tokenized post-processed sequences.

Note: This is a naive implementation with batch size 1 for simplicity. It can be
extended to support > 1 batch sizes, or returning activations from the model.
"""

import functools
import jax
import jax.numpy as jnp
from transformer_grammars import common
from transformer_grammars.data import preprocessing
from transformer_grammars.data import text_dataset
from transformer_grammars.data import tokenizer_utils as utils
from transformer_grammars.training import checkpoint


def _sequences_iterator(fname, add_eos):
  """Iterator over the pretokenized dataset."""
  ds = text_dataset.PreEncodedTextDataset(
      filename=fname, num_samples=None, add_bos=True, add_eos=add_eos
  )
  ds = ds.raw_dataset(
      shuffle=False,
      shuffle_buffer=None,
      sample_without_replacement=False,
      num_epochs=1,
  )
  return ds.as_numpy_iterator()


@functools.partial(jax.jit, static_argnums=(0, 1))
def _call_model(forward, maskrules, params, state, chunk):
  """Calls the model for scoring purposes."""
  model_inputs = common.model_input_from_chunk(chunk, maskrules)
  logits, state = forward(params, state, rng=None, **model_inputs)
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  mask = jnp.logical_and(
      jnp.greater(chunk.labels, 0), jnp.greater(chunk.seq_idx, -1)[:, None]
  ).astype(jnp.int32)
  log_probs *= mask[:, :, None]
  labels_log_probs = jax.vmap(jax.vmap(lambda t, idx: t[idx], 0, 0), 0, 0)(
      log_probs, chunk.labels
  )
  chunk_log_prob = jnp.sum(labels_log_probs, axis=1)
  output = (log_probs, labels_log_probs, chunk_log_prob)
  # Batch size is 1, so drop the batch dimension inside the jitted call.
  output = jax.tree_util.tree_map(functools.partial(jnp.squeeze, axis=0),
                                  output)
  return output, state


def main(tokenizer, checkpoint_path, input_, add_eos, _):
  """Score."""

  # Extract value from flag handles.
  tokenizer = tokenizer.value
  checkpoint_path = checkpoint_path.value
  input_ = input_.value
  add_eos = add_eos.value

  # Get the token type ranges, i.e. which token IDs correspond to terminals,
  # to opening non-terminals, to closing non-terminals, etc.
  dic, ranges = utils.get_dictionary_and_ranges(tokenizer)

  # Load the model checkpoint.
  ckpt = checkpoint.load_checkpoint(checkpoint_path)
  model_cfg = ckpt.config
  params = ckpt.params

  # Build the appropriate masking rules object.
  maskrules = common.build_maskrules(model_cfg)

  # Build the forward function that corresponds to the model config and
  # masking rules.
  forward = common.build_forward(
      model_cfg, maskrules, ranges, is_training=False
  ).apply

  # Get an iterator over the pre-tokenized dataset. This contains unbatched
  # sequences of ints.
  sequences_it = _sequences_iterator(input_, add_eos)

  # Get an iterator over the batches (of chunks). We have batch size == 1 for
  # simplicity here. Chunks are fixed-length successive portions of the
  # sequence, plus auxiliary quantities that are computed out of the model
  # but that the model requires (i.e. what the masking rules compute).
  chunks_it = preprocessing.get_chunks_from_dataset(
      sequences_it,
      maskrules,
      ranges,
      shape_prefix=(1,),
      multithread=False,
      use_monitor_thread=False,
  )

  state = None
  seq_log_prob = 0.0
  total_log_prob = 0.0
  for chunk in chunks_it:
    (_, labels_log_probs, chunk_log_prob), state = _call_model(
        forward, maskrules, params, state, chunk
    )
    inputs = chunk.inputs[0]
    labels = chunk.labels[0]
    seq_log_prob += chunk_log_prob
    total_log_prob += chunk_log_prob
    if chunk.beginning_of_seq.item():
      print("=" * 80)
    for inp, lab, lp in zip(inputs, labels, labels_log_probs):
      if inp == 0:
        continue
      if lab != 0:
        print(f"Input: {dic[inp]}\tLabel: {dic[lab]}\tLog prob: {lp:.2f}")
      else:
        print(f"Input: {dic[inp]}\tLabel: (no prediction)")

    if chunk.end_of_seq.item():
      print(f"Sequence log probability: {seq_log_prob:.2f}")
      print("=" * 80)
      print("")
      seq_log_prob = 0.0
  print(f"Total dataset log probability: {total_log_prob:.2f}")
