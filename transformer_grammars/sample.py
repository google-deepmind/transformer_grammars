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

"""Sample from a trained model.

Whilst simple in principle, sampling is somewhat complicated with our
implementation which computes the attention mask, the relative positions, and
the memory update tensors outside of the JAX model core. This is an example of
how to do it at the cost of one full forward pass per token. Just like for
scoring, this is a naive implementation with batch size 1.

Warning: we do not verify here that the samples are correctly parenthesized
before passing them to the model. In our experience, this is not an issue with
a trained model, but might be one for very small or very undertrained ones.
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
from transformer_grammars import common
from transformer_grammars.data import preprocessing
from transformer_grammars.data import text_dataset
from transformer_grammars.data import tokenizer_utils as utils
from transformer_grammars.training import checkpoint


def _sequences_iterator(fname):
  """Iterator over the pretokenized dataset."""
  ds = text_dataset.PreEncodedTextDataset(
      filename=fname, num_samples=None, add_bos=True, add_eos=False
  )
  ds = ds.raw_dataset(
      shuffle=False,
      shuffle_buffer=None,
      sample_without_replacement=False,
      num_epochs=1,
  )
  return ds.as_numpy_iterator()


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _call_model(forward, maskrules, temperature, params, state, key, chunk):
  """Calls the model for sampling purposes."""
  model_inputs = common.model_input_from_chunk(chunk, maskrules)
  logits, state = forward(params, state, rng=None, **model_inputs)
  # Find last non-padding token, assuming that not all tokens are padding,
  # which shouldn't be the case.
  non_padding = jnp.greater(model_inputs["inputs"], 0).astype(jnp.int32)
  last_non_padding = jnp.max(
      jnp.arange(model_inputs["inputs"].shape[1]) * non_padding, axis=1
  )
  last_logits = jax.vmap(lambda x, idx: x[idx], in_axes=(0, 0))(
      logits, last_non_padding
  )
  last_logits = last_logits.at[:, :2].add(-100.0)  # Disallow PAD and BOS.
  last_logits /= temperature
  # (We could transform logits here.)
  next_key, key = jax.random.split(key)
  next_token = jax.random.categorical(key, last_logits, axis=1)
  output = (next_token,)
  # Batch size is 1, so drop the batch dimension inside the jitted call.
  output = jax.tree_util.tree_map(lambda x: x[0], output)
  return output, state, next_key


def _sample(forward, params, maskrules, ranges, dic, temperature, key, prefix,
            num_steps):
  """Generates samples from a prefix.

  We first pass the successive chunks corresponding to the prefix to the
  model, and we sample a token from the last one. We append it to the prefix,
  then call the model again, with the memory state as it was after the last
  full chunk, and correspondingly skipping the parts of the input
  corresponding to these full chunks.

  As an example, assuming a Transformer Grammars model, a sequence length of
  4, and a prefix "(S (NP the":

                +---> predicted: hungry
                |
  +----+-----+-----+-------+
  | (S | (NP | the | <pad> |
  +----+-----+-----+-------+

                       +---> predicted: cat
                       |
  +----+-----+-----+--------+
  | (S | (NP | the | hungry |
  +----+-----+-----+--------+

                                +---> predicted: NP)
                                |
  +----+-----+-----+--------++-----+
  | (S | (NP | the | hungry || cat |
  +----+-----+-----+--------++-----+

  As we are now in the second chunk, we can continue from the memory state
  after "hungry", skipping "(S (NP the hungry". (We could have saved the state
  one step before, but this requires a careful handling of the case where a
  single sampled token extends the current prefix by 2, for closing
  non-terminals. We do not do this given the minimal gains, and for
  simplicity.)

                                            +---> predicted: (VP
                                            |
                             +-----+-----+-----+
                             | cat | NP) | NP) |
                             +-----+-----+-----+

                                                  +---> predicted: meows
                                                  |
                             +-----+-----+-----+-----+
                             | cat | NP) | NP) | (VP |
                             +-----+-----+-----+-----+

                                                          +---> predicted: VP)
                                                          |
                             +-----+-----+-----+-----++-------+
                             | cat | NP) | NP) | (VP || meows |
                             +-----+-----+-----+-----++-------+

                                                      +-------+-----+-----+
                                                      | meows | VP) | VP) |
                                                      +-------+-----+-----+

  and so on.

  Args:
    forward: Forward function to call the model.
    params: Model parameters.
    maskrules: Masking rules object.
    ranges: Token type ranges.
    dic: Dictionary.
    temperature: Temperature applied on the logits for sampling.
    key: PRNG key.
    prefix: Prefix to sample from.
    num_steps: Number of sampling steps.

  Returns:
    Next RNG key.
  """
  last_idx = None
  state = None
  seq = prefix
  skip_chunks = 0
  for _ in range(num_steps):
    idx = 0
    chunks = preprocessing.get_chunks_from_dataset(
        [seq],
        maskrules,
        ranges,
        shape_prefix=(1,),
        multithread=False,
        use_monitor_thread=False,
    )
    # Keep the linter happy:
    chunk_idx = 0
    chunk = None
    for chunk_idx, chunk in enumerate(chunks):
      # Skip chunks already evaluated.
      if chunk_idx < skip_chunks:
        idx += chunk.inputs.shape[1]
        continue
      (next_token,), next_state, key = _call_model(
          forward, maskrules, temperature, params, state, key, chunk
      )
      inputs = chunk.inputs[0]
      for inp in inputs:
        if inp != 0:  # Do not print padding tokens.
          idx += 1
          if last_idx is None:
            # Print the initial prompt.
            print(f">>> {dic[inp]}")
          elif idx > last_idx:
            # And tokens that have been added to the input since
            # the previous step.
            print(f"+++ {dic[inp]}")
      # Do not update the state if this is the final chunk, because it
      # may be incomplete.
      if not chunk.end_of_seq.item():
        state = next_state
    assert chunk.end_of_seq.item()
    next_token = int(jax.device_get(next_token))
    assert next_token not in (0, 1)  # PAD and BOS should not be sampled.
    seq = np.concatenate([seq, [next_token]])
    # Keep track of the last token printed.
    last_idx = idx
    # And how many chunks are to be skipped.
    skip_chunks = chunk_idx
  return key


def main(tokenizer, checkpoint_path, input_, seed, temperature, num_steps, _):
  """Sample."""

  # Extract values from flag handles.
  tokenizer = tokenizer.value
  checkpoint_path = checkpoint_path.value
  input_ = input_.value
  seed = seed.value
  temperature = temperature.value
  num_steps = num_steps.value

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
  prefixes_it = _sequences_iterator(input_)

  # Initialize the RNG used for sampling.
  key = jax.random.PRNGKey(seed)

  for prefix in prefixes_it:
    key = _sample(
        forward,
        params,
        maskrules,
        ranges,
        dic,
        temperature,
        key,
        prefix,
        num_steps,
    )
