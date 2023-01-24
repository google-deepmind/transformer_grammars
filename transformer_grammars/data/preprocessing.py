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

"""Data preprocessing between the dataset and the C++ masking rules."""

import functools
import itertools
import operator
import queue
import threading
import time
from typing import Callable, Dict, Generator, Sequence

from absl import logging
import numpy as np
from transformer_grammars.models.masking import masking_types as types
from transformer_grammars.models.masking import utils as masking_utils
import tree

Chunk = types.Chunk


def lshift(arr):
  assert len(arr.shape) == 1
  arr = arr[1:]
  return np.pad(arr, [(0, 1)], mode="constant")


def compute_inputs_and_labels(
    inp: Dict[str, np.ndarray], use_untyped_closing_nt_for_labels: bool = False
) -> Dict[str, np.ndarray]:
  """Computes a sequence of observations and a sequence of labels.

  Args:
    inp: Input dict, containing a NumPy sequence of shape [T] associated to the
      key 'seq', and possibly other keys/values.
    use_untyped_closing_nt_for_labels: Whether typed (False) or untyped (True)
      closing non-terminals are to appear in the labels.

  Returns:
    Dict with keys 'inputs' and 'labels'.
  """
  if use_untyped_closing_nt_for_labels:
    raise NotImplementedError
  # Be careful not to mutate the input, as the same objects may belong to an
  # iterator on which itertools.tee is applied.
  inp_ = inp.copy()
  for_observation = inp_.pop("for_observation")
  for_target = inp_.pop("for_target")
  return dict(inputs=for_observation, labels=lshift(for_target), **inp_)


def pad_to_multiple(inp: np.ndarray, seqlen: int) -> np.ndarray:
  if len(inp.shape) >= 1:
    # inp has shape [T, ...]
    current_len = inp.shape[0]
    # a % b is the remainder of the Euclidean division of a by b, even if a is
    # negative, so it always positive.
    padding = -current_len % seqlen
    return np.pad(
        inp, [(0, padding)] + [(0, 0)] * (len(inp.shape) - 1), "constant"
    )
  else:
    return inp


def compute_token_types(
    inp: Dict[str, np.ndarray], ranges: masking_utils.TokenTypeRanges
) -> Dict[str, np.ndarray]:
  """Computes token types using a dictionary."""
  for key in ("inputs", "labels"):
    if ranges is not None:
      # Only ever happens for terminals on PTB
      # For CC, we have explicit ranges available, for datasets tokenised with
      # SentencePiece, we derive ranges from the .vocab file, so this is very
      # much a corner case.
      inp[f"{key}_ttypes"] = ranges.token_type_from_token(
          inp[key], use_jax=False
      )
    else:
      inp[f"{key}_ttypes"] = np.zeros_like(inp[key])
  return inp


def chunks_generator(it, ranges, maskrules) -> Generator[Chunk, None, None]:
  """Yields chunks to be passed to model from enumerated sequences iterator."""
  for tpl in it:
    item_idx, item = tpl
    if not isinstance(item, dict):
      item = dict(for_observation=item, for_target=item)
    item = compute_inputs_and_labels(item)
    item = compute_token_types(item, ranges)
    for chunk in maskrules.chunks_for_sequence(
        item["inputs"],
        item["inputs_ttypes"],
        item["labels"],
        item["labels_ttypes"],
    ):
      yield Chunk(np.array(item_idx, dtype=np.int32), *chunk)


def _batches_generator(
    chunks_gen_callable: Callable[[], Generator[Chunk, None, None]],
    shape_prefix: Sequence[int]
) -> Generator[Chunk, None, None]:
  """Yields batches (batched chunks)."""
  batch_size = functools.reduce(operator.mul, shape_prefix, 1)

  zipped_it = itertools.zip_longest(
      *[chunks_gen_callable() for _ in range(batch_size)], fillvalue=None
  )

  def safe_zipped_gen():
    zeroed_example = None
    for arrays in zipped_it:
      # The worker threads may pass an exception instead of a batch element for
      # it to be caught in the main thread.
      for arr in arrays:
        if isinstance(arr, Exception):
          raise arr
      if zeroed_example is None:
        not_none_indices = [
            i for (i, arr) in enumerate(arrays) if arr is not None
        ]
        assert not_none_indices
        not_none_idx = not_none_indices[0]
        not_none_example = arrays[not_none_idx]
        zeroed_example = tree.map_structure(np.zeros_like, not_none_example)
        zeroed_example = zeroed_example._replace(
            seq_idx=zeroed_example.seq_idx - 1
        )
      arrays = [arr if arr is not None else zeroed_example for arr in arrays]
      assert all(arr is not None for arr in arrays)
      yield arrays

  def stack_and_reshape(arrays):
    arr = np.stack(arrays)
    return arr.reshape(shape_prefix + tuple(arr.shape[1:]))

  for zipped in safe_zipped_gen():
    yield tree.map_structure(lambda *args: stack_and_reshape(args), *zipped)


def get_chunks_from_dataset(
    it, maskrules, ranges, shape_prefix, multithread, use_monitor_thread=False
) -> Generator[Chunk, None, None]:
  """Generates batches of chunks from sequences from a dataset.

  In multithreaded mode, this creates as many threads as the batch size. A batch
  is composed of elements, each preprocessed by a thread.

  Args:
    it: Iterator over raw dataset elements (either unbatched, unpadded sequences
      of ints, or dicts thereof).
    maskrules: Masking rules to use.
    ranges: Token types ranges.
    shape_prefix: Prefix of the shape that comes before the time dimension.
    multithread: Whether to use multithreaded mode or not.
    use_monitor_thread: Whether to use a monitor thread or not (periodically
      logging the number of elements in the queues, useful for debugging).

  Yields:
    Chunks.
  """
  it = enumerate(it)
  if not multithread:
    chunks_gen_callable = lambda: chunks_generator(it, ranges, maskrules)
    yield from _batches_generator(chunks_gen_callable, shape_prefix)
  else:
    batch_size = functools.reduce(operator.mul, shape_prefix, 1)
    in_queue = queue.Queue(maxsize=100)
    queues = [queue.Queue(maxsize=5) for _ in range(batch_size)]

    def producer():
      for x in it:
        in_queue.put(x)

    def worker(i):
      try:
        subit = (in_queue.get() for _ in itertools.count())
        for chunk in chunks_generator(subit, ranges, maskrules):
          queues[i].put(chunk)
      except Exception as exc:  # pylint: disable=broad-except
        queues[i].put(exc)

    threading.Thread(target=producer, daemon=True).start()
    for i in range(batch_size):
      threading.Thread(target=worker, args=(i,), daemon=True).start()

    def reader_gen(i):
      while True:
        yield queues[i].get()

    if use_monitor_thread:

      def monitor():
        while True:
          logging.info("in_queue: %d", in_queue.qsize())
          for i, q in enumerate(queues):
            logging.info("queue[%d]: %d", i, q.qsize())
          time.sleep(5)

      threading.Thread(target=monitor, daemon=True).start()

    reader_gens = [reader_gen(i) for i in range(batch_size)]
    reader_callables = reader_gens.pop

    yield from _batches_generator(reader_callables, shape_prefix)

