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

"""Simple text-based dataset."""

import functools
from typing import Dict, List, Optional, Tuple
import tensorflow.compat.v1 as tf
import tree

BOS_ID = 1
EOS_ID = 2
PREFETCH_COUNT = 32768


def _ints_from_string(add_bos, add_eos, s):
  seq = tf.strings.to_number(tf.strings.split(s, ",").values, tf.int32)
  parts = []
  if add_bos:
    parts.append(tf.constant([BOS_ID], shape=(1,), dtype=tf.int32))
  parts.append(seq)
  if add_eos:
    parts.append(tf.constant([EOS_ID], shape=(1,), dtype=tf.int32))
  return tf.concat(parts, axis=0)


def _parts_from_tsv_string(fields_spec, s):
  d = {}
  for name, idx in fields_spec.items():
    values = tf.strings.split(s, "\t").values
    d[name] = values[idx]
  return d


def _repeat_and_shuffle(
    ds,
    *,
    num_epochs,
    shuffle,
    shuffle_buffer,
    sample_without_replacement,
    seed,
    prefetch_count,
):
  """Apply shuffling to a dataset."""
  if shuffle and sample_without_replacement:
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True, seed=seed)
  ds = ds.repeat(num_epochs)
  if shuffle and not sample_without_replacement:
    ds = ds.shuffle(shuffle_buffer, seed=seed)
  ds = ds.prefetch(prefetch_count)
  return ds


def _get_shard_from_list(
    l: List[str], shard_idx: int, num_shards: int
) -> List[str]:
  """Returns a strided slice from a list."""
  if not 0 <= shard_idx < num_shards:
    raise ValueError(
        f"The shard index {shard_idx:d} is not compatible with the number "
        f"of shards {num_shards:d}."
    )
  if num_shards <= 0:
    raise ValueError(
        f"The number of shards {num_shards:d} must be positive."
    )
  if num_shards > len(l):
    raise ValueError(
        f"Cannot have more shards ({num_shards:d}) than items to shard "
        f"({len(l):d})."
    )
  return sorted(l)[shard_idx::num_shards]


def text_dataset(
    *,
    filenames: List[str],
    shuffle: bool,
    shuffle_buffer: Optional[int],
    return_key: bool,
    seed: Optional[int] = None,
    shard: Optional[Tuple[int, int]] = None,
) -> tf.data.Dataset:
  """Returns a raw text tf.data.Dataset, with the usual options."""
  if len(filenames) >= 2 and return_key and (shuffle or shuffle_buffer):
    raise RuntimeError(
        "return_key=True with shuffling is not supported for "
        "text datasets with more than one input file."
    )
  if return_key:
    num_parallel_reads = 1
  else:
    num_parallel_reads = 8

  if shard is not None:
    (shard_idx, num_shards) = shard
    filenames = _get_shard_from_list(filenames, shard_idx, num_shards)

  filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
  if shuffle and len(filenames) >= 2 and not return_key:
    # Shuffle filenames.
    filenames_ds = filenames_ds.shuffle(
        buffer_size=len(filenames), reshuffle_each_iteration=True, seed=seed
    )

  return tf.data.TextLineDataset(
      filenames=filenames_ds,
      buffer_size=(8 * 1024 * 1024),  # 8 MB
      num_parallel_reads=num_parallel_reads,
  )


class PreEncodedTextDataset:
  """Dataset of pre-encoded tokens.

  In the single field case, each line is a sequence of comma-separated
  integers:
    3,4,5
    8,19,38

  In the multiple fields cases, each line is a tab-separated sequence of
  sequences of comma-separated integers:
    3,4,5<TAB>3,4,6
    8,19,38<TAB>8,20,38
  """

  def __init__(
      self,
      filename: str,
      num_samples: Optional[int],
      add_bos: bool,
      add_eos: bool,
      multiple_fields: Optional[Dict[str, int]] = None,
      prefetch_count: int = PREFETCH_COUNT,
      max_seqlen: Optional[int] = None,
  ):
    """Initialises the TSVDataset.

    Args:
      filename: Name (or sharded filename, or globbing pattern) of the file
        constituting the dataset, i.e. the following are accepted: "foo.txt",
        "baz@123.txt" "/path/to/*.txt"
      num_samples: Total number of samples (pairs) in the dataset. Only required
        when the dataset is used for validation.
      add_bos: Prepend a beginning-of-sentence token (1) to each sequence.
      add_eos: Append an end-of-sentence token (2) to each sequence.
      multiple_fields: When not None, dict of field names in the output, mapping
        to field numbers in the input.
      prefetch_count: Number of items to pre-fetch from dataset.
      max_seqlen: Optional maximum sequence length. When set, only sequences
        strictly shorter than the maximum are kept.
    """
    self._num_samples = num_samples
    self._add_bos = add_bos
    self._add_eos = add_eos
    self._multiple_fields = multiple_fields
    self._prefetch_count = prefetch_count
    self._max_seqlen = max_seqlen
    filenames = filename.split(",")
    self._filenames = filenames
    if not self._filenames:
      raise ValueError(f"No filenames corresponding to {filename!s}.")

  @property
  def num_examples(self):
    """Number of examples."""
    return self._num_samples

  def raw_dataset(
      self,
      *,
      shuffle: bool,
      shuffle_buffer: Optional[int],
      sample_without_replacement: bool,
      num_epochs: Optional[int] = None,
      seed: Optional[int] = None,
  ) -> tf.data.Dataset:
    """Returns a raw tf.data.Dataset.

    In the single field case (self._multiple_fields is None), the dataset
    returned contains unbatched, unpadded, sequences of integers of dynamic
    shapes.

    In the multiple field case (self._multiple_fields is not None), the
    dataset returned contains dicts with unbatched, unpadded, sequences of
    integers of dynamic shapes as values, with the values of
    self._multiple_fields as keys, such that if output is a dict in the returned
    dataset, output[self._multiple_fields[0]] is the sequence in the first
    position in the input file, etc.

    Args:
      shuffle: Whether the dataset is shuffled (True) or not (False).
      shuffle_buffer: If applicable, size of the shuffle buffer.
      sample_without_replacement: Whether the dataset is shuffled without
        replacement (True) or not (False).
      num_epochs: If not None, number of epochs to repeat the dataset for.
        Otherwise, repeat infinitely.
      seed: If not None, seed to use for shuffling.

    Returns:
      Dataset described previously.
    """

    ds = text_dataset(
        filenames=self._filenames,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        return_key=False,
        seed=seed,
    )
    if self._multiple_fields:
      ds = ds.map(
          functools.partial(_parts_from_tsv_string, self._multiple_fields)
      )
    ds = ds.map(
        functools.partial(
            tree.map_structure,
            functools.partial(_ints_from_string, self._add_bos, self._add_eos),
        )
    )
    if self._max_seqlen:

      def _keep(item):
        return functools.reduce(
            tf.logical_and,
            [tf.shape(x)[0] < self._max_seqlen for x in tree.flatten(item)],
        )

      ds = ds.filter(_keep)
    ds = ds.cache()
    ds = _repeat_and_shuffle(
        ds,
        num_epochs=num_epochs,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        sample_without_replacement=sample_without_replacement,
        seed=seed,
        prefetch_count=self._prefetch_count,
    )
    return ds
