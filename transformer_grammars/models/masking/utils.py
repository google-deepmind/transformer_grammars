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

"""Masks for Transformer Grammars models."""

import dataclasses
from typing import Optional, Tuple, Union

from absl import logging
import jax.numpy as jnp
import numpy as np

from transformer_grammars.data import constants
from transformer_grammars.models.masking import constants as mc
from transformer_grammars.models.masking import cpp_masking as mcpp


def _in_range(range_, arr, np_=jnp):
  min_, max_ = range_
  return np_.logical_and(np_.greater_equal(arr, min_), np_.less(arr, max_))


def _interval_from_list(l):
  minval = min(l)
  maxval = max(l)
  if len(set(l)) != len(l):
    raise ValueError("The list contains duplicated elements.")
  # No duplicated elements.
  if set(range(minval, maxval + 1)) == set(l):
    return (minval, maxval + 1)
  raise ValueError(
      "The values in the list do not exactly correspond to an interval."
  )


@dataclasses.dataclass(frozen=True)
class TokenTypeRanges:
  """Mapping between token IDs ranges to token types."""

  start_token: int
  pad_token: int
  end_token: int
  placeholder_token: Optional[int]
  opening_non_terminals: Tuple[int, int]
  closing_non_terminals: Tuple[int, int]
  terminals: Tuple[int, int]
  has_extra_untyped_closing_non_terminal: bool
  vocab_size: int

  @classmethod
  def from_dictionary_metadata(
      cls,
      *,
      num_reserved,
      num_terminals,
      num_opening_non_terminals,
      num_closing_non_terminals,
      extra_untyped_closing_non_terminal,
  ):
    """Returns ranges from dictionary metadata."""
    if num_reserved < 4:
      raise ValueError("At least 4 reserved tokens are required.")
    terminals_start = num_reserved
    terminals_end = num_reserved + num_terminals
    opening_nt_start = terminals_end
    opening_nt_end = opening_nt_start + num_opening_non_terminals
    closing_nt_start = opening_nt_end
    closing_nt_end = closing_nt_start + num_closing_non_terminals
    vocab_size = closing_nt_end + (
        1 if extra_untyped_closing_non_terminal else 0
    )
    return cls(
        start_token=constants.BOS,
        pad_token=constants.PAD,
        end_token=constants.EOS,
        terminals=(terminals_start, terminals_end),
        opening_non_terminals=(opening_nt_start, opening_nt_end),
        closing_non_terminals=(closing_nt_start, closing_nt_end),
        placeholder_token=constants.PLACEHOLDER,
        has_extra_untyped_closing_non_terminal=extra_untyped_closing_non_terminal,
        vocab_size=vocab_size,
    )

  @classmethod
  def from_sentencepiece_vocab(cls, vocab):
    return cls(
        start_token=vocab.bos,
        pad_token=vocab.pad,
        end_token=vocab.eos,
        placeholder_token=vocab.unk,
        opening_non_terminals=_interval_from_list(vocab.opening_nts),
        closing_non_terminals=_interval_from_list(vocab.closing_nts),
        terminals=_interval_from_list(vocab.terminals + [vocab.whitespace]),
        has_extra_untyped_closing_non_terminal=False,
        vocab_size=len(vocab.dictionary),
    )

  def token_type_from_token(
      self, seq: Union[jnp.array, np.ndarray], *, use_jax=True
  ):
    """Returns an array of token types from an array of token IDs."""
    if use_jax:
      np_ = jnp
    else:
      np_ = np
    start_token_mask = np_.equal(seq, self.start_token)
    pad_token_mask = np_.equal(seq, self.pad_token)
    if self.placeholder_token is not None:
      placeholder_mask = np_.equal(seq, self.placeholder_token)
    else:
      placeholder_mask = np_.zeros_like(start_token_mask)
    opening_nt_mask = _in_range(self.opening_non_terminals, seq, np_)
    closing_nt_mask = _in_range(self.closing_non_terminals, seq, np_)
    if self.has_extra_untyped_closing_non_terminal:
      closing_nt_mask = np_.logical_or(
          closing_nt_mask, np_.equal(self.closing_non_terminals[1], seq)
      )
    terminal_mask = _in_range(self.terminals, seq, np_)
    result = 0
    for mask, id_ in zip(
        [
            start_token_mask,
            pad_token_mask,
            placeholder_mask,
            opening_nt_mask,
            closing_nt_mask,
            terminal_mask,
        ],
        [
            mc.SOS,
            mc.PAD,
            mc.PLACEHOLDER,
            mc.OPENING_NT,
            mc.CLOSING_NT,
            mc.TERMINAL,
        ],
    ):
      result += mask.astype(np_.int32) * id_
    return result


def token_type_from_token(ranges: TokenTypeRanges, seq: jnp.array):
  """Returns an array of token types from an array of token IDs."""
  return ranges.token_type_from_token(seq, use_jax=True)


def get_masking_rules(name, **kwargs):
  """Returns the masking rules instance."""
  logging.info("Creating masking rules %s with kwargs=%s", name, repr(kwargs))
  if name == "stack_compose_double_closing_nt":
    cls = mcpp.StackComposeDoubleClosingNT
  elif name == "txl":
    cls = mcpp.TXLCausalMasking
  else:
    raise NotImplementedError
  if kwargs is None:
    kwargs = dict()
  maskrules = cls(**kwargs)
  return maskrules
