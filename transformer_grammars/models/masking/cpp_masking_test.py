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

"""Tests for transformer_grammars.models.masking.cpp_masking."""

import unittest
from absl import logging
import numpy as np
from parameterized import parameterized
from transformer_grammars.models.masking import constants as mc
from transformer_grammars.models.masking import cpp_masking
from transformer_grammars.models.masking import masking_types as types

# pylint: disable=g-generic-assert

# Sets of kwargs for test_ctor_does_not_raise
_DEFAULT_KWARGS = dict(sequence_length=4, memory_length=4)
_DELTA_DEPTH_KWARGS = dict(
    sequence_length=4,
    memory_length=4,
    relative_pos="delta_depth",
    use_different_attn_fns=True,
)


class FastMaskingTest(unittest.TestCase):

  def assertAllEqual(self, actual, expected):
    self.assertEqual(
        np.all(expected == actual), True, f"{expected} != {actual}"
    )

  def assertLen(self, container, length):
    self.assertEqual(len(container), length)

  def assertContainsSubsequence(self, seq, subseq):
    self.assertTrue(subseq in seq)

  def test_stack_compose_init_docstring(self):
    """Tests that a docstring is set on __init__ for stack/compose."""
    self.assertContainsSubsequence(
        cpp_masking.StackComposeDoubleClosingNT.__init__.__doc__,
        "Initialises the stack/compose masking rule.",
    )

  def test_txl_init_docstring(self):
    """Tests that a docstring is set on __init__ for TXL-style masking."""
    self.assertContainsSubsequence(
        cpp_masking.TXLCausalMasking.__init__.__doc__,
        "Initialises the TXL-style causal masking rule.",
    )

  @parameterized.expand([
      ("default", _DEFAULT_KWARGS),
      ("delta_depth_different_attn_fns", _DELTA_DEPTH_KWARGS),
  ])
  def test_ctor_does_not_raise(self, _, kwargs):
    """Tests that construction of the rules succeeds for correct kwargs."""
    _ = cpp_masking.StackComposeDoubleClosingNT(**kwargs)

  def test_ctor_raises_on_invalid_relative_position_type(self):
    """Tests that construction of the rules fails on invalid relpos type."""
    with self.assertRaises(RuntimeError):
      _ = cpp_masking.StackComposeDoubleClosingNT(
          sequence_length=4,
          memory_length=4,
          relative_pos="foo",
          use_different_attn_fns=True,
      )

  @parameterized.expand([
      ("different_attn_fns", True, 2),
      ("single_attn_fn", False, 1),
  ])
  def test_num_attention_functions(
      self, _, use_different_attn_fns, expected_num_attention_functions
  ):
    """Tests the `num_attention_functions` property of the masking rule."""
    maskrules = cpp_masking.StackComposeDoubleClosingNT(
        sequence_length=4,
        memory_length=4,
        relative_pos="delta_depth",
        use_different_attn_fns=use_different_attn_fns,
    )
    self.assertEqual(
        maskrules.num_attention_functions, expected_num_attention_functions
    )

  @parameterized.expand([
      ("delta_depth", "delta_depth", True),
      ("no_relative_positions", "", False),
  ])
  def test_use_relative_positions_stack_compose(
      self, _, relative_pos, expected_use_relative_positions
  ):
    """Tests the `use_relative_positions` property of the masking rule."""
    maskrules = cpp_masking.StackComposeDoubleClosingNT(
        sequence_length=4,
        memory_length=4,
        relative_pos=relative_pos,
        use_different_attn_fns=True,
    )
    self.assertEqual(
        maskrules.use_relative_positions, expected_use_relative_positions
    )

  def test_use_relative_positions_txl(self):
    """Tests the `use_relative_positions` property in the TXL case."""
    maskrules = cpp_masking.TXLCausalMasking(sequence_length=4, memory_length=4)
    self.assertFalse(maskrules.use_relative_positions)

  def _data(self):
    seq = np.array(
        [
            1,  # <s>
            2,  # (S
            3,  # (NP
            4,  # the
            5,  # hungry
            6,  # cat
            7,  # NP)
            8,  # (VP
            9,  # meows
            10,  # NP)
            11,  # S)
            0,  # <pad>
        ],
        dtype=np.int32,
    )
    ttypes = np.array(
        [
            mc.OPENING_NT,  # <s>
            mc.OPENING_NT,  # (S
            mc.OPENING_NT,  # (NP
            mc.TERMINAL,  # the
            mc.TERMINAL,  # hungry
            mc.TERMINAL,  # cat
            mc.CLOSING_NT,  # NP)
            mc.OPENING_NT,  # (VP
            mc.TERMINAL,  # meows
            mc.CLOSING_NT,  # VP)
            mc.CLOSING_NT,  # S)
            mc.PAD,  # <pad>
        ],
        dtype=np.int32,
    )
    inputs = seq[:-1]
    labels = seq[1:]
    inputs_ttypes = ttypes[:-1]
    labels_ttypes = ttypes[1:]
    return inputs, labels, inputs_ttypes, labels_ttypes

  def test_stack_compose_double_closing_nt(self):
    """Runs stack/compose code on known inputs, compares with gold outputs."""
    inputs, labels, inputs_ttypes, labels_ttypes = self._data()

    maskrules = cpp_masking.StackComposeDoubleClosingNT(
        sequence_length=4, memory_length=4, use_different_attn_fns=True
    )
    chunks = maskrules.chunks_for_sequence(
        inputs, inputs_ttypes, labels, labels_ttypes
    )
    chunks = [types.Chunk(None, *chunk) for chunk in chunks]

    for chunk in chunks:
      logging.info("Got chunk: %s", repr(chunk))

    actual_t_inputs = np.concatenate([chunk.inputs for chunk in chunks], axis=0)
    expected_t_inputs = np.array([
        1,  # <s>
        2,  # (S
        3,  # (NP
        4,  # the
        5,  # hungry
        6,  # cat
        7,  # NP)
        7,  # NP)
        8,  # (VP
        9,  # meows
        10,  # NP)
        10,  # NP)
        11,  # S)
        11,  # S)
        0,  # <pad>
        0,  # <pad>
    ])
    logging.info("Actual transformed inputs: %s", repr(actual_t_inputs))
    self.assertAllEqual(expected_t_inputs, actual_t_inputs)

    actual_t_labels = np.concatenate([chunk.labels for chunk in chunks], axis=0)
    expected_t_labels = np.array([
        2,  # (S
        3,  # (NP
        4,  # the
        5,  # hungry
        6,  # cat
        7,  # NP)
        0,  # <pad> !
        8,  # (VP
        9,  # meows
        10,  # NP)
        0,  # <pad> !
        11,  # S)
        0,  # <pad> !
        0,  # <pad>
        0,  # <pad>
        0,  # <pad>
    ])
    logging.info("Actual transformed labels: %s", repr(actual_t_labels))
    self.assertAllEqual(expected_t_labels, actual_t_labels)

    # Sequence padded to length 16, so 4 chunks of size 4
    self.assertLen(chunks, 4)

    self.assertAllEqual(chunks[0].inputs, np.array([1, 2, 3, 4]))
    self.assertAllEqual(
        chunks[0].inputs_ttypes,
        np.array([mc.OPENING_NT, mc.OPENING_NT, mc.OPENING_NT, mc.TERMINAL]),
    )
    self.assertAllEqual(chunks[0].labels, np.array([2, 3, 4, 5]))
    self.assertAllEqual(
        chunks[0].labels_ttypes,
        np.array([mc.OPENING_NT, mc.OPENING_NT, mc.TERMINAL, mc.TERMINAL]),
    )
    self.assertAllEqual(chunks[0].attn_indicator, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[0].memory_padding_mask, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[0].memory_pos, np.array([-1, -1, -1, -1]))
    self.assertAllEqual(chunks[0].depth, np.array([0, 1, 2, 3]))
    self.assertAllEqual(chunks[0].beginning_of_seq, np.array(1))
    self.assertAllEqual(chunks[0].end_of_seq, np.array(0))
    self.assertAllEqual(
        chunks[0].smartmem_mem_from_seq, np.eye(4, dtype=np.int32)
    )
    self.assertAllEqual(
        chunks[0].smartmem_mem_from_mem, np.zeros((4, 4), dtype=np.int32)
    )
    # Stack attention: <s> -> <s>
    self.assertAllEqual(chunks[0].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[0].attn_relpos[0], np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[0], np.array([0, 0, 0, 0]))
    # Stack attention: (S -> <s> (S
    self.assertAllEqual(chunks[0].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[0].attn_relpos[1], np.array([0, 0, 0, 0, 1, 0, 0, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[1], np.array([0, 0, 0, 0]))
    # Stack attention: (NP -> <s> (S (NP
    self.assertAllEqual(chunks[0].attn_mask[2], np.array([1, 1, 1, 0]))
    self.assertAllEqual(
        chunks[0].attn_relpos[2], np.array([0, 0, 0, 0, 2, 1, 0, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[2], np.array([0, 0, 0, 0]))
    # Stack attention: the -> <s> (S (NP the
    self.assertAllEqual(chunks[0].attn_mask[3], np.array([1, 1, 1, 1]))
    self.assertAllEqual(
        chunks[0].attn_relpos[3], np.array([0, 0, 0, 0, 3, 2, 1, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[3], np.array([0, 0, 0, 0]))

    self.assertAllEqual(chunks[1].inputs, np.array([5, 6, 7, 7]))
    self.assertAllEqual(
        chunks[1].inputs_ttypes,
        np.array([mc.TERMINAL, mc.TERMINAL, mc.CLOSING_NT, mc.CLOSING_NT_2]),
    )
    self.assertAllEqual(chunks[1].labels, np.array([6, 7, 0, 8]))
    self.assertAllEqual(
        chunks[1].labels_ttypes,
        np.array([mc.TERMINAL, mc.CLOSING_NT, mc.PAD, mc.OPENING_NT]),
    )
    self.assertAllEqual(chunks[1].attn_indicator, np.array([0, 0, 1, 0]))
    self.assertAllEqual(chunks[1].memory_padding_mask, np.array([1, 1, 1, 1]))
    self.assertAllEqual(chunks[1].memory_pos, np.array([0, 1, 2, 3]))
    self.assertAllEqual(chunks[1].depth, np.array([3, 3, 2, 2]))
    self.assertAllEqual(chunks[1].beginning_of_seq, np.array(0))
    self.assertAllEqual(chunks[1].end_of_seq, np.array(0))
    self.assertAllEqual(
        chunks[1].smartmem_mem_from_seq, np.eye(4, dtype=np.int32)
    )
    self.assertAllEqual(
        chunks[1].smartmem_mem_from_mem, np.zeros((4, 4), dtype=np.int32)
    )
    # Stack attention: hungry -> [<s> (S (NP the] hungry
    self.assertAllEqual(chunks[1].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[1].attn_relpos[0], np.array([3, 2, 1, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[0], np.array([1, 1, 1, 1]))
    # Stack attention: cat -> [<s> (S (NP the] hungry cat
    self.assertAllEqual(chunks[1].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[1].attn_relpos[1], np.array([3, 2, 1, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[1], np.array([1, 1, 1, 1]))
    # COMPOSE attention: NP) -> [(NP the] hungry cat NP)
    self.assertAllEqual(chunks[1].attn_mask[2], np.array([1, 1, 1, 0]))
    self.assertAllEqual(
        chunks[1].attn_relpos[2], np.array([0, 0, 0, -1, -1, -1, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[2], np.array([0, 0, 1, 1]))
    # Stack attention: NP) -> [<s> (NP] NP) NP)
    self.assertAllEqual(chunks[1].attn_mask[3], np.array([0, 0, 1, 1]))
    self.assertAllEqual(
        chunks[1].attn_relpos[3], np.array([2, 1, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[3], np.array([1, 1, 0, 0]))

    self.assertAllEqual(chunks[2].inputs, np.array([8, 9, 10, 10]))
    self.assertAllEqual(
        chunks[2].inputs_ttypes,
        np.array([mc.OPENING_NT, mc.TERMINAL, mc.CLOSING_NT, mc.CLOSING_NT_2]),
    )
    self.assertAllEqual(chunks[2].labels, np.array([9, 10, 0, 11]))
    self.assertAllEqual(
        chunks[2].labels_ttypes,
        np.array([mc.TERMINAL, mc.CLOSING_NT, mc.PAD, mc.CLOSING_NT]),
    )
    self.assertAllEqual(chunks[2].attn_indicator, np.array([0, 0, 1, 0]))
    self.assertAllEqual(chunks[2].memory_padding_mask, np.array([1, 1, 1, 1]))
    self.assertAllEqual(chunks[2].memory_pos, np.array([4, 5, 6, 7]))
    self.assertAllEqual(chunks[2].depth, np.array([2, 3, 2, 2]))
    self.assertAllEqual(chunks[2].beginning_of_seq, np.array(0))
    self.assertAllEqual(chunks[2].end_of_seq, np.array(0))
    self.assertAllEqual(
        chunks[2].smartmem_mem_from_seq, np.eye(4, dtype=np.int32)
    )
    self.assertAllEqual(
        chunks[2].smartmem_mem_from_mem, np.zeros((4, 4), dtype=np.int32)
    )
    # Stack attention: (VP -> [[<s> (S]] [NP)] (VP
    self.assertAllEqual(chunks[2].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[2].attn_relpos[0], np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[0], np.array([0, 0, 1, 0]))
    # Stack attention: meows -> [[<s> (S]] [NP)] (VP
    self.assertAllEqual(chunks[2].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[2].attn_relpos[1], np.array([0, 0, 1, 0, 1, 0, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[1], np.array([0, 0, 1, 0]))
    # COMPOSE attention: VP) -> (VP meows VP)
    self.assertAllEqual(chunks[2].attn_mask[2], np.array([1, 1, 1, 0]))
    self.assertAllEqual(
        chunks[2].attn_relpos[2], np.array([0, 0, 0, 0, 0, -1, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[2], np.array([0, 0, 0, 0]))
    # Stack attention: VP) -> [[<s> (S]] [NP)] (VP VP)
    self.assertAllEqual(chunks[2].attn_mask[3], np.array([0, 0, 1, 1]))
    self.assertAllEqual(
        chunks[2].attn_relpos[3], np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[3], np.array([0, 0, 1, 0]))

    self.assertAllEqual(chunks[3].inputs, np.array([11, 11, 0, 0]))
    self.assertAllEqual(
        chunks[3].inputs_ttypes,
        np.array([mc.CLOSING_NT, mc.CLOSING_NT_2, mc.PAD, mc.PAD]),
    )
    self.assertAllEqual(chunks[3].labels, np.array([0, 0, 0, 0]))
    self.assertAllEqual(
        chunks[3].labels_ttypes, np.array([mc.PAD, mc.PAD, mc.PAD, mc.PAD])
    )
    self.assertAllEqual(chunks[3].attn_indicator, np.array([1, 0, 0, 0]))
    self.assertAllEqual(chunks[3].memory_padding_mask, np.array([1, 1, 1, 1]))
    self.assertAllEqual(chunks[3].memory_pos, np.array([8, 9, 10, 11]))
    self.assertAllEqual(chunks[3].depth, np.array([1, 1, 0, 0]))
    self.assertAllEqual(chunks[3].beginning_of_seq, np.array(0))
    self.assertAllEqual(chunks[3].end_of_seq, np.array(1))
    self.assertAllEqual(
        chunks[3].smartmem_mem_from_seq,
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int32,
        ),
    )
    self.assertAllEqual(
        chunks[3].smartmem_mem_from_mem, np.zeros((4, 4), dtype=np.int32)
    )
    # COMPOSE attention: S) -> [[(S NP)]] [VP)] S)
    self.assertAllEqual(chunks[3].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[3].attn_relpos[0], np.array([0, 0, -1, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[3].memory_attn_mask[0], np.array([0, 0, 1, 0]))
    # Stack attention: S) -> [[<s>]] S) S)
    self.assertAllEqual(chunks[3].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[3].attn_relpos[1], np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[3].memory_attn_mask[1], np.array([0, 0, 0, 0]))
    # Attention: <pad> -> nothing
    self.assertAllEqual(chunks[3].attn_mask[2], np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[3].memory_attn_mask[2], np.array([0, 0, 0, 0]))
    self.assertAllEqual(
        chunks[3].attn_relpos[2], np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    # Attention: <pad> -> nothing
    self.assertAllEqual(chunks[3].attn_mask[3], np.array([0, 0, 0, 0]))
    self.assertAllEqual(
        chunks[3].attn_relpos[3], np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[3].memory_attn_mask[3], np.array([0, 0, 0, 0]))

  def test_stack_compose_double_closing_nt_smartmem(self):
    """Runs stack/compose code on known inputs, compares with gold outputs.

    With "smart" memory, i.e. updating the memory so that tokens that won't
    be attended to in the future are not added to the memory at all.
    """
    inputs, labels, inputs_ttypes, labels_ttypes = self._data()

    maskrules = cpp_masking.StackComposeDoubleClosingNT(
        sequence_length=4,
        memory_length=4,
        use_different_attn_fns=True,
        gather_into_new_memory=True,
    )
    chunks = maskrules.chunks_for_sequence(
        inputs, inputs_ttypes, labels, labels_ttypes
    )
    chunks = [types.Chunk(None, *chunk) for chunk in chunks]

    for chunk in chunks:
      logging.info("Got chunk: %s", repr(chunk))

    actual_t_inputs = np.concatenate([chunk.inputs for chunk in chunks], axis=0)
    expected_t_inputs = np.array([
        1,  # <s>
        2,  # (S
        3,  # (NP
        4,  # the
        5,  # hungry
        6,  # cat
        7,  # NP)
        7,  # NP)
        8,  # (VP
        9,  # meows
        10,  # NP)
        10,  # NP)
        11,  # S)
        11,  # S)
        0,  # <pad>
        0,  # <pad>
    ])
    logging.info("Actual transformed inputs: %s", repr(actual_t_inputs))
    self.assertAllEqual(expected_t_inputs, actual_t_inputs)

    actual_t_labels = np.concatenate([chunk.labels for chunk in chunks], axis=0)
    expected_t_labels = np.array([
        2,  # (S
        3,  # (NP
        4,  # the
        5,  # hungry
        6,  # cat
        7,  # NP)
        0,  # <pad> !
        8,  # (VP
        9,  # meows
        10,  # NP)
        0,  # <pad> !
        11,  # S)
        0,  # <pad> !
        0,  # <pad>
        0,  # <pad>
        0,  # <pad>
    ])
    logging.info("Actual transformed labels: %s", repr(actual_t_labels))
    self.assertAllEqual(expected_t_labels, actual_t_labels)

    # Sequence padded to length 16, so 4 chunks of size 4
    self.assertLen(chunks, 4)

    self.assertAllEqual(chunks[0].inputs, np.array([1, 2, 3, 4]))
    self.assertAllEqual(
        chunks[0].inputs_ttypes,
        np.array([mc.OPENING_NT, mc.OPENING_NT, mc.OPENING_NT, mc.TERMINAL]),
    )
    self.assertAllEqual(chunks[0].labels, np.array([2, 3, 4, 5]))
    self.assertAllEqual(
        chunks[0].labels_ttypes,
        np.array([mc.OPENING_NT, mc.OPENING_NT, mc.TERMINAL, mc.TERMINAL]),
    )
    self.assertAllEqual(chunks[0].attn_indicator, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[0].memory_padding_mask, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[0].memory_pos, np.array([-1, -1, -1, -1]))
    self.assertAllEqual(chunks[0].depth, np.array([0, 1, 2, 3]))
    self.assertAllEqual(chunks[0].beginning_of_seq, np.array(1))
    self.assertAllEqual(chunks[0].end_of_seq, np.array(0))
    self.assertAllEqual(
        chunks[0].smartmem_mem_from_seq, np.eye(4, dtype=np.int32)
    )
    self.assertAllEqual(
        chunks[0].smartmem_mem_from_mem, np.zeros((4, 4), dtype=np.int32)
    )
    # Stack attention: <s> -> <s>
    self.assertAllEqual(chunks[0].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[0].attn_relpos[0], np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[0], np.array([0, 0, 0, 0]))
    # Stack attention: (S -> <s> (S
    self.assertAllEqual(chunks[0].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[0].attn_relpos[1], np.array([0, 0, 0, 0, 1, 0, 0, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[1], np.array([0, 0, 0, 0]))
    # Stack attention: (NP -> <s> (S (NP
    self.assertAllEqual(chunks[0].attn_mask[2], np.array([1, 1, 1, 0]))
    self.assertAllEqual(
        chunks[0].attn_relpos[2], np.array([0, 0, 0, 0, 2, 1, 0, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[2], np.array([0, 0, 0, 0]))
    # Stack attention: the -> <s> (S (NP the
    self.assertAllEqual(chunks[0].attn_mask[3], np.array([1, 1, 1, 1]))
    self.assertAllEqual(
        chunks[0].attn_relpos[3], np.array([0, 0, 0, 0, 3, 2, 1, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[3], np.array([0, 0, 0, 0]))

    self.assertAllEqual(chunks[1].inputs, np.array([5, 6, 7, 7]))
    self.assertAllEqual(
        chunks[1].inputs_ttypes,
        np.array([mc.TERMINAL, mc.TERMINAL, mc.CLOSING_NT, mc.CLOSING_NT_2]),
    )
    self.assertAllEqual(chunks[1].labels, np.array([6, 7, 0, 8]))
    self.assertAllEqual(
        chunks[1].labels_ttypes,
        np.array([mc.TERMINAL, mc.CLOSING_NT, mc.PAD, mc.OPENING_NT]),
    )
    self.assertAllEqual(chunks[1].attn_indicator, np.array([0, 0, 1, 0]))
    self.assertAllEqual(chunks[1].memory_padding_mask, np.array([1, 1, 1, 1]))
    self.assertAllEqual(chunks[1].memory_pos, np.array([0, 1, 2, 3]))
    self.assertAllEqual(chunks[1].depth, np.array([3, 3, 2, 2]))
    self.assertAllEqual(chunks[1].beginning_of_seq, np.array(0))
    self.assertAllEqual(chunks[1].end_of_seq, np.array(0))
    self.assertAllEqual(
        chunks[1].smartmem_mem_from_seq,
        np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
            dtype=np.int32,
        ),
    )
    self.assertAllEqual(
        chunks[1].smartmem_mem_from_mem,
        np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int32,
        ),
    )
    # Stack attention: hungry -> [<s> (S (NP the] hungry
    self.assertAllEqual(chunks[1].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[1].attn_relpos[0], np.array([3, 2, 1, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[0], np.array([1, 1, 1, 1]))
    # Stack attention: cat -> [<s> (S (NP the] hungry cat
    self.assertAllEqual(chunks[1].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[1].attn_relpos[1], np.array([3, 2, 1, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[1], np.array([1, 1, 1, 1]))
    # COMPOSE attention: NP) -> [(NP the] hungry cat NP)
    self.assertAllEqual(chunks[1].attn_mask[2], np.array([1, 1, 1, 0]))
    self.assertAllEqual(
        chunks[1].attn_relpos[2], np.array([0, 0, 0, -1, -1, -1, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[2], np.array([0, 0, 1, 1]))
    # Stack attention: NP) -> [<s> (NP] NP) NP)
    self.assertAllEqual(chunks[1].attn_mask[3], np.array([0, 0, 1, 1]))
    self.assertAllEqual(
        chunks[1].attn_relpos[3], np.array([2, 1, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[3], np.array([1, 1, 0, 0]))

    self.assertAllEqual(chunks[2].inputs, np.array([8, 9, 10, 10]))
    self.assertAllEqual(
        chunks[2].inputs_ttypes,
        np.array([mc.OPENING_NT, mc.TERMINAL, mc.CLOSING_NT, mc.CLOSING_NT_2]),
    )
    self.assertAllEqual(chunks[2].labels, np.array([9, 10, 0, 11]))
    self.assertAllEqual(
        chunks[2].labels_ttypes,
        np.array([mc.TERMINAL, mc.CLOSING_NT, mc.PAD, mc.CLOSING_NT]),
    )
    self.assertAllEqual(chunks[2].attn_indicator, np.array([0, 0, 1, 0]))
    self.assertAllEqual(chunks[2].memory_padding_mask, np.array([0, 1, 1, 1]))
    self.assertAllEqual(chunks[2].memory_pos, np.array([-1, 0, 1, 6]))
    self.assertAllEqual(chunks[2].depth, np.array([2, 3, 2, 2]))
    self.assertAllEqual(chunks[2].beginning_of_seq, np.array(0))
    self.assertAllEqual(chunks[2].end_of_seq, np.array(0))
    self.assertAllEqual(
        chunks[2].smartmem_mem_from_seq,
        np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
            dtype=np.int32,
        ),
    )
    self.assertAllEqual(
        chunks[2].smartmem_mem_from_mem,
        np.array(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=np.int32,
        ),
    )

    # Stack attention: (VP -> [<s> (S NP)] (VP
    self.assertAllEqual(chunks[2].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[2].attn_relpos[0], np.array([0, 2, 1, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[0], np.array([0, 1, 1, 1]))
    # Stack attention: meows -> [<s> (S NP)] (VP
    self.assertAllEqual(chunks[2].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[2].attn_relpos[1], np.array([0, 3, 2, 1, 1, 0, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[1], np.array([0, 1, 1, 1]))
    # COMPOSE attention: VP) -> (VP meows VP)
    self.assertAllEqual(chunks[2].attn_mask[2], np.array([1, 1, 1, 0]))
    self.assertAllEqual(
        chunks[2].attn_relpos[2], np.array([0, 0, 0, 0, 0, -1, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[2], np.array([0, 0, 0, 0]))
    # Stack attention: VP) -> [<s> (S NP)] (VP VP)
    self.assertAllEqual(chunks[2].attn_mask[3], np.array([0, 0, 1, 1]))
    self.assertAllEqual(
        chunks[2].attn_relpos[3], np.array([0, 2, 1, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[3], np.array([0, 1, 1, 1]))

    self.assertAllEqual(chunks[3].inputs, np.array([11, 11, 0, 0]))
    self.assertAllEqual(
        chunks[3].inputs_ttypes,
        np.array([mc.CLOSING_NT, mc.CLOSING_NT_2, mc.PAD, mc.PAD]),
    )
    self.assertAllEqual(chunks[3].labels, np.array([0, 0, 0, 0]))
    self.assertAllEqual(
        chunks[3].labels_ttypes, np.array([mc.PAD, mc.PAD, mc.PAD, mc.PAD])
    )
    self.assertAllEqual(chunks[3].attn_indicator, np.array([1, 0, 0, 0]))
    self.assertAllEqual(chunks[3].memory_padding_mask, np.array([1, 1, 1, 1]))
    self.assertAllEqual(chunks[3].memory_pos, np.array([0, 1, 6, 10]))
    self.assertAllEqual(chunks[3].depth, np.array([1, 1, 0, 0]))
    self.assertAllEqual(chunks[3].beginning_of_seq, np.array(0))
    self.assertAllEqual(chunks[3].end_of_seq, np.array(1))
    self.assertAllEqual(
        chunks[3].smartmem_mem_from_seq,
        np.array(
            [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int32,
        ),
    )
    self.assertAllEqual(
        chunks[3].smartmem_mem_from_mem,
        np.array(
            [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int32,
        ),
    )
    # COMPOSE attention: S) -> [(S NP) VP)] S)
    self.assertAllEqual(chunks[3].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[3].attn_relpos[0], np.array([0, 0, -1, -1, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[3].memory_attn_mask[0], np.array([0, 1, 1, 1]))
    # Stack attention: S) -> [<s>] S) S)
    self.assertAllEqual(chunks[3].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[3].attn_relpos[1], np.array([1, 0, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[3].memory_attn_mask[1], np.array([1, 0, 0, 0]))
    # Attention: <pad> -> nothing
    self.assertAllEqual(chunks[3].attn_mask[2], np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[3].memory_attn_mask[2], np.array([0, 0, 0, 0]))
    self.assertAllEqual(
        chunks[3].attn_relpos[2], np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    # Attention: <pad> -> nothing
    self.assertAllEqual(chunks[3].attn_mask[3], np.array([0, 0, 0, 0]))
    self.assertAllEqual(
        chunks[3].attn_relpos[3], np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[3].memory_attn_mask[3], np.array([0, 0, 0, 0]))

  def test_txl(self):
    """Runs TXL masking code to known inputs, compares with gold outputs."""
    inputs, labels, inputs_ttypes, labels_ttypes = self._data()

    maskrules = cpp_masking.TXLCausalMasking(sequence_length=4, memory_length=4)
    chunks = maskrules.chunks_for_sequence(
        inputs, inputs_ttypes, labels, labels_ttypes
    )
    chunks = [types.Chunk(None, *chunk) for chunk in chunks]

    # Sequence padded to length 12, so 3 chunks of size 4
    self.assertLen(chunks, 3)

    for chunk in chunks:
      logging.info("Got chunk: %s", repr(chunk))

    actual_inputs = np.concatenate([chunk.inputs for chunk in chunks], axis=0)
    expected_inputs = np.array([
        1,  # <s>
        2,  # (S
        3,  # (NP
        4,  # the
        5,  # hungry
        6,  # cat
        7,  # NP)
        8,  # (VP
        9,  # meows
        10,  # NP)
        11,  # S)
        0,  # <pad>
    ])
    logging.info("Actual inputs: %s", repr(actual_inputs))
    self.assertAllEqual(expected_inputs, actual_inputs)

    actual_labels = np.concatenate([chunk.labels for chunk in chunks], axis=0)
    expected_labels = np.array([
        2,  # (S
        3,  # (NP
        4,  # the
        5,  # hungry
        6,  # cat
        7,  # NP)
        8,  # (VP
        9,  # meows
        10,  # NP)
        11,  # S)
        0,  # <pad>
        0,  # <pad>
    ])
    logging.info("Actual labels: %s", repr(actual_labels))
    self.assertAllEqual(expected_labels, actual_labels)

    self.assertAllEqual(chunks[0].inputs, np.array([1, 2, 3, 4]))
    self.assertAllEqual(
        chunks[0].inputs_ttypes,
        np.array([mc.OPENING_NT, mc.OPENING_NT, mc.OPENING_NT, mc.TERMINAL]),
    )
    self.assertAllEqual(chunks[0].labels, np.array([2, 3, 4, 5]))
    self.assertAllEqual(
        chunks[0].labels_ttypes,
        np.array([mc.OPENING_NT, mc.OPENING_NT, mc.TERMINAL, mc.TERMINAL]),
    )
    self.assertAllEqual(chunks[0].attn_indicator, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[0].memory_padding_mask, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[0].memory_pos, np.array([-1, -1, -1, -1]))
    self.assertAllEqual(chunks[0].depth, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[0].beginning_of_seq, np.array(1))
    self.assertAllEqual(chunks[0].end_of_seq, np.array(0))
    self.assertAllEqual(
        chunks[0].smartmem_mem_from_seq, np.eye(4, dtype=np.int32)
    )
    self.assertAllEqual(
        chunks[0].smartmem_mem_from_mem, np.zeros((4, 4), dtype=np.int32)
    )

    self.assertAllEqual(chunks[0].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[0].attn_relpos[0], np.array([0, 0, 0, 0, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[0], np.array([0, 0, 0, 0]))

    self.assertAllEqual(chunks[0].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[0].attn_relpos[1], np.array([0, 0, 0, 0, 1, 0, 0, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[1], np.array([0, 0, 0, 0]))

    self.assertAllEqual(chunks[0].attn_mask[2], np.array([1, 1, 1, 0]))
    self.assertAllEqual(
        chunks[0].attn_relpos[2], np.array([0, 0, 0, 0, 2, 1, 0, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[2], np.array([0, 0, 0, 0]))

    self.assertAllEqual(chunks[0].attn_mask[3], np.array([1, 1, 1, 1]))
    self.assertAllEqual(
        chunks[0].attn_relpos[3], np.array([0, 0, 0, 0, 3, 2, 1, 0])
    )
    self.assertAllEqual(chunks[0].memory_attn_mask[3], np.array([0, 0, 0, 0]))

    self.assertAllEqual(chunks[1].inputs, np.array([5, 6, 7, 8]))
    self.assertAllEqual(
        chunks[1].inputs_ttypes,
        np.array([mc.TERMINAL, mc.TERMINAL, mc.CLOSING_NT, mc.OPENING_NT]),
    )
    self.assertAllEqual(chunks[1].labels, np.array([6, 7, 8, 9]))
    self.assertAllEqual(
        chunks[1].labels_ttypes,
        np.array([mc.TERMINAL, mc.CLOSING_NT, mc.OPENING_NT, mc.TERMINAL]),
    )
    self.assertAllEqual(chunks[1].attn_indicator, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[1].memory_padding_mask, np.array([1, 1, 1, 1]))
    self.assertAllEqual(chunks[1].memory_pos, np.array([0, 1, 2, 3]))
    self.assertAllEqual(chunks[1].depth, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[1].beginning_of_seq, np.array(0))
    self.assertAllEqual(chunks[1].end_of_seq, np.array(0))
    self.assertAllEqual(
        chunks[1].smartmem_mem_from_seq, np.eye(4, dtype=np.int32)
    )
    self.assertAllEqual(
        chunks[1].smartmem_mem_from_mem, np.zeros((4, 4), dtype=np.int32)
    )

    self.assertAllEqual(chunks[1].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[1].attn_relpos[0], np.array([4, 3, 2, 1, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[0], np.array([1, 1, 1, 1]))

    self.assertAllEqual(chunks[1].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[1].attn_relpos[1], np.array([5, 4, 3, 2, 1, 0, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[1], np.array([1, 1, 1, 1]))

    self.assertAllEqual(chunks[1].attn_mask[2], np.array([1, 1, 1, 0]))
    self.assertAllEqual(
        chunks[1].attn_relpos[2], np.array([6, 5, 4, 3, 2, 1, 0, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[2], np.array([1, 1, 1, 1]))

    self.assertAllEqual(chunks[1].attn_mask[3], np.array([1, 1, 1, 1]))
    self.assertAllEqual(
        chunks[1].attn_relpos[3], np.array([7, 6, 5, 4, 3, 2, 1, 0])
    )
    self.assertAllEqual(chunks[1].memory_attn_mask[3], np.array([1, 1, 1, 1]))

    self.assertAllEqual(chunks[2].inputs, np.array([9, 10, 11, 0]))
    self.assertAllEqual(
        chunks[2].inputs_ttypes,
        np.array([mc.TERMINAL, mc.CLOSING_NT, mc.CLOSING_NT, mc.PAD]),
    )
    self.assertAllEqual(chunks[2].labels, np.array([10, 11, 0, 0]))
    self.assertAllEqual(
        chunks[2].labels_ttypes,
        np.array([mc.CLOSING_NT, mc.CLOSING_NT, mc.PAD, mc.PAD]),
    )
    self.assertAllEqual(chunks[2].attn_indicator, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[2].memory_padding_mask, np.array([1, 1, 1, 1]))
    self.assertAllEqual(chunks[2].memory_pos, np.array([4, 5, 6, 7]))
    self.assertAllEqual(chunks[2].depth, np.array([0, 0, 0, 0]))
    self.assertAllEqual(chunks[2].beginning_of_seq, np.array(0))
    self.assertAllEqual(chunks[2].end_of_seq, np.array(1))
    self.assertAllEqual(
        chunks[2].smartmem_mem_from_seq, np.eye(4, dtype=np.int32)
    )
    self.assertAllEqual(
        chunks[2].smartmem_mem_from_mem, np.zeros((4, 4), dtype=np.int32)
    )

    self.assertAllEqual(chunks[2].attn_mask[0], np.array([1, 0, 0, 0]))
    self.assertAllEqual(
        chunks[2].attn_relpos[0], np.array([4, 3, 2, 1, 0, 0, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[0], np.array([1, 1, 1, 1]))

    self.assertAllEqual(chunks[2].attn_mask[1], np.array([1, 1, 0, 0]))
    self.assertAllEqual(
        chunks[2].attn_relpos[1], np.array([5, 4, 3, 2, 1, 0, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[1], np.array([1, 1, 1, 1]))

    self.assertAllEqual(chunks[2].attn_mask[2], np.array([1, 1, 1, 0]))
    self.assertAllEqual(
        chunks[2].attn_relpos[2], np.array([6, 5, 4, 3, 2, 1, 0, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[2], np.array([1, 1, 1, 1]))

    self.assertAllEqual(chunks[2].attn_mask[3], np.array([1, 1, 1, 1]))
    self.assertAllEqual(
        chunks[2].attn_relpos[3], np.array([7, 6, 5, 4, 3, 2, 1, 0])
    )
    self.assertAllEqual(chunks[2].memory_attn_mask[3], np.array([1, 1, 1, 1]))
