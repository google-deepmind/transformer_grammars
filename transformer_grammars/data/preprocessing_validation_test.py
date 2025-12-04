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

"""Standalone tests for input validation in preprocessing functions."""

import unittest
import numpy as np


def lshift(arr):
  """Shift array left by one element, padding with 0."""
  assert len(arr.shape) == 1
  arr = arr[1:]
  return np.pad(arr, [(0, 1)], mode="constant")


def compute_inputs_and_labels(
    inp: dict, use_untyped_closing_nt_for_labels: bool = False
) -> dict:
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
  assert "for_observation" in inp_, "Input dict must contain 'for_observation' key"
  assert "for_target" in inp_, "Input dict must contain 'for_target' key"
  for_observation = inp_.pop("for_observation")
  for_target = inp_.pop("for_target")
  return dict(inputs=for_observation, labels=lshift(for_target), **inp_)


def compute_token_types(inp: dict, ranges) -> dict:
  """Computes token types using a dictionary."""
  assert "inputs" in inp, "Input dict must contain 'inputs' key"
  assert "labels" in inp, "Input dict must contain 'labels' key"
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


class ComputeInputsAndLabelsTest(unittest.TestCase):
  """Tests for compute_inputs_and_labels validation."""

  def test_valid_input(self):
    """Test that valid input produces correct output."""
    inp = {
        "for_observation": np.array([1, 2, 3, 4]),
        "for_target": np.array([2, 3, 4, 5]),
    }
    result = compute_inputs_and_labels(inp)
    
    np.testing.assert_array_equal(result["inputs"], np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(result["labels"], np.array([3, 4, 5, 0]))
    
  def test_missing_for_observation_key(self):
    """Test that missing 'for_observation' key raises AssertionError."""
    inp = {
        "for_target": np.array([2, 3, 4, 5]),
    }
    with self.assertRaises(AssertionError) as context:
      compute_inputs_and_labels(inp)
    self.assertIn("for_observation", str(context.exception))

  def test_missing_for_target_key(self):
    """Test that missing 'for_target' key raises AssertionError."""
    inp = {
        "for_observation": np.array([1, 2, 3, 4]),
    }
    with self.assertRaises(AssertionError) as context:
      compute_inputs_and_labels(inp)
    self.assertIn("for_target", str(context.exception))

  def test_missing_both_keys(self):
    """Test that missing both keys raises AssertionError."""
    inp = {}
    with self.assertRaises(AssertionError) as context:
      compute_inputs_and_labels(inp)
    self.assertIn("for_observation", str(context.exception))

  def test_does_not_mutate_input(self):
    """Test that the input dict is not mutated."""
    inp = {
        "for_observation": np.array([1, 2, 3]),
        "for_target": np.array([2, 3, 4]),
        "extra_key": "extra_value",
    }
    inp_copy = inp.copy()
    compute_inputs_and_labels(inp)
    self.assertEqual(inp, inp_copy)

  def test_preserves_extra_keys(self):
    """Test that extra keys in input are preserved in output."""
    inp = {
        "for_observation": np.array([1, 2, 3]),
        "for_target": np.array([2, 3, 4]),
        "metadata": "test_data",
    }
    result = compute_inputs_and_labels(inp)
    self.assertEqual(result["metadata"], "test_data")


class ComputeTokenTypesTest(unittest.TestCase):
  """Tests for compute_token_types validation."""

  def test_valid_input_no_ranges(self):
    """Test that valid input produces correct output when ranges is None."""
    inp = {
        "inputs": np.array([1, 2, 3]),
        "labels": np.array([2, 3, 4]),
    }
    result = compute_token_types(inp, ranges=None)
    
    np.testing.assert_array_equal(result["inputs_ttypes"], np.zeros_like([1, 2, 3]))
    np.testing.assert_array_equal(result["labels_ttypes"], np.zeros_like([2, 3, 4]))

  def test_missing_inputs_key(self):
    """Test that missing 'inputs' key raises AssertionError."""
    inp = {
        "labels": np.array([2, 3, 4]),
    }
    with self.assertRaises(AssertionError) as context:
      compute_token_types(inp, ranges=None)
    self.assertIn("inputs", str(context.exception))

  def test_missing_labels_key(self):
    """Test that missing 'labels' key raises AssertionError."""
    inp = {
        "inputs": np.array([1, 2, 3]),
    }
    with self.assertRaises(AssertionError) as context:
      compute_token_types(inp, ranges=None)
    self.assertIn("labels", str(context.exception))

  def test_missing_both_keys(self):
    """Test that missing both keys raises AssertionError."""
    inp = {}
    with self.assertRaises(AssertionError) as context:
      compute_token_types(inp, ranges=None)
    self.assertIn("inputs", str(context.exception))

  def test_adds_token_type_keys(self):
    """Test that token type keys are added to output."""
    inp = {
        "inputs": np.array([1, 2, 3]),
        "labels": np.array([2, 3, 4]),
    }
    result = compute_token_types(inp, ranges=None)
    
    self.assertIn("inputs_ttypes", result)
    self.assertIn("labels_ttypes", result)


if __name__ == "__main__":
  unittest.main()
