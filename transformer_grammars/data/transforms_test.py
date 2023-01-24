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

"""Tests for transformer_grammars.data.transforms."""

import unittest
import nltk
from transformer_grammars.data import transforms

_HUNGRY_CAT = nltk.Tree.fromstring("(S (NP the hungry cat) (VP meows))")
_LOUD_HUNGRY_CAT = nltk.Tree.fromstring(
    "(S (NP the hungry cat) (VP meows loudly))"
)


class TransformsTest(unittest.TestCase):

  def test_reverse_structure(self):
    self.assertEqual(
        transforms.reverse_structure(nltk.Tree.fromstring("(A x (B y))")),
        nltk.Tree.fromstring("(A (B x) y)"),
    )
    self.assertEqual(
        transforms.reverse_structure(nltk.Tree.fromstring("(A x (B y (C z)))")),
        nltk.Tree.fromstring("(A (B (C x) y) z)"),
    )

  def test_drop_pos_tags(self):
    self.assertEqual(
        transforms.drop_pos_tags(nltk.Tree.fromstring("(A (B x))")),
        nltk.Tree.fromstring("(A x)"),
    )

  def test_get_terminals(self):
    self.assertEqual(
        list(transforms.get_terminals(_HUNGRY_CAT)),
        ["the", "hungry", "cat", "meows"],
    )

  def test_make_left_branching(self):
    self.assertEqual(
        transforms.make_left_branching(_HUNGRY_CAT),
        nltk.Tree.fromstring("(S (NP (VP the hungry) cat) meows)"),
    )
    self.assertEqual(
        transforms.make_left_branching(_LOUD_HUNGRY_CAT),
        nltk.Tree.fromstring("(S (NP (VP the hungry) cat) meows loudly)"),
    )

  def test_make_right_branching(self):
    self.assertEqual(
        transforms.make_right_branching(_HUNGRY_CAT),
        nltk.Tree.fromstring("(S the (NP hungry (VP cat meows)))"),
    )
    self.assertEqual(
        transforms.make_right_branching(_LOUD_HUNGRY_CAT),
        nltk.Tree.fromstring("(S the hungry (NP cat (VP meows loudly)))"),
    )
