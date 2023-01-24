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

"""Transforms the input structures.

This was used for our control experiments, and is not required otherwise.

3 modes are supported:

- reverse:
  maps (A (B x) y) to (A x (B y))

- left-branching:
  creates a left-branching binary tree with the labels of the input structure,
  then attaches the extra terminals to the root node

- right-branching:
  same thing, with a right-branching binary tree

NOTE: all of these operations are applied at the sentence-level.

"""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import nltk
import tqdm
from transformer_grammars.data import constants
from transformer_grammars.data import transforms


_VALID_MODES = ("reverse", "lb", "rb")

_INPUT = flags.DEFINE_string("input", None, "Input file")
_OUTPUT = flags.DEFINE_string("output", None, "Output file")
_HAS_PRETERMS = flags.DEFINE_bool(
    "has_preterms",
    True,
    "Whether the input file contains preterminals (POS tags)",
)
_DOC_LEVEL = flags.DEFINE_bool(
    "document_level",
    False,
    "Whether the input file contains documents (DOC ...)",
)
_MODE = flags.DEFINE_enum(
    "mode",
    None,
    _VALID_MODES,
    "How the trees should be transformed. Must be one of reverse, lb, rb.",
)


def _transform_tree(tree, mode, has_preterms, doc_level):
  """Transforms a tree."""
  assert mode in _VALID_MODES
  if has_preterms:
    tree = transforms.drop_pos_tags(tree)
  if doc_level:
    if tree.label() != "DOC":
      raise RuntimeError(
          "The label of the root node is %s, where DOC was expected."
          % tree.label()
      )
    transformed_sentences = [
        _transform_tree(sent, mode, False, False) for sent in tree
    ]
    return nltk.Tree("DOC", transformed_sentences)
  else:
    return transforms.transform_sentence(tree, constants.TreeTransform(mode))


def main(argv: Sequence[str]) -> None:
  del argv

  input_fname = _INPUT.value
  output_fname = _OUTPUT.value
  mode = _MODE.value
  has_preterms = _HAS_PRETERMS.value
  doc_level = _DOC_LEVEL.value
  logging.info("Input file: %s", input_fname)
  logging.info("Output file: %s", output_fname)
  logging.info("Mode: %s", mode)
  logging.info("Has preterminals: %s", str(has_preterms))
  logging.info("Document level: %s", str(doc_level))

  with open(input_fname, "r") as in_f:
    with open(output_fname, "w") as out_f:
      for line in tqdm.tqdm(in_f):
        tree = transforms.tree_from_string(line)
        transformed_tree = _transform_tree(tree, mode, has_preterms, doc_level)
        transformed_line = transforms.string_from_tree(transformed_tree) + "\n"
        out_f.write(transformed_line)


if __name__ == "__main__":
  app.run(main)
