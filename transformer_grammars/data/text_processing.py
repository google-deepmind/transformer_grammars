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

"""Text processing functions."""

from typing import Iterable, Sequence
from absl import logging
from transformer_grammars.data import sentence
from transformer_grammars.data import sp_utils


def postprocess_token_ids(
    ids: Sequence[int], vocab: sp_utils.SentencePieceVocab
) -> Iterable[int]:
  """Removes extra-whitespace from token IDs output by SentencePiece."""
  new_ids = []
  between_words = True  # We behave differently depending on whether we are
                        # between two words, or within a word.
  for id_ in ids:
    if between_words:
      if vocab.is_whitespace(id_):
        # Do nothing, stay in the same state.
        pass
      elif vocab.is_non_terminal(id_):
        # Emit the token, stay in the same state.
        new_ids.append(id_)
      elif vocab.is_terminal(id_):
        # Emit the token, but add the potentially missing whitespace.
        if vocab.is_whitespace_prefixed_terminal(id_):
          new_ids.append(id_)
        else:
          new_ids.append(vocab.whitespace)
          new_ids.append(id_)
        between_words = False
      else:
        logging.warning("Encountered token %d which is neither a terminal or a "
                        "non-terminal. UNK? Skipping.", id_)
    else:
      if vocab.is_whitespace(id_):
        new_ids.append(id_)
      elif vocab.is_terminal(id_):
        new_ids.append(id_)
      elif vocab.is_non_terminal(id_):
        if new_ids[-1] == vocab.whitespace:
          # We've already left the previous word, so emit the current token, but
          # retrospectively remove the whitespace we left.
          new_ids.pop()
        new_ids.append(id_)
        between_words = True
      else:
        logging.warning("Encountered token %d which is neither a terminal or a "
                        "non-terminal. UNK? Skipping.", id_)
  return new_ids


def choe_charniak_from_tree(
    s: str, has_preterms: bool = True, untyped_closing_terminal: bool = False
):
  """Converts a tree (as a string) to its Choe-Charniak representation."""
  sent = sentence.PhraseStructureSentence(s, has_preterms=has_preterms)
  return sent.convert_to_choe_charniak(untyped_closing_terminal)


def convert_to_choe_charniak(
    input_fname: str,
    output_fname: str,
    has_preterms: bool = True,
    untyped_closing_terminal: bool = False,
):
  """Given a PTB-style input file, linearise trees to a Choe & Charniak format.

  If the input line is a tab-separated sequence of values, the tree is assumed
  to be the last one, and the preceding values are copied unchanged to the
  output.

  Args:
    input_fname: string for the PTB-style input file name.
    output_fname: string for the PTB-style output file name.
    has_preterms: whether the input file has preterminals (POS tags).
    untyped_closing_terminal: whether the output should have untyped closing NTs
      or not.

  Returns:
    None.
  """
  with open(input_fname, "r") as input_file:
    with open(output_fname, "w+") as output_cc:
      sent_num = 0
      for line in input_file:
        line = line.rstrip()
        if "\t" in line:
          *prefix, line = line.split("\t")
        else:
          prefix = []
        choe_charniak = choe_charniak_from_tree(
            line,
            has_preterms=has_preterms,
            untyped_closing_terminal=untyped_closing_terminal,
        )
        output_line = "\t".join(prefix + [choe_charniak])
        output_cc.write(output_line + "\n")
        sent_num += 1

      logging.info("Processed %d lines from %s", sent_num, input_fname)
