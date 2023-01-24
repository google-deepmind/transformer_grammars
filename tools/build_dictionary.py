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

"""Build a dictionary from a list of whitespace-separated tokens."""

import itertools
import json
from absl import app
from absl import flags
from absl import logging
from transformer_grammars.data import constants
from transformer_grammars.data import text_processing


_INPUT_FNAME = flags.DEFINE_string("input", None, "Input filename.")
_OUTPUT_PREFIX = flags.DEFINE_string(
    "output", None, "Output prefix for dictionary."
)
_CONVERT_TO_CC = flags.DEFINE_bool(
    "convert_to_choe_charniak",
    False,
    (
        "Whether the input should be converted to Choe-Charniak before being"
        " processed further."
    ),
)
_USE_EXTRA_UNTYPED_CLOSING_NON_TERMINAL = flags.DEFINE_bool(
    "use_extra_untyped_closing_non_terminal",
    False,
    (
        "Whether the learnt dictionary should include an extra untyped closing"
        " non-terminal."
    ),
)


def main(unused_argv):
  # We accumulate terminals and non-terminals separately so that we can assign
  # to each group a consecutive range of IDs.
  terminals = set()
  opening_non_terminals = set()
  closing_non_terminals = set()

  with open(_INPUT_FNAME.value, "r") as in_f:
    for l in in_f:
      l = l.strip()
      if _CONVERT_TO_CC.value:
        l = text_processing.choe_charniak_from_tree(l)
      for word in l.split(" "):
        if word == constants.PLACEHOLDER_TOKEN:
          continue
        elif word in constants.RESERVED_WORDS:
          raise ValueError(
              f"Cannot encode word {word} as it is a reserved word."
          )
        elif constants.OPENING_NON_TERMINAL_REGEXP.match(word):
          if word not in opening_non_terminals:
            logging.info("Found ONT: %s", word)
          opening_non_terminals.add(word)
        elif constants.CLOSING_NON_TERMINAL_REGEXP.match(word):
          if word not in closing_non_terminals:
            logging.info("Found CNT: %s", word)
          closing_non_terminals.add(word)
        else:
          terminals.add(word)

  num_reserved = len(constants.RESERVED_WORDS)
  num_terminals = len(terminals)
  num_opening_non_terminals = len(opening_non_terminals)
  num_closing_non_terminals = len(closing_non_terminals)

  start_idx = 0
  for name, num in [
      ("reserved tokens", num_reserved),
      ("terminals", num_terminals),
      ("opening non terminals", num_opening_non_terminals),
      ("closing non terminals", num_closing_non_terminals),
  ]:
    end_idx = start_idx + num
    logging.info("%d %s, %d ≤ token_id < %d", num, name, start_idx, end_idx)
    start_idx = end_idx

  if num_closing_non_terminals != num_opening_non_terminals:
    raise RuntimeError(
        f"The number of opening non-terminals ({num_opening_non_terminals}) "
        "does not match the number of closing non-terminals "
        f"({num_closing_non_terminals}).")

  if (
      num_opening_non_terminals > 0
      and _USE_EXTRA_UNTYPED_CLOSING_NON_TERMINAL.value
  ):
    logging.info(
        "Input has non-terminals tokens (Choe-Charniak representation)"
        ' so adding one extra untyped closing non-terminal ")".'
    )
    extra_untyped_closing_non_terminal = True
    untyped_closing_non_terminals = [constants.UNTYPED_CLOSING_NON_TERMINAL]
  else:
    extra_untyped_closing_non_terminal = False
    untyped_closing_non_terminals = []

  dic_metadata = dict(
      num_reserved=num_reserved,
      num_terminals=num_terminals,
      # We write the number of opening and closing non-terminals independently,
      # to avoid being confusing with a single `num_non_terminals` which can be
      # understood as either the number of non-terminals of each type, or the
      # total number of non-terminals of both types.
      num_opening_non_terminals=num_opening_non_terminals,
      num_closing_non_terminals=num_closing_non_terminals,
      extra_untyped_closing_non_terminal=extra_untyped_closing_non_terminal,
  )

  dic_fname = _OUTPUT_PREFIX.value + ".txt"
  dic_metadata_fname = _OUTPUT_PREFIX.value + ".json"

  with open(dic_fname, "w") as out_f:
    for w in itertools.chain(
        constants.RESERVED_WORDS,
        sorted(terminals),
        sorted(opening_non_terminals),
        sorted(closing_non_terminals),
        untyped_closing_non_terminals,
    ):
      out_f.write(w + "\n")

  with open(dic_metadata_fname, "w") as metadata_f:
    json.dump(dic_metadata, metadata_f, indent=4)


if __name__ == "__main__":
  app.run(main)
