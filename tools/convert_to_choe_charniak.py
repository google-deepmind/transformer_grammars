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

"""Converts a file from tree representation to Choe-Charniak.

For example:
  (S (NP the hungry cat) (VP meows))       (+ possibly pre-terminals)
is converted to
  (S (NP the hungry cat NP) (VP meows VP) S)
"""

from typing import Sequence

from absl import app
from absl import flags
from transformer_grammars.data import text_processing

_INPUT = flags.DEFINE_string("input", None, "Input file")
_OUTPUT = flags.DEFINE_string("output", None, "Output file")
_HAS_PRETERMS = flags.DEFINE_bool(
    "has_preterms",
    True,
    "Whether the input file contains preterminals (POS tags)",
)
_USE_UNTYPED_CLOSING_TERMINALS = flags.DEFINE_bool(
    "use_untyped_closing_terminals",
    False,
    (
        "Whether the output file should have typed closing non-terminals (e.g. "
        "S), NP), etc.) or a single untyped closing non-terminal X)"
    ),
)


def main(argv: Sequence[str]) -> None:
  del argv
  text_processing.convert_to_choe_charniak(
      _INPUT.value,
      _OUTPUT.value,
      has_preterms=_HAS_PRETERMS.value,
      untyped_closing_terminal=_USE_UNTYPED_CLOSING_TERMINALS.value,
  )


if __name__ == "__main__":
  app.run(main)
