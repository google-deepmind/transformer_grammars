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

# pylint: disable=line-too-long
r"""Postprocess documents after encoding with spm_encode.

After encoding with SentencePiece, documents represented as Choe-Charniak
strings contain a word piece for whitespace before and after non-terminals, e.g.

(DOC (S (NP the hungry cat NP) (VP meows VP) S) DOC)

is encoded into pieces as

▁  (DOC ▁  (S ▁  (NP ▁the ▁  hungry ▁  cat  ▁  NP) ▁  (VP ▁me  ow   s  ▁  VP) ▁
S) ▁  DOC)
59 7    59 19 59 12  62   59 9464   59 6104 59 39  59 25  1207 5209 63 59 52  59
46 59 34

It's not desirable to have such whitespaces after the non-terminals, as they
implicitly separate words. What we want instead is:

(DOC (S (NP ▁the ▁  hungry ▁  cat  NP) (VP ▁me  ow   s  VP) S) DOC)
7    19 12  62   59 9464   59 6104 39  25  1207 5209 63 52  46 34

(Or shall we even have (WORD ... WORD) constructs to delineate words?)

This script strips out the redundant whitespace tokens.

Usage:

  python postprocess_encoded_docs.py -- \
  --vocab model.vocab \
  --input foo.txt \
  --output foo2.txt
"""

from typing import Sequence
from absl import app
from absl import flags
from transformer_grammars.data import sp_utils
from transformer_grammars.data import text_processing


_VOCAB_FNAME = flags.DEFINE_string("vocab", None, ".vocab file to use")
_INPUT_FNAME = flags.DEFINE_string(
    "input", None, "Input file, output of spm_encode"
)
_OUTPUT_FNAME = flags.DEFINE_string("output", None, "Output file")


def process_line(l, vocab):
  """Processes a single line from the input."""
  input_ids = [int(x) for x in l.split(" ")]
  return ",".join(
      str(x) for x in text_processing.postprocess_token_ids(input_ids, vocab)
  )


def main(argv: Sequence[str]) -> None:
  del argv

  with open(_VOCAB_FNAME.value, "r") as f:
    vocab = sp_utils.SentencePieceVocab.from_vocab_file(f)

  with open(_INPUT_FNAME.value, "r") as inp:
    with open(_OUTPUT_FNAME.value, "w") as output:
      for l in inp:
        output.write(process_line(l, vocab) + "\n")


if __name__ == "__main__":
  app.run(main)
