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

"""Encode whitespace-separated tokens into integers using a dictionary.

The output is a CSV file, the rows of which contain encoded tokens for a single
sequence.
"""

from absl import app
from absl import flags
from absl import logging
from transformer_grammars.data import constants
from transformer_grammars.data import dictionary
from transformer_grammars.data import text_processing


_INPUT_FNAME = flags.DEFINE_string("input", None, "Input filename.")
_DICTIONARY_PREFIX = flags.DEFINE_string(
    "dictionary", None, "Dictionary prefix (i.e. filename w/o extension)."
)
_OUTPUT_FNAME = flags.DEFINE_string("output", None, "Output filename for IDs.")
_CONVERT_TO_CC = flags.DEFINE_bool(
    "convert_to_choe_charniak",
    False,
    (
        "Whether the input should be converted to Choe-Charniak before being"
        " processed further."
    ),
)


def _csv_from_list_of_ints(l):
  return ",".join(map(str, l))


def main(_):
  # Load the dictionary.
  dic = dictionary.Dict()
  with open(_DICTIONARY_PREFIX.value + ".txt", "r") as f:
    dic.load_from_file(f)
  dic.freeze()

  max_len = 0

  with open(_INPUT_FNAME.value, "r") as in_f:
    with open(_OUTPUT_FNAME.value, "w") as out_f:
      for l in in_f:
        l = l.strip()
        if _CONVERT_TO_CC.value:
          l = text_processing.choe_charniak_from_tree(l)
        encoded_l = []
        for word in l.split(" "):
          if (
              word != constants.PLACEHOLDER_TOKEN
              and word in constants.RESERVED_WORDS
          ):
            raise ValueError(
                "Cannot encode word %s as it is a reserved word." % word
            )
          id_ = dic[word]
          encoded_l.append(id_)
        max_len = max(max_len, len(encoded_l))
        out_f.write(_csv_from_list_of_ints(encoded_l) + "\n")

  logging.info("Maximum sequence length: %d", max_len)


if __name__ == "__main__":
  app.run(main)
