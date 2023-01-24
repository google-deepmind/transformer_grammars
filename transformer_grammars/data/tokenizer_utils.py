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

"""Utils to load word-based or SentencePiece vocabs."""

import json
import os.path
from absl import logging

from transformer_grammars.data import dictionary
from transformer_grammars.data import sp_utils
from transformer_grammars.models.masking import utils as masking_utils


def _read_dictionary(dictionary_fname: str):
  dic = dictionary.Dict()
  with open(dictionary_fname, "r") as f:
    dic.load_from_file(f)
  dic.freeze()
  return dic


def get_dictionary_and_ranges(fname):
  """Returns dictionary and token ranges from a dictionary (or SPM) file."""

  def read_token_types(vocab_fname):
    with open(vocab_fname, "r") as f:
      vocab = sp_utils.SentencePieceVocab.from_vocab_file(f)
    token_type_ranges = masking_utils.TokenTypeRanges.from_sentencepiece_vocab(
        vocab
    )
    return vocab.dictionary, token_type_ranges

  if fname.endswith(".model"):
    vocab_fname = os.path.splitext(fname)[0] + ".vocab"
    return read_token_types(vocab_fname)
  elif fname.endswith(".vocab"):
    logging.warning(
        "get_dictionary_and_ranges should be called with the .model file"
    )
    return read_token_types(fname)
  else:
    dic = _read_dictionary(fname)
    # Load dictionary metadata
    dic_metadata_path = os.path.splitext(fname)[0] + ".json"
    with open(dic_metadata_path) as f:
      dic_metadata = json.load(f)
    token_type_ranges = masking_utils.TokenTypeRanges.from_dictionary_metadata(
        **dic_metadata
    )

  logging.info("Using token ranges:\n%s", repr(token_type_ranges))

  return dic, token_type_ranges
