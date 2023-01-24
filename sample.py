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

"""Sample from a trained model."""

# This needs to be put first -- it prevents TF from allocating GPU memory.
import os
os.environ["TF_ENABLED_DEVICE_TYPES"] = "CPU"

# pylint: disable=g-import-not-at-top,g-bad-import-order
import functools
from absl import app
from absl import flags
from transformer_grammars import sample


_CHECKPOINT = flags.DEFINE_string("checkpoint", None, "Checkpoint to load.")
_INPUT = flags.DEFINE_string(
    "input", None, "File containing sequences to score, as tokenized TSV files."
)
_TOKENIZER = flags.DEFINE_string("tokenizer", None, "Tokenizer.")
_SEED = flags.DEFINE_integer("seed", 42, "Sampling PRNG seed.")
_TEMPERATURE = flags.DEFINE_float("temperature", 0.7, "Sampling temperature.")
_NUM_STEPS = flags.DEFINE_integer("num_stemps", 300, "Sampling steps.")


if __name__ == "__main__":
  app.run(
      functools.partial(
          sample.main,
          _TOKENIZER,
          _CHECKPOINT,
          _INPUT,
          _SEED,
          _TEMPERATURE,
          _NUM_STEPS
      )
  )
