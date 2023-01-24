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

"""Checkpointing."""

import pickle
from typing import Any
from absl import logging
import chex


@chex.dataclass
class Checkpoint:
  step: int
  params: Any
  opt_state: Any
  config: Any


class CheckpointLoadingError(Exception):
  pass


def load_checkpoint(fname) -> Checkpoint:
  """Loads a checkpoint from a file."""
  try:
    with open(fname, "rb") as f:
      return pickle.load(f)
  except Exception as ex:
    logging.info("Exception %s raised when loading checkpoint", str(ex))
    raise CheckpointLoadingError from ex
