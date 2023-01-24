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

"""Train a TG / TXL (trees) / TXL (words) model."""


# This needs to be put first -- it prevents TF from allocating GPU memory.
import os
os.environ["TF_ENABLED_DEVICE_TYPES"] = "CPU"

# pylint: disable=g-import-not-at-top,g-bad-import-order
import functools
from absl import app
from absl import flags
from ml_collections import config_flags
from transformer_grammars.training import train


_CONFIG = config_flags.DEFINE_config_file("config")

if __name__ == "__main__":
  flags.mark_flag_as_required("config")
  app.run(functools.partial(train.main, _CONFIG))
