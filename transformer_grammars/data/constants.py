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

"""Shared constants."""

import enum
import re

PLACEHOLDER_TOKEN = "<XXX>"
RESERVED_WORDS = ("<PAD>", "<s>", "</s>", PLACEHOLDER_TOKEN)
OPENING_NON_TERMINAL_REGEXP = re.compile(r"^\([^ ]+$")
CLOSING_NON_TERMINAL_REGEXP = re.compile(r"^[^ ]+\)$")
UNTYPED_CLOSING_NON_TERMINAL = ")"

PAD = 0
BOS = 1
EOS = 2
PLACEHOLDER = 3


class TreeTransform(enum.Enum):
  NONE = "none"
  REVERSE = "reverse"
  LEFT_BRANCHING = "lb"
  RIGHT_BRANCHING = "rb"
