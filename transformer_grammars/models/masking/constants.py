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

"""Constants for masking code."""

import enum

PAD = 0
SOS = 1
OPENING_NT = 2
TERMINAL = 3
CLOSING_NT = 4
PLACEHOLDER = 5
TOKEN_TYPES = [PAD, SOS, OPENING_NT, TERMINAL, CLOSING_NT]


class TokenTypesEnum(enum.IntEnum):
  PAD = PAD
  SOS = SOS
  OPENING_NT = OPENING_NT
  TERMINAL = TERMINAL
  CLOSING_NT = CLOSING_NT
  PLACEHOLDER = PLACEHOLDER


# For proposals involving duplicating tokens.
CLOSING_NT_2 = 14
