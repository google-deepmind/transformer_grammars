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

"""Example config. Copy into a new file for your training run.

In this config, no TG-specific option is enabled, and the model is effectively
a Transformer-XL.

Remember to set the correct training and validation filenames in the config as
required (search for <<<).
"""

import ml_collections as dm_collections


def get_config(debug=False):
  """Return config object for training."""

  def m(default_value, debug_value):
    """Switcher that returns the default or debug value based on debug flag."""
    return debug_value if debug else default_value

  return dm_collections.ConfigDict(
      dict(
          sentencepiece_vocab_filename="<<< SP .vocab filename >>>",
          training=dict(
              # Number of training steps (i.e. number of gradient updates,
              # i.e. number of training batches drawn)
              num_steps=100_000,
              # Gradient clipping
              clip_grad_norm=3.0,
              # Global (not per-device) batch size
              batch_size=64,
              dataset=dict(
                  name="PreEncodedTextDataset",
                  kwargs=dict(
                      filename="<<< training .csv filename >>>",
                      num_samples=None,
                      add_bos=True,
                      add_eos=False,  # Not needed for Choe-Charniak.
                  ),
              ),
              optimizer=dict(
                  name="adam",
                  kwargs=dict(
                      b1=0.9,
                      b2=0.999,
                  ),
              ),
              # Learning rate schedule.
              lr_schedule=dict(
                  name="linear_warmup_then_cosine_anneal",
                  kwargs=dict(
                      start_lr=1e-7,
                      min_lr=3e-7,
                      max_lr=1.5e-4,
                      warmup_steps=8000,
                      cosine_cycle_length=100_000,
                  ),
              ),
          ),
          model=dict(
              vocab_size=32768,
              d_model=m(1024, 8),
              num_layers=m(16, 1),
              num_heads=m(8, 8),
              ffw_hidden_size=m(4096, 8),
              embedding_dropout=0.1,
              core_dropout=0.1,
              core_output_dropout=0.1,
              sequence_length=m(256, 128),
              memory_length=m(256, 128),
              tied_input_output_embeddings=True,
              relative_position_embeddings=1,
              tied_layer_weights=0,
          ),
          evaluation=dict(
              interval_steps=500,
              batch_size=8,
              sequence_length=256,
              dataset=dict(
                  name="PreEncodedTextDataset",
                  kwargs=dict(
                      filename="<<< validation .csv filename >>>",
                      num_samples=None,
                      add_bos=True,
                      add_eos=False,  # Not needed for Choe-Charniak.
                  ),
              ),
          ),
          logging=dict(
              interval_steps=10,
          ),
          checkpointing=dict(
              interval_steps=500,
          ),
      )
  )
