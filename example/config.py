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

"""Config for language modelling on example data.

This is an example to quickly try the training code -- it does not train the
model to convergence.
"""

import ml_collections as dm_collections


def get_config():
  """Config for training."""

  return dm_collections.ConfigDict(
      dict(
          sentencepiece_vocab_filename="spm/spm.vocab",
          # dictionary_metadata_filename='word_based.json',
          training=dict(
              num_steps=1_000,  # Only 1k steps for the example.
              clip_grad_norm=3.0,
              batch_size=64,
              dataset=dict(
                  name="PreEncodedTextDataset",
                  kwargs=dict(
                      filename="data/train.csv",
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
              d_model=512,
              num_layers=6,
              num_heads=8,
              ffw_hidden_size=2048,
              embedding_dropout=0.1,
              core_dropout=0.1,
              core_output_dropout=0.1,
              sequence_length=256,
              memory_length=256,
              tied_input_output_embeddings=True,
              relative_position_embeddings=1,
              tied_layer_weights=0,
              # TG settings
              extra_attention_mask_name="stack_compose_double_closing_nt",
              extra_attention_mask_kwargs=dict(
                  relative_pos="delta_depth",
                  # Do not use different STACK/COMPOSE attention weights.
                  use_different_attn_fns=0,
                  # Transparency probability
                  transparency_prob=0.0,
                  # Smart memory
                  gather_into_new_memory=1,
                  # Depth below or at which the node is transparent
                  # -1 means that it's never transparent.
                  # <s> has depth 0, (DOC depth 1, so for the top level (S
                  # to be transparent, we need this to be set to 2
                  transparency_depth_threshold=-1,
              ),
              min_relative_position=-1,
              max_relative_position=62,  # So 64 positions possible
              # Layer-hybrid
              num_unrestricted_layers=0,
              # Head-hybrid
              num_unrestricted_heads=None,
          ),
          evaluation=dict(
              interval_steps=500,
              batch_size=8,
              sequence_length=256,
              dataset=dict(
                  name="PreEncodedTextDataset",
                  kwargs=dict(
                      filename="data/valid.csv",
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
