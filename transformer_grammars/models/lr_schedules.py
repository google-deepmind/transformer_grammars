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

"""Learning rate schedule functions."""

import jax.numpy as jnp
import numpy as np


def cosine_anneal(min_lr, max_lr, cosine_cycle_length):
  """Cosine annealing from max_lr to min_lr in cosine_cycle_length steps."""

  def schedule(step):
    t = jnp.minimum(step / cosine_cycle_length, 1.0)
    cosine_decay = 0.5 * (1.0 + jnp.cos(np.pi * t))
    return min_lr + (max_lr - min_lr) * cosine_decay

  return schedule


def linear_warmup(min_lr, max_lr, num_steps):
  """Linear warmup schedule from min_lr to max_lr in num_steps."""

  def schedule(step):
    step = jnp.minimum(step, num_steps)
    return min_lr + (step / num_steps) * (max_lr - min_lr)

  return schedule


def constant_lr(lr):
  """Constant learning rate."""

  def schedule(unused_step):
    del unused_step
    return lr

  return schedule


def linear_warmup_then_cosine_anneal(
    start_lr, max_lr, min_lr, warmup_steps, cosine_cycle_length
):
  """Linear warmup for warmup_steps steps followed by cosine anneal."""

  linear_schedule = linear_warmup(start_lr, max_lr, warmup_steps)
  cosine_schedule = cosine_anneal(min_lr, max_lr, cosine_cycle_length)

  def schedule(step):
    return jnp.where(
        step < warmup_steps,
        linear_schedule(step),
        cosine_schedule(step - warmup_steps),
    )

  return schedule


def inverse_sqrt(warmup_steps):
  def schedule(step):
    return 1 / jnp.sqrt(jnp.maximum(step, warmup_steps))

  return schedule
