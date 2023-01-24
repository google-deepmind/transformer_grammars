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

"""Main training code."""

import functools
import json
import pickle
from typing import Any, Mapping

from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import more_itertools
import optax
import tensorflow_datasets as tfds
from transformer_grammars import common
from transformer_grammars.data import preprocessing
from transformer_grammars.data import sp_utils
from transformer_grammars.data import text_dataset
from transformer_grammars.models import lr_schedules
from transformer_grammars.models.masking import utils as masking_utils
from transformer_grammars.training import checkpoint


def _get_first(tree):
  return jax.tree_map(lambda arr: jax.device_get(arr[0]), tree)


def _replicate_to_local_devices(tree):
  return jax.tree_map(
      lambda arr: jax.device_put_replicated(arr, jax.local_devices()), tree
  )


def _build_from_cfg(module, cfg):
  builder = getattr(module, cfg.name)
  return builder(**cfg.kwargs)


def _build_dataset_instance(ctor_name, kwargs):
  if ctor_name == "PreEncodedTextDataset":
    ctor = text_dataset.PreEncodedTextDataset
  else:
    raise NotImplementedError
  return ctor(**kwargs)


def _build_input(
    name,
    batch_size,
    dataset_cfg,
    maskrules,
    token_type_ranges,
    *,
    shuffle,
    shuffle_buffer,
    num_epochs,
    peekable,
    multithread,
):
  """Builds an input iterator."""
  logging.info("Building %s dataset.", name)
  num_devices = jax.device_count()
  num_local_devices = jax.local_device_count()
  global_batch_size = batch_size
  per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

  if ragged:
    raise ValueError(
        f"Global batch size {global_batch_size} must be divisible by "
        f"num devices {num_devices}"
    )

  logging.info(
      (
          "Global batch size: %d, num devices: %d, num local devices: %d, "
          "per-device batch size: %d"
      ),
      global_batch_size,
      num_devices,
      num_local_devices,
      per_device_batch_size,
  )

  ds = _build_dataset_instance(dataset_cfg.name, dataset_cfg.kwargs)
  ds = ds.raw_dataset(
      shuffle=shuffle,
      shuffle_buffer=shuffle_buffer,
      sample_without_replacement=shuffle,
      num_epochs=num_epochs,
      seed=None,
  )
  it = tfds.as_numpy(ds)
  it = preprocessing.get_chunks_from_dataset(
      it,
      maskrules,
      token_type_ranges,
      (num_local_devices, per_device_batch_size),
      multithread=multithread,
      use_monitor_thread=False,
  )
  if peekable:
    it = more_itertools.peekable(it)

  logging.info("Dataset built.")

  return it


def _build_train_input(cfg, maskrules, token_type_ranges):
  return _build_input(
      "training",
      cfg.batch_size,
      cfg.dataset,
      maskrules,
      token_type_ranges,
      shuffle=True,
      shuffle_buffer=int(2e5),
      num_epochs=None,
      peekable=True,
      multithread=True,
  )


def _build_eval_input(cfg, maskrules, token_type_ranges):
  return _build_input(
      "evaluation",
      cfg.batch_size,
      cfg.dataset,
      maskrules,
      token_type_ranges,
      shuffle=False,
      shuffle_buffer=0,
      num_epochs=1,
      peekable=False,
      multithread=False,
  )


def _load_dictionary_metadata(config):
  metadata_fname = config.dictionary_metadata_filename
  with open(metadata_fname, "r") as f:
    metadata = json.load(f)
  logging.info("Loaded dictionary metadata:\n%s", repr(metadata))
  return metadata


def _load_sentencepiece_vocab(config):
  sentencepiece_vocab_filename = config.sentencepiece_vocab_filename
  with open(sentencepiece_vocab_filename, "r") as f:
    vocab = sp_utils.SentencePieceVocab.from_vocab_file(f)
  logging.info("Loaded SentencePiece vocab:\n%s", repr(vocab))
  return vocab


def _load_token_type_ranges(config):
  """Loads token type ranges info from dictionary metadata or SP .vocab file."""
  if config.get("dictionary_metadata_filename", ""):
    dic_metadata = _load_dictionary_metadata(config)
    token_type_ranges = masking_utils.TokenTypeRanges.from_dictionary_metadata(
        **dic_metadata
    )
  elif config.get("sentencepiece_vocab_filename", ""):
    vocab = _load_sentencepiece_vocab(config)
    token_type_ranges = masking_utils.TokenTypeRanges.from_sentencepiece_vocab(
        vocab
    )
  else:
    token_type_ranges = None
  logging.info("Using token ranges:\n%s", repr(token_type_ranges))
  return token_type_ranges


def _initialize_model(model_cfg, maskrules, token_type_ranges, init_rng, batch):
  init_inputs = common.model_input_from_chunk(batch, maskrules)
  forward = common.build_forward(
      model_cfg, maskrules, token_type_ranges, is_training=True
  )
  p_init = jax.pmap(forward.init)
  params, state = p_init(init_rng, **init_inputs)
  return params, state


################################################################################
# Training
################################################################################


def _loss(apply, maskrules, vocab_size, params, state, rng, batch):
  """Computes the loss."""
  inputs = common.model_input_from_chunk(batch, maskrules)
  logits, state = apply(params, state, rng=rng, **inputs)
  mask = jnp.logical_and(
      jnp.greater(batch.labels, 0), jnp.greater(batch.seq_idx, -1)[:, None]
  ).astype(jnp.int32)
  labels_one_hot = hk.one_hot(batch.labels, vocab_size)
  loss = optax.softmax_cross_entropy(logits, labels_one_hot)
  total_loss = jnp.sum(mask * loss)
  total_count = jnp.sum(mask)
  # Compute the average loss per-token for the batches received on each device
  # independently, then use that to get the per-device gradient, then average
  # those. This is fine here, as batches on each device roughly have the same
  # number of non-masked tokens.
  loss = total_loss / total_count
  scaled_loss = loss / jax.device_count()  # For gradients, to avoid a pmean.
  aux = (state, (loss, total_loss, total_count))
  return scaled_loss, aux


def _learning_rate(cfg, step):
  return _build_from_cfg(lr_schedules, cfg)(step)


def _optimizer(cfg, learning_rate):
  optimizer = getattr(optax, cfg.name)
  return optimizer(learning_rate, **cfg.kwargs)


@chex.dataclass
class TrainingState:
  rng: jnp.array
  step: jnp.array
  params: Any
  state: Any
  opt_state: Any


def _build_update(config, maskrules, token_type_ranges):
  """Builds the training state update function."""

  forward = common.build_forward(
      config.model, maskrules, token_type_ranges, is_training=True
  )

  def update(training_state, batch):
    """Updates the training state from a batch of data."""
    loss_fn = functools.partial(
        _loss, forward.apply, maskrules, token_type_ranges.vocab_size
    )
    grad_loss_fn = jax.grad(loss_fn, has_aux=True)
    scaled_grads, (state, (loss, *_)) = grad_loss_fn(
        training_state.params,
        training_state.state,
        training_state.rng,
        batch,
    )
    grads = jax.lax.psum(scaled_grads, axis_name="i")

    # Clip gradients
    grad_norm = optax.global_norm(grads)
    assert not grad_norm.shape
    clip_grad_norm = config.training.get("clip_grad_norm", 0.0)
    if clip_grad_norm:
      # Implement our own gradient clipping (by norm) as optax's doesn't handle
      # gradients with 0 norm (which shouldn't happen, but still.)
      clipping_factor = jnp.minimum(1.0, clip_grad_norm / (grad_norm + 1e-6))
      clipped_grads = jax.tree_util.tree_map(
          lambda t: t * clipping_factor, grads
      )
    else:
      clipped_grads = grads
    clipped_grad_norm = optax.global_norm(clipped_grads)

    # Compute and apply updates via our optimizer.
    learning_rate = _learning_rate(
        config.training.lr_schedule, training_state.step
    )
    _, opt_update = _optimizer(config.training.optimizer, learning_rate)
    updates, opt_state = opt_update(grads, training_state.opt_state)
    params = optax.apply_updates(training_state.params, updates)

    # Compute norms.
    params_norm = optax.global_norm(params)
    update_norm = optax.global_norm(updates)

    mask = jnp.greater(batch.inputs, 0).astype(jnp.int32)
    indic_mean = jnp.sum(batch.attn_indicator * mask) / jnp.sum(mask)

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = {
        "loss": loss,
        "learning_rate": learning_rate,
        "params_norm": params_norm,
        "update_norm": update_norm,
        "unclipped_grad_norm": grad_norm,
        "clipped_grad_norm": clipped_grad_norm,
        "attn_indicator_mean": indic_mean,
        "tokens_per_batch": jnp.sum(mask),
    }
    # These should be summed, not averaged, across devices.
    scalars["tokens_per_batch"] *= jax.device_count()
    scalars = jax.lax.pmean(scalars, axis_name="i")

    step = training_state.step + 1
    rng, _ = jax.random.split(training_state.rng)

    new_training_state = TrainingState(
        rng=rng, step=step, params=params, state=state, opt_state=opt_state
    )

    return new_training_state, scalars

  return update


################################################################################
# Evaluation
################################################################################


def _build_evaluator(eval_cfg, model_cfg, maskrules, token_type_ranges):
  """Builds the evaluator function."""
  apply = common.build_forward(
      model_cfg, maskrules, token_type_ranges, is_training=False
  ).apply

  def eval_batch(params, state, batch):
    rng = None
    _, aux = _loss(
        apply,
        maskrules,
        token_type_ranges.vocab_size,
        params,
        state,
        rng,
        batch,
    )
    state, (_, total_loss, total_count) = aux
    total_loss = jax.lax.psum(total_loss, axis_name="i")
    total_count = jax.lax.psum(total_count, axis_name="i")
    return state, (total_loss, total_count)

  p_eval_batch = jax.pmap(eval_batch, axis_name="i")
  ds = _build_eval_input(eval_cfg, maskrules, token_type_ranges)
  ds = more_itertools.seekable(ds)

  def eval_epoch(py_step, training_state):
    logging.info("Evaluating at step %d.", py_step)
    params = training_state.params
    state = None
    total_loss = 0.0
    total_count = 0.0

    for batch in ds:
      state, batch_metrics = p_eval_batch(params, state, batch)
      batch_metrics = _get_first(batch_metrics)
      total_loss += batch_metrics[0]
      total_count += batch_metrics[1]
    logging.info(
        "[eval % 10d] total_loss=%s\ttotal_count=%d",
        py_step,
        total_loss,
        total_count,
    )
    ds.seek(0)  # Reset the evaluation dataset without recreating it.

  return eval_epoch


################################################################################
# Main loop
################################################################################


def _split_init_and_train_rngs(seed, _):
  orig_rng = jax.random.PRNGKey(seed)
  init_rng, train_rng = jax.random.split(orig_rng)
  train_rng = jax.random.fold_in(train_rng, jax.lax.axis_index("i"))
  return init_rng, train_rng


def _should_do(cfg, py_step):
  return py_step % cfg.interval_steps == 0


def _log(unused_cfg, py_step, metrics):
  metrics_str = "\t".join(
      (
          f"{k}={_get_first(v)!s}" for (k, v) in metrics.items()
      )
  )
  logging.info("[train % 9d] %s", py_step, metrics_str)


def _save_checkpoint(unused_cfg, py_step, training_state, model_cfg):
  logging.info("Saving checkpoint at step %d.", py_step)
  params = _get_first(training_state.params)
  opt_state = _get_first(training_state.opt_state)
  ckpt = checkpoint.Checkpoint(
      step=py_step,
      params=params,
      opt_state=opt_state,
      config=model_cfg.to_dict(),
  )
  with open("checkpoint.pkl", "wb") as f:
    pickle.dump(ckpt, f)


def _reload_from_checkpoint(_, current_state):
  ckpt = checkpoint.load_checkpoint("checkpoint.pkl")
  params = _replicate_to_local_devices(ckpt.params)
  opt_state = _replicate_to_local_devices(ckpt.opt_state)
  py_step = ckpt.step
  jax_step = _replicate_to_local_devices(jnp.array(py_step, dtype=jnp.int32))
  training_state = current_state.replace(
      step=jax_step, params=params, opt_state=opt_state
  )
  return training_state, py_step


def _log_shapes(mapping, prefix=""):
  for k, v in mapping.items():
    key = f"{prefix}/{k}" if prefix else k
    if isinstance(v, Mapping):
      _log_shapes(v, prefix=key)
    elif isinstance(v, tuple):
      for i, elem in enumerate(v):
        _log_shapes(elem, prefix=key + f"[{i}]")
    else:
      logging.info("\t%s: %s", key, repr(v.shape[1:]))


def main(config, _):
  """Simultaneous train+eval loop."""

  jnp.set_printoptions(precision=4)

  # Load the config
  config = config.value

  # Checks.
  if jax.local_device_count() < jax.device_count():
    raise RuntimeError("Multiple processes (hosts) training is not supported.")

  # Setup the RNGs.
  dummy_input = jax.device_put_replicated(jnp.zeros(()), jax.local_devices())
  init_rng, train_rng = jax.pmap(
      functools.partial(_split_init_and_train_rngs, 0), axis_name="i"
  )(
      dummy_input  # Dummy input, unused.
  )

  # Load token type ranges.
  token_type_ranges = _load_token_type_ranges(config)

  # Load masking rules.
  # Because these carry properties that the model core needs to know about,
  # build them early.
  maskrules = common.build_maskrules(config.model)

  # Create the training dataset.
  ds = _build_train_input(config.training, maskrules, token_type_ranges)
  first_batch = ds.peek()

  # Create the update function.
  p_update = jax.pmap(
      _build_update(config, maskrules, token_type_ranges), axis_name="i"
  )

  # Create the evaluator.
  evaluator = _build_evaluator(
      config.evaluation, config.model, maskrules, token_type_ranges
  )

  # Initialize the training state.
  params, state = _initialize_model(
      config.model, maskrules, token_type_ranges, init_rng, first_batch
  )
  opt_init, _ = _optimizer(config.training.optimizer, 0.0)
  opt_state = jax.pmap(opt_init)(params)
  step = _replicate_to_local_devices(jnp.zeros((), dtype=jnp.int32))
  training_state = TrainingState(
      rng=train_rng,
      step=step,
      params=params,
      state=state,
      opt_state=opt_state,
  )
  # Keep a Python and a JAX (on-device) copy of the current step to avoid
  # transfers.
  py_step = 0

  logging.info("Parameters shapes:")
  _log_shapes(training_state.params)

  # Possibly overwrite it from a checkpoint (except for the RNG)
  try:
    training_state, py_step = _reload_from_checkpoint(None, training_state)
  except checkpoint.CheckpointLoadingError:
    logging.warning(
        "No checkpoint found, or unusable -- starting from scratch."
    )
  else:
    logging.warning("Checkpoint found -- restarting from step %d.", py_step)

  # Training loop.
  logging.info("Starting training.")
  while py_step < config.training.num_steps:
    training_state, metrics = p_update(training_state, next(ds))
    py_step += 1

    last = py_step == config.training.num_steps
    if last or _should_do(config.logging, py_step):
      _log(config.logging, py_step, metrics)

    if last or _should_do(config.checkpointing, py_step):
      _save_checkpoint(
          config.checkpointing, py_step, training_state, config.model
      )

    if last or _should_do(config.evaluation, py_step):
      evaluator(py_step, training_state)

  logging.info("Training complete.")
