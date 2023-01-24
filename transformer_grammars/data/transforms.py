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

"""Tree transformations."""

import nltk

from transformer_grammars.data import constants


def tree_from_string(s):
  return nltk.Tree.fromstring(s)


def string_from_tree(tree):
  return tree._pformat_flat("", "()", False)  # pylint: disable=protected-access


def get_terminals(tree):
  """Returns the terminals in a tree."""
  for node in tree:
    if isinstance(node, str):
      yield node
    else:
      yield from get_terminals(node)


def get_inode_labels(tree):
  """Get labels of non-terminals."""
  if isinstance(tree, str):
    pass
  else:
    yield tree.label()
    for node in tree:
      yield from get_inode_labels(node)


def reverse(tree):
  if isinstance(tree, str):
    return tree
  else:
    nodes = [reverse(node) for node in reversed(list(tree))]
    return nltk.Tree(tree.label(), nodes)


def replace_leaves(tree, leaves):
  it = iter(leaves)
  if isinstance(tree, str):
    return next(it)
  else:
    new_nodes = [replace_leaves(node, it) for node in tree]
    return nltk.Tree(tree.label(), new_nodes)


def reverse_structure(tree):
  rev_tree = reverse(tree)
  terminals = get_terminals(tree)
  return replace_leaves(rev_tree, terminals)


def drop_pos_tags(tree):
  if isinstance(tree, str):
    return tree
  if len(tree) == 1 and isinstance(tree[0], str):
    return tree[0]
  return nltk.Tree(tree.label(), [drop_pos_tags(node) for node in tree])


def anonymize_pos_tags(tree):
  if isinstance(tree, str):
    return tree
  if len(tree) == 1 and isinstance(tree[0], str):
    return nltk.Tree("XX", [tree[0]])
  return nltk.Tree(tree.label(), [anonymize_pos_tags(node) for node in tree])


def _make_left_or_right_branching_binary(labels, leaves_it, leftwards=True):
  """Makes a left/right-branching binary tree with given labels and leaves."""
  if not labels:
    return next(leaves_it)
  if len(labels) == 1:
    return nltk.Tree(labels[0], [next(leaves_it), next(leaves_it)])
  if leftwards:
    subtree = _make_left_or_right_branching_binary(
        labels[1:], leaves_it, leftwards
    )
    right_leaf = next(leaves_it)
    return nltk.Tree(labels[0], [subtree, right_leaf])
  else:
    left_leaf = next(leaves_it)  # Get the left leaf before recursing.
    subtree = _make_left_or_right_branching_binary(
        labels[1:], leaves_it, leftwards
    )
    return nltk.Tree(labels[0], [left_leaf, subtree])


def make_left_or_right_branching(tree, leftwards):
  """Converts a tree to left-branching (or right-) + trail of words."""
  labels = list(get_inode_labels(tree))
  leaves = list(get_terminals(tree))
  if len(labels) + 1 > len(leaves):
    # Set the extra labels, which can't be used for the binary tree, aside.
    if len(leaves) == 1:
      # Then labels[:-0] doesn't do what we want, which is selecting the whole
      # string.
      extra_labels = labels
      binary_tree_labels = []
    else:
      extra_labels = labels[: -len(leaves) + 1]
      binary_tree_labels = labels[-len(leaves) + 1 :]
  else:
    extra_labels = []
    binary_tree_labels = labels
  num_leaves_in_binary_tree = len(binary_tree_labels) + 1
  # Some leaves of the tree are going to be part of the binary tree proper, some
  # are going to be part of the trail of leaves either on the left or on the
  # right, attached to the root.
  if leftwards:
    binary_tree_leaves_it = iter(leaves[:num_leaves_in_binary_tree])
    remaining_leaves = leaves[num_leaves_in_binary_tree:]
  else:
    binary_tree_leaves_it = iter(leaves[-num_leaves_in_binary_tree:])
    remaining_leaves = leaves[:-num_leaves_in_binary_tree]
  new_tree = _make_left_or_right_branching_binary(
      binary_tree_labels, binary_tree_leaves_it, leftwards
  )
  if leftwards:
    for leaf in remaining_leaves:
      new_tree.append(leaf)
  else:
    for leaf in reversed(remaining_leaves):
      new_tree.insert(0, leaf)
  for label in reversed(extra_labels):
    new_tree = nltk.Tree(label, [new_tree])
  assert list(get_terminals(new_tree)) == leaves
  assert list(get_inode_labels(new_tree)) == labels
  return new_tree


def make_left_branching(tree):
  return make_left_or_right_branching(tree, leftwards=True)


def make_right_branching(tree):
  return make_left_or_right_branching(tree, leftwards=False)


def transform_sentence(tree, mode):
  """Transforms a tree corresponding to a sentence."""
  if mode == constants.TreeTransform.NONE:
    return tree
  elif mode == constants.TreeTransform.REVERSE:
    return reverse_structure(tree)
  elif mode == constants.TreeTransform.LEFT_BRANCHING:
    return make_left_branching(tree)
  elif mode == constants.TreeTransform.RIGHT_BRANCHING:
    return make_right_branching(tree)
  else:
    raise NotImplementedError
