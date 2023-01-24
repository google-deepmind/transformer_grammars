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

"""Defines a data structure for a sentence.

This class encapsulates various aspects of a sentence:
  i. The original string sequence,
  ii. The lowercased string sequence, and
  iii. The POS tag of each token in the sentence, and
  iv. the depth-first traversal of the tree.
UNK-ification is handled separately by the Unkifier class.
"""

from nltk.tree import Tree


class TaggedSentence(object):
  """A list of words with their POS tags."""

  def __init__(self, word_tag_pairs, words_dict, unkifier):
    """Initialize from a list of word/tag pairs.

    Args:
      word_tag_pairs: see docs for nltk.tree.pos()
      words_dict: instance of Dict
      unkifier: instance of Unkifier

    Returns:
      None

    Raises:
      No exceptions.
    """
    self.wtp = word_tag_pairs
    self._length = len(self.wtp)
    # is_test is True, because we don't want to mutate the unkifier's vocabulary
    self.unk_toks_str = [unkifier.unkify(w, True) for (w, _) in self.wtp]
    self.unk_toks = [words_dict[tok] for tok in self.unk_toks_str]

  def __len__(self):
    return self._length

  def __str__(self):
    return " ".join([w + "/" + t for (w, t) in self.wtp])


class PhraseStructureSentence(object):
  """The Sentence class encapsulates various useful aspects of a sentence."""

  def __init__(self, sent_line, has_preterms=True):
    """Initialize the Sentence instance from its phrase-structure trees.

    UNK-ification is handled at the TopDownRnngOracle class and not here.

    Args:
      sent_line: tree string for this sentence, e.g. "(S (NP x) (VP y))".
      has_preterms: whether or not preterminal symbols exist on the tree.

    Returns:
      None

    Raises:
      ValueError: failure to parse the string to a tree data structure.
    """
    self._sent_tree = Tree.fromstring(sent_line.strip())
    self.has_preterms = has_preterms

    # Cached DFS traversal
    self._dfs_traversal = None

    # Get the tags and token strings from the constituency tree.
    self.tags, self.raw_tokens = self.get_tags_tokens()

    # Lowercased tokens are used for pretrained word embedding lookup.
    self.lower_tokens = [token.lower() for token in self.raw_tokens]

    # Nonterminals are used to obtain the stack representation in the RNNG.
    self.nonterminals = self.get_nonterminals()

  def get_tags_tokens(self):
    """Given a constituency tree, get all tags (preterminals) and terminals.

    Args:

    Returns:
      A tuple of (tags, tokens), where "tags" is a list of tags and "tokens"
      is a list of tokens, and by definition len(tags) == len(tokens)

      For instance, if self.sent_tree = "(S (NP The hungry cat) (VP
      meows))",
      then the function would return (['DET', 'JJ', 'NNS', 'VBZ'],
        ['The', 'hungry', 'cat', 'meows'])

    Raises:
      AssertionError: The number of terminals and preterminals do not match.
    """
    tags = []
    tokens = []
    for sym_type, symbol in self.dfs_traverse():
      if "TERM" not in sym_type:
        continue
      curr_list = tags if sym_type == "PRETERM" else tokens
      curr_list.append(symbol)

    if self.has_preterms and len(tags) != len(tokens):
      raise AssertionError("Different number of tags and tokens.")

    return (tags, tokens) if self.has_preterms else (None, tokens)

  def get_nonterminals(self):
    """Given a constituency tree, get all nonterminal symbols.

    Args:

    Returns:
      A list of nonterminal symbols that occur in the sentence.
      If self.sent_tree = "(S (NP The hungry cat) (VP meows))",
      then the function would return ['S', 'NP', 'VP'].
    """
    nonterminals = []
    for sym_type, symbol in self.dfs_traverse():
      if sym_type != "NT":
        continue
      nonterminals.append(symbol)

    return nonterminals

  def _dfs_traverse_recursive(self, tree, output):
    """A recursive function that actually does the depth-first traversal.

    Args:
      tree: the nltk tree object of the sentence.
      output: the temporary output (reused in the recursive call).

    Returns:
      The list of symbols in the current subtree.
    """
    if isinstance(tree, (str)):  # Terminal symbol
      output.append(("TERM", tree))

    else:  # Nonterminal or preterminal.
      sym_type = "NT"
      if tree.height() == 2 and self.has_preterms:
        sym_type = "PRETERM"
      output.append((sym_type, tree.label()))

      if sym_type == "PRETERM":  # Preterminals can only have one child.
        assert len(tree) == 1
      for subtree in tree:
        self._dfs_traverse_recursive(subtree, output)
      if sym_type == "NT":
        output.append(("REDUCE", tree.label()))

    return output

  def dfs_traverse(self):
    """A generator function that does a depth-first traversal over the tree.

    Args:

    Yields:
      A generator that contains the next symbols in the traversal.
      Given "(S (NP (DET The) (JJ hungry) (NN cat)) (VP (VBZ meows)))",
      and has_preterms=True, this generator function would return:
      -------------------------------------------------------------------------
      ("NT", "S"),
      ("NT", "NP"),
      ("PRETERM", "DET"),
      ("TERM", "The"),
      ("PRETERM", "JJ"),
      ("TERM", "hungry"),
      ("PRETERM", "NN"),
      ("TERM", "cat"),
      ("REDUCE", "NP"),
      ("NT", "VP"),
      ("PRETERM", "VBZ"),
      ("TERM", "meows"),
      ("REDUCE", "VP"),
      ("REDUCE", "S")
      -------------------------------------------------------------------------
      If has_preterms=False, then all the "PRETERM" entries will be ignored.
    """

    # If dfs_traverse() has already been called, the result is available in
    # self._dfs_traversal.
    if self._dfs_traversal is not None:
      yield from self._dfs_traversal
    else:
      # Split the sentence to iterate over the symbols.
      output = []
      self._dfs_traverse_recursive(self._sent_tree, output)
      self._dfs_traversal = output  # Cache the value
      for sym in output:
        yield sym

  def _lc_traverse_recursive(self, tree, output):
    """A recursive function that actually does the left-corner traversal.

    Args:
      tree: the nltk tree object of the sentence.
      output: the temporary output (reused in the recursive call).

    Returns:
      The list of symbols in the current subtree.
    """
    if isinstance(tree, (str)):  # Terminal symbol
      output.append(("TERM", tree))

    else:  # Nonterminal or preterminal.
      sym_type = "NT"
      if tree.height() == 2 and self.has_preterms:
        sym_type = "PRETERM"
      if sym_type == "NT":
        self._lc_traverse_recursive(tree[0], output)
      output.append((sym_type, tree.label()))

      if sym_type == "PRETERM":  # Preterminals can only have one child.
        assert len(tree) == 1
        self._lc_traverse_recursive(tree[0], output)
      else:
        for subtree in tree[1:]:
          self._lc_traverse_recursive(subtree, output)
        output.append(("REDUCE", tree.label()))

    return output

  def lc_traverse(self):
    """A generator function that does a left-corner traversal over the tree.

    Args:

    Yields:
      A generator that contains the next symbols in the traversal.
      Given "(S (NP (DET The) (JJ hungry) (NN cat)) (VP (VBZ meows)))",
      and has_preterms=True, this generator function would return:
      -------------------------------------------------------------------------
      ("PRETERM", "DET"),
      ("TERM", "The"),
      ("NT", "NP"),
      ("PRETERM", "JJ"),
      ("TERM", "hungry"),
      ("PRETERM", "NN"),
      ("TERM", "cat"),
      ("REDUCE", "NP"),
      ("NT", "S"),
      ("PRETERM", "VBZ"),
      ("TERM", "meows"),
      ("NT", "VP"),
      ("REDUCE", "VP"),
      ("REDUCE", "S")
      -------------------------------------------------------------------------
      If has_preterms=False, then all the "PRETERM" entries will be ignored.
    """

    # Split the sentence to iterate over the symbols.
    output = []
    self._lc_traverse_recursive(self._sent_tree, output)
    for sym in output:
      yield sym

  def _bu_traverse_recursive(self, tree, output):
    """A recursive function that actually does the bottom-up traversal.

    Args:
      tree: the nltk tree object of the sentence.
      output: the temporary output (reused in the recursive call).

    Returns:
      The list of symbols in the current subtree.
    """
    if isinstance(tree, (str)):  # Terminal symbol
      output.append(("TERM", tree))

    else:  # Nonterminal or preterminal.
      sym_type = "NT"
      if tree.height() == 2 and self.has_preterms:
        sym_type = "PRETERM"
      if sym_type == "NT":
        for subtree in tree:
          self._bu_traverse_recursive(subtree, output)
        output.append(("REDUCE", (len(tree), tree.label())))
      else:  # Preterminals can only have one child.
        assert len(tree) == 1 and sym_type == "PRETERM"
        output.append((sym_type, tree.label()))
        self._bu_traverse_recursive(tree[0], output)

    return output

  def bu_traverse(self):
    """A generator function that does a bottom-up traversal over the tree.

    Args:

    Yields:
      A generator that contains the next symbols in the traversal.
      Given "(S (NP (DET The) (JJ hungry) (NN cat)) (VP (VBZ meows)))",
      and has_preterms=True, this generator function would return:
      -------------------------------------------------------------------------
      ("PRETERM", "DET"),
      ("TERM", "The"),
      ("PRETERM", "JJ"),
      ("TERM", "hungry"),
      ("PRETERM", "NN"),
      ("TERM", "cat"),
      ("REDUCE", (3, "NP")),
      ("PRETERM", "VBZ"),
      ("TERM", "meows"),
      ("REDUCE", (1, "VP")),
      ("REDUCE", (2, "S"))
      ("STOP", None)
      -------------------------------------------------------------------------
      If has_preterms=False, then all the "PRETERM" entries will be ignored.
    """

    # Split the sentence to iterate over the symbols.
    output = []
    self._bu_traverse_recursive(self._sent_tree, output)
    output.append(("STOP", None))
    for sym in output:
      yield sym

  def convert_to_choe_charniak(self, untyped_closing_terminal=False):
    """Given a tree, convert to a Choe & Charniak format.

    Ignores preterminals.

    Args:
      untyped_closing_terminal: Use untyped closing terminals.

    Returns:
      A string with the tree in the Choe & Charniak format.

    Raises:
      ValueError: Unrecognised symbol type beyond NT, REDUCE, TERM, and
      PRETERM.
    """
    output = []
    for sym_type, symbol in self.dfs_traverse():
      if sym_type == "PRETERM":
        continue

      if sym_type == "NT":
        output.append("(%s" % symbol)
      elif sym_type == "REDUCE" and not untyped_closing_terminal:
        output.append("%s)" % symbol)
      elif sym_type == "REDUCE" and untyped_closing_terminal:
        output.append("X)")
      elif sym_type == "TERM":
        output.append(symbol)
      else:
        raise ValueError("Unrecognised symbol type.")

    return " ".join(output)
