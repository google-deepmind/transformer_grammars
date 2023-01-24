// Copyright 2021-2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// C++ implementation of masking rules for Transformer Grammars

// Computing the attention mask, relative positions, the attention function
// indicator, etc. is very easy when the tree corresponding to the structure of
// the sequence has been built using the token types. This is an iterative
// process involving manipulating data structures one element at a time, so it
// is slow in Python, but reasonably fast in C++.

#include "masking_rules.h" // NOLINT

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"

namespace deepmind {

namespace tg_masking {

// Token types
constexpr int32_t kPad = 0;
constexpr int32_t kSos = 1;
constexpr int32_t kOpeningNonTerminal = 2;
constexpr int32_t kClosingNonTerminal = 4;
constexpr int32_t kClosingNonTerminal2 = 14;

absl::StatusOr<std::unique_ptr<Node>> Node::FromTokenTypes(
    const Eigen::VectorXi& token_types, float transparency_prob,
    int32_t transparency_depth_threshold) {
  auto root =
      std::make_unique<Node>(token_types[0] == 0 ? kSos : token_types[0],
                             nullptr, nullptr, 0, 0, false);
  Node* parent = root.get();
  Node* prev = root.get();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (size_t i = 1; i < token_types.size(); ++i) {
    int32_t ttype = token_types[i];
    if (ttype == kPad) {
      continue;
    } else if (ttype == kClosingNonTerminal) {
      // We need this check, as it may be triggered if the input is not
      // correctly parenthesized. Or should we throw instead?
      if (parent == nullptr) {
        return absl::InvalidArgumentError(
            "parent is a null pointer. Is the input correctly parenthesized?");
      }
      parent = parent->parent_;
    }
    if (parent == nullptr) {
      return absl::InvalidArgumentError(
          "parent is a null pointer. Is the input correctly parenthesized?");
    }
    int32_t node_depth = parent->depth_ + 1;
    bool transparent = (ttype == kOpeningNonTerminal) &&
                       ((dis(gen) <= transparency_prob) ||
                        node_depth <= transparency_depth_threshold);
    parent->children_.emplace_back(std::make_unique<Node>(
        ttype, parent, prev, i, node_depth, transparent));
    Node* node = parent->children_.back().get();
    if (prev == nullptr) {
      return absl::InvalidArgumentError(
          "parent is a null pointer. Is the input correctly parenthesized?");
    }
    prev->next_ = node;
    prev = node;
    if (ttype == kOpeningNonTerminal) {
      parent = node;
    }
  }
  return root;
}

int32_t Node::GetTokenType() const { return token_type_; }

const Node* Node::GetParent() const { return parent_; }

const Node* Node::GetPrevious() const { return prev_; }

const Node* Node::GetNext() const { return next_; }

int32_t Node::GetPosition() const { return position_; }

int32_t Node::GetDepth() const { return depth_; }

bool Node::IsTransparent() const { return transparent_; }

// Does not transfer ownership.
std::vector<const Node*> Node::GetChildren() const {
  std::vector<const Node*> children;
  for (const std::unique_ptr<Node>& node : children_) {
    children.emplace_back(node.get());
    if (node->IsTransparent()) {
      for (const Node* childnode : node->GetChildren()) {
        children.emplace_back(childnode);
      }
    }
  }
  return children;
}

std::vector<const Node*> Node::GetSiblings() const {
  if (parent_ == nullptr) {
    return {};
  } else {
    return parent_->GetChildren();
  }
}

AbstractMaskingRule::AbstractMaskingRule(size_t sequence_len, size_t memory_len,
                                         bool use_relative_positions)
    : sequence_len_(sequence_len),
      memory_len_(memory_len),
      use_relative_positions_(use_relative_positions) {}

// -----------------------------------------------------------------------------
// TXL causal masking

TXLCausalMasking::TXLCausalMasking(size_t sequence_len, size_t memory_len)
    : AbstractMaskingRule(sequence_len, memory_len, false) {}

absl::StatusOr<std::vector<Chunk>> TXLCausalMasking::GetChunksForSequence(
    Eigen::VectorXi inputs, Eigen::VectorXi labels,
    Eigen::VectorXi input_tokens_types,
    Eigen::VectorXi label_tokens_types) const {
  const size_t len = inputs.size();
  const int32_t num_chunks =
      (len / sequence_len_) + (len % sequence_len_ ? 1 : 0);

  // Allocate output arrays.
  Eigen::VectorXi inputs_ = Eigen::VectorXi::Zero(num_chunks * sequence_len_);
  Eigen::VectorXi input_tokens_types_ =
      Eigen::VectorXi::Zero(num_chunks * sequence_len_);
  Eigen::VectorXi labels_ = Eigen::VectorXi::Zero(num_chunks * sequence_len_);
  Eigen::VectorXi label_tokens_types_ =
      Eigen::VectorXi::Zero(num_chunks * sequence_len_);

  // There is no transformation for TXL, so simply copy.
  inputs_(Eigen::seqN(0, len)) = inputs;
  input_tokens_types_(Eigen::seqN(0, len)) = input_tokens_types;
  labels_(Eigen::seqN(0, len)) = labels;
  label_tokens_types_(Eigen::seqN(0, len)) = label_tokens_types;

  // Initialize the memory padding mask to 0
  Eigen::VectorXi memory_padding_mask_ = Eigen::VectorXi::Zero(memory_len_);

  std::vector<Chunk> result;
  for (size_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
    Chunk chunk;

    int32_t sequence_start = chunk_index * sequence_len_;      // Inclusive
    int32_t sequence_end = (chunk_index + 1) * sequence_len_;  // Exclusive
    int32_t memory_start = sequence_start - memory_len_;       // Inclusive
    int32_t memory_end = sequence_start;                       // Exclusive

    chunk.inputs = inputs_(Eigen::seqN(sequence_start, sequence_len_));
    chunk.input_tokens_types =
        input_tokens_types_(Eigen::seqN(sequence_start, sequence_len_));
    chunk.labels = labels_(Eigen::seqN(sequence_start, sequence_len_));
    chunk.label_tokens_types =
        label_tokens_types_(Eigen::seqN(sequence_start, sequence_len_));

    // Do not emit empty chunks (may happen when the transformed length is
    // overestimated, but does not for TXL)
    if ((chunk.inputs.array() == 0).all()) {
      break;
    }

    // The attention indicator is always 0 for TXL.
    chunk.attention_indicator = Eigen::VectorXi::Zero(sequence_len_);
    chunk.attention_mask = Eigen::MatrixXi::Zero(sequence_len_, sequence_len_);
    chunk.attention_relative_position =
        Eigen::MatrixXi::Zero(sequence_len_, sequence_len_ + memory_len_);
    chunk.memory_attention_mask =
        Eigen::MatrixXi::Zero(sequence_len_, memory_len_);
    chunk.memory_padding_mask = Eigen::VectorXi::Zero(memory_len_);
    chunk.memory_position = Eigen::VectorXi::Zero(memory_len_);
    chunk.depth = Eigen::VectorXi::Zero(sequence_len_);  // Always 0 for TXL.
    chunk.update_memory_from_sequence =
        Eigen::MatrixXi::Zero(sequence_len_, memory_len_);
    chunk.update_memory_from_memory =
        Eigen::MatrixXi::Zero(memory_len_, memory_len_);

    // Copy the padding mask into the current chunk -- it indicates which
    // positions of the *current memory* are valid.
    for (size_t i = 0; i < memory_len_; ++i) {
      chunk.memory_padding_mask[i] = memory_padding_mask_[i];
    }

    // Update the padding mask, assuming memory_len_ == sequence_len_.
    // TODO: Relax this.
    for (size_t i = 0; i < memory_len_; ++i) {
      memory_padding_mask_[i] = chunk.inputs[i] > 0 ? 1 : 0;
      if (chunk_index > 0) {
        chunk.memory_position[i] = (chunk_index - 1) * sequence_len_ + i;
      } else {
        chunk.memory_position[i] = -1;
      }
    }

    for (size_t i = 0; i < sequence_len_; ++i) {
      size_t sequence_index = chunk_index * sequence_len_ + i;
      chunk.update_memory_from_sequence(i, i) = 1;
      for (size_t j = 0; j <= sequence_index; ++j) {
        int32_t position_to_attend = j;  // TXL causal attention
        int32_t relative_position_value =
            sequence_index - j;  // TXL relative positions
        if ((sequence_start <= position_to_attend) &&
            (position_to_attend < sequence_end)) {
          chunk.attention_mask(i, position_to_attend - sequence_start) = 1;
        }
        if ((memory_start <= position_to_attend) &&
            (position_to_attend < memory_end)) {
          chunk.memory_attention_mask(i, position_to_attend - memory_start) = 1;
        }
        if ((memory_start <= position_to_attend) &&
            (position_to_attend < sequence_end)) {
          chunk.attention_relative_position(
              i, position_to_attend - memory_start) = relative_position_value;
        }
      }
    }
    result.push_back(std::move(chunk));
  }

  return result;
}

// End of TXL causal masking
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Stack/compose with duplicated closing non-terminal

StackComposeDoubleClosingNT::StackComposeDoubleClosingNT(
    size_t sequence_len, size_t memory_len, bool use_different_attention_fns,
    float transparency_prob, bool use_relative_positions,
    bool gather_into_new_memory, int32_t transparency_depth_threshold)
    : AbstractMaskingRule(sequence_len, memory_len, use_relative_positions),
      use_different_attention_fns_(use_different_attention_fns),
      transparency_prob_(transparency_prob),
      gather_into_new_memory_(gather_into_new_memory),
      transparency_depth_threshold_(transparency_depth_threshold) {}

absl::StatusOr<std::vector<Chunk>>
StackComposeDoubleClosingNT::GetChunksForSequence(
    Eigen::VectorXi inputs, Eigen::VectorXi labels,
    Eigen::VectorXi input_tokens_types,
    Eigen::VectorXi label_tokens_types) const {
  // Transform the inputs/labels following the rules.
  Eigen::VectorXi transformed_inputs;
  Eigen::VectorXi transformed_labels;
  Eigen::VectorXi transformed_input_tokens_types;
  Eigen::VectorXi transformed_label_tokens_types;
  std::tie(transformed_inputs, transformed_labels,
           transformed_input_tokens_types, transformed_label_tokens_types) =
      Transform(inputs, labels, input_tokens_types, label_tokens_types);

  // Get the 3-uple of positions to attend, relative positions, and attention
  // indicator for the whole sequence.
  std::vector<PerTokenAttentionInfo> attention_info_vec =
    ComputeAttentionQuantities(transformed_input_tokens_types).value();

  // Create the chunks of data to be passed to the model.
  return MakeChunks(transformed_inputs, transformed_labels,
                    transformed_input_tokens_types,
                    transformed_label_tokens_types, attention_info_vec);
}

std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
StackComposeDoubleClosingNT::Transform(
    Eigen::VectorXi inputs, Eigen::VectorXi labels,
    Eigen::VectorXi input_tokens_types,
    Eigen::VectorXi label_tokens_types) const {
  const size_t len = inputs.size();
  // Allocate the output arrays.
  size_t transformed_len = len * 2;
  size_t transformed_rem = transformed_len % GetSequenceLength();
  if (transformed_rem > 0) {
    transformed_len += (GetSequenceLength() - transformed_rem);
  }
  Eigen::VectorXi transformed_inputs = Eigen::VectorXi::Zero(transformed_len);
  Eigen::VectorXi transformed_input_tokens_types =
      Eigen::VectorXi::Zero(transformed_len);
  Eigen::VectorXi transformed_labels = Eigen::VectorXi::Zero(transformed_len);
  Eigen::VectorXi transformed_label_tokens_types =
      Eigen::VectorXi::Zero(transformed_len);

  size_t j = 0;
  for (size_t i = 0; i < len; ++i) {
    if (input_tokens_types[i] == kClosingNonTerminal) {
      // Encountered a closing non-terminal in the input, so duplicate it.
      transformed_inputs[j] = inputs[i];
      transformed_inputs[j + 1] = inputs[i];
      transformed_input_tokens_types[j] = kClosingNonTerminal;
      transformed_input_tokens_types[j + 1] = kClosingNonTerminal2;
      transformed_labels[j] = 0;
      transformed_labels[j + 1] = labels[i];
      transformed_label_tokens_types[j] = kPad;
      transformed_label_tokens_types[j + 1] = label_tokens_types[i];
      j += 2;
    } else {
      transformed_inputs[j] = inputs[i];
      transformed_input_tokens_types[j] = input_tokens_types[i];
      transformed_labels[j] = labels[i];
      transformed_label_tokens_types[j] = label_tokens_types[i];
      ++j;
    }
  }

  return std::make_tuple(transformed_inputs, transformed_labels,
                         transformed_input_tokens_types,
                         transformed_label_tokens_types);
}

absl::StatusOr<std::vector<PerTokenAttentionInfo>>
StackComposeDoubleClosingNT::ComputeAttentionQuantities(
    Eigen::VectorXi transformed_input_tokens_types) const {
  const size_t transformed_len = transformed_input_tokens_types.size();

  std::vector<PerTokenAttentionInfo> result;

  std::unique_ptr<Node> root = Node::FromTokenTypes(
    transformed_input_tokens_types, transparency_prob_,
    transparency_depth_threshold_).value();
  const Node* cur = root.get();
  std::vector<const Node*> stack;
  // To quickly check whether an element is on the stack.
  absl::flat_hash_set<const Node*> stack_set;
  while (cur) {
    std::vector<const Node*> to_attend;
    int32_t attention_indicator = 0;
    PerTokenAttentionInfo attention_info;
    attention_info.depth = cur->GetDepth();
    if (cur->GetTokenType() == kPad) {
      // Nothing to do, padding tokens do not attend
      continue;
    } else if (cur->GetTokenType() == kClosingNonTerminal) {
      // CLOSING_NT is used for the 1st closing NT
      // Compose attention: compose opening NT and tokens at the same level, if
      // they are still on the stack.
      std::vector<const Node*> parent_and_siblings =
          cur->GetPrevious()->GetSiblings();
      parent_and_siblings.insert(parent_and_siblings.begin(),
                                 cur->GetPrevious()->GetParent());
      for (const Node* node : parent_and_siblings) {
        // Only attend to parents and siblings on the stack.
        if (stack_set.find(node) != stack_set.end()) {
          to_attend.push_back(node);
        }
      }
      // Pop from the stack the elements that will be attended.
      if (!cur->GetPrevious()->GetParent()->IsTransparent()) {
        stack.erase(stack.end() - to_attend.size(), stack.end());
        for (const Node* x : to_attend) {
          stack_set.erase(x);
        }
      }
      to_attend.push_back(cur);
      stack.push_back(cur);
      stack_set.insert(cur);
      // Use a different MHA for compose when use_different_attention_fns is
      // true.
      if (use_different_attention_fns_) {
        attention_indicator = 1;
      }
    } else {
      // Stack attention
      to_attend = stack;
      to_attend.push_back(cur);
      if (cur->GetTokenType() != kClosingNonTerminal2) {
        stack.push_back(cur);
        stack_set.insert(cur);
      }
    }
    attention_info.relative_positions.reserve(to_attend.size());
    attention_info.positions_to_attend.reserve(to_attend.size());
    for (const Node* nodeptr : to_attend) {
      attention_info.positions_to_attend.push_back(nodeptr->GetPosition());
      attention_info.relative_positions.push_back(cur->GetDepth() -
                                                  nodeptr->GetDepth());
    }
    for (const Node* nodeptr : stack) {
      attention_info.stack.push_back(nodeptr->GetPosition());
    }
    attention_info.indicator = attention_indicator;
    result.emplace_back(attention_info);
    cur = cur->GetNext();
  }

  // Pad the results if the length of the transformed input (computed ahead of
  // time) is longer than the actual necessary length.
  while (result.size() < transformed_len) {
    result.emplace_back();
  }
  return result;
}

std::vector<Chunk> StackComposeDoubleClosingNT::MakeChunks(
    Eigen::VectorXi transformed_inputs, Eigen::VectorXi transformed_labels,
    Eigen::VectorXi transformed_input_tokens_types,
    Eigen::VectorXi transformed_label_tokens_types,
    std::vector<PerTokenAttentionInfo> attention_info_vec) const {
  const size_t transformed_len = transformed_inputs.size();
  int32_t num_chunks = transformed_len / sequence_len_;

  // Initialize the memory padding mask to 0
  Eigen::VectorXi memory_padding_mask_ = Eigen::VectorXi::Zero(memory_len_);

  // Track which index in the memory corresponds to which linear position in the
  // sequence.
  absl::flat_hash_map<int32_t, size_t> memory_index_from_position;
  Eigen::VectorXi memory_position_from_index_ =
      Eigen::VectorXi::Constant(memory_len_, -1);
  size_t last_nonzero_index = 0;

  std::vector<Chunk> result;
  for (size_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
    // Initialize the new chunk.
    Chunk chunk;

    // Find positions for the beginning/end of the sequence/memory.
    int32_t sequence_start = chunk_index * sequence_len_;      // Inclusive
    int32_t sequence_end = (chunk_index + 1) * sequence_len_;  // Exclusive
    int32_t memory_start = sequence_start - memory_len_;       // Inclusive
    int32_t memory_end = sequence_start;                       // Exclusive

    // Slice to point to the right sections of size sequence_len_ and
    // memory_len_.
    chunk.inputs =
        transformed_inputs(Eigen::seqN(sequence_start, sequence_len_));
    chunk.input_tokens_types = transformed_input_tokens_types(
        Eigen::seqN(sequence_start, sequence_len_));
    chunk.labels =
        transformed_labels(Eigen::seqN(sequence_start, sequence_len_));
    chunk.label_tokens_types = transformed_label_tokens_types(
        Eigen::seqN(sequence_start, sequence_len_));

    // Do not emit empty chunks.
    if ((chunk.inputs.array() == 0).all()) {
      break;
    }

    // Allocate output arrays for the chunk.
    chunk.attention_indicator = Eigen::VectorXi::Zero(sequence_len_);
    chunk.attention_mask = Eigen::MatrixXi::Zero(sequence_len_, sequence_len_);
    chunk.attention_relative_position =
        Eigen::MatrixXi::Zero(sequence_len_, sequence_len_ + memory_len_);
    chunk.memory_attention_mask =
        Eigen::MatrixXi::Zero(sequence_len_, memory_len_);
    chunk.memory_padding_mask = Eigen::VectorXi::Zero(memory_len_);
    chunk.memory_position = Eigen::VectorXi::Zero(memory_len_);
    chunk.depth = Eigen::VectorXi::Zero(sequence_len_);
    chunk.update_memory_from_sequence =
        Eigen::MatrixXi::Zero(sequence_len_, memory_len_);
    chunk.update_memory_from_memory =
        Eigen::MatrixXi::Zero(memory_len_, memory_len_);

    // Copy the padding mask into the current chunk -- it indicates which
    // positions of the *current memory* are valid. Reset the memory padding
    // state.
    for (size_t i = 0; i < memory_len_; ++i) {
      chunk.memory_padding_mask[i] = memory_padding_mask_[i];
      memory_padding_mask_[i] = 0;
      chunk.memory_position[i] = memory_position_from_index_[i];
      memory_position_from_index_[i] = -1;
    }

    // Convert the attention quantities represented as lists of positions to
    // attend, as well as their relative positions, to what the model expects.
    for (size_t i = 0; i < sequence_len_; ++i) {
      size_t sequence_index = chunk_index * sequence_len_ + i;
      const PerTokenAttentionInfo& attention_info =
          attention_info_vec.at(sequence_index);
      const std::vector<int32_t>& to_attend =
          attention_info.positions_to_attend;
      const std::vector<int32_t>& relative_position =
          attention_info.relative_positions;
      const int32_t attention_indicator = attention_info.indicator;
      const int32_t depth = attention_info.depth;
      if (chunk.inputs[i] > 0) {
        last_nonzero_index = sequence_index;
      }
      chunk.attention_indicator[i] = attention_indicator;
      chunk.depth[i] = depth;
      if (!gather_into_new_memory_ && (chunk.inputs[i] > 0)) {
        memory_padding_mask_[i] = 1;
        chunk.update_memory_from_sequence(i, i) = 1;
        memory_position_from_index_[i] = sequence_index;
      }
      for (size_t j = 0; j < to_attend.size(); ++j) {
        int32_t position_to_attend = to_attend.at(j);
        int32_t relative_position_value = relative_position.at(j);
        if ((sequence_start <= position_to_attend) &&
            (position_to_attend < sequence_end)) {
          chunk.attention_mask(i, position_to_attend - sequence_start) = 1;
          chunk.attention_relative_position(
              i, position_to_attend - memory_start) = relative_position_value;
        }
        if ((!gather_into_new_memory_) &&
            (memory_start <= position_to_attend) &&
            (position_to_attend < memory_end)) {
          chunk.memory_attention_mask(i, position_to_attend - memory_start) = 1;
          chunk.attention_relative_position(
              i, position_to_attend - memory_start) = relative_position_value;
        } else if (gather_into_new_memory_) {
          auto it = memory_index_from_position.find(position_to_attend);
          if (it != memory_index_from_position.end()) {
            size_t memory_index_to_attend = it->second;
            chunk.memory_attention_mask(i, memory_index_to_attend) = 1;
            chunk.attention_relative_position(i, memory_index_to_attend) =
                relative_position_value;
          }
        }
      }
    }

    // Memory management
    if (gather_into_new_memory_) {
      // "smart" memory management, i.e. don't shift and mask, but gather the
      // positions that we may attend to in the future

      // The positions we may attend to in the future and the ones on the stack:
      const std::vector<int32_t>& attendable =
          attention_info_vec.at(last_nonzero_index).stack;

      // Reset the memory book-keeping structures mask.
      for (size_t i = 0; i < memory_len_; ++i) {
        memory_padding_mask_[i] = 0;
      }
      absl::flat_hash_map<int32_t, size_t> new_memory_index_from_position;

      // Start filling the memory from the end.
      int32_t new_memory_index = memory_len_ - 1;

      for (auto it = attendable.rbegin(); it != attendable.rend(); ++it) {
        if (new_memory_index < 0) {
          break;
        }
        int32_t attendable_position = *it;
        // Is this attendable position part of the current sequence?
        if ((attendable_position >= sequence_start) &&
            (attendable_position < sequence_end)) {
          int32_t attendable_sequence_index =
              attendable_position - sequence_start;
          chunk.update_memory_from_sequence(attendable_sequence_index,
                                            new_memory_index) = 1;
          memory_padding_mask_[new_memory_index] = 1;
          memory_position_from_index_[new_memory_index] = attendable_position;
          new_memory_index_from_position[attendable_position] =
              new_memory_index;
          --new_memory_index;
        } else {
          // Is this attendable position part of the memory?
          auto it2 = memory_index_from_position.find(attendable_position);
          if (it2 != memory_index_from_position.end()) {
            int32_t attendable_memory_index = it2->second;
            chunk.update_memory_from_memory(attendable_memory_index,
                                            new_memory_index) = 1;
            memory_padding_mask_[new_memory_index] = 1;
            memory_position_from_index_[new_memory_index] = attendable_position;
            new_memory_index_from_position[attendable_position] =
                new_memory_index;
            --new_memory_index;
          }
        }
      }

      // Put the new book-keeping structures in place for the next step.
      memory_index_from_position = new_memory_index_from_position;
    }

    result.push_back(std::move(chunk));
  }

  return result;
}

// End of stack/compose with duplicated closing non-terminal
// -----------------------------------------------------------------------------

}  // namespace tg_masking

}  // namespace deepmind
