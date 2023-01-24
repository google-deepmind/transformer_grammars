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

#ifndef TRANSFORMER_GRAMMARS_MODELS_MASKING_MASKING_RULES_H_
#define TRANSFORMER_GRAMMARS_MODELS_MASKING_MASKING_RULES_H_

#include <absl/status/statusor.h>

#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

namespace deepmind {

namespace tg_masking {

// Node in a tree representing the syntactic structure of a sentence. A tree is
// built from the Choe-Charniak representation, then used to compute the
// attention mask, attention relative positions, and attention indicators (i.e.
// which type of attention to apply, typically STACK or COMPOSE).
class Node {
 public:
  Node(int32_t token_type, Node* parent, Node* prev, int32_t position,
       int32_t depth, bool transparent)
      : token_type_(token_type),
        parent_(parent),
        prev_(prev),
        next_(nullptr),
        position_(position),
        depth_(depth),
        transparent_(transparent),
        children_() {}

  // Delete the default constructor to prevent creating objects with
  // uninitialized members.
  Node() = delete;

  // Returns the root of the tree corresponding to the given token types.
  static absl::StatusOr<std::unique_ptr<Node>> FromTokenTypes(
      const Eigen::VectorXi& token_types, float transparency_prob,
      int32_t transparency_depth_threshold);

  int32_t GetTokenType() const;

  // Does not transfer ownership.
  const Node* GetParent() const;

  // Does not transfer ownership.
  const Node* GetPrevious() const;

  // Does not transfer ownership.
  const Node* GetNext() const;

  // Linear position in the Choe-Charniak sequence.
  int32_t GetPosition() const;

  int32_t GetDepth() const;

  // Does not transfer ownership.
  std::vector<const Node*> GetChildren() const;

  // Does not transfer ownership.
  std::vector<const Node*> GetSiblings() const;

  bool IsTransparent() const;

 private:
  // Token type (PAD, SOS, opening non-terminal, terminal, closing non-terminal)
  int32_t token_type_;
  Node* parent_;
  Node* prev_;
  Node* next_;
  int32_t position_;
  int32_t depth_;
  bool transparent_;
  // The parent owns the children.
  std::vector<std::unique_ptr<Node>> children_;
};

// Fixed-size chunk passed to the model.
struct Chunk {
  // Transformed inputs
  Eigen::VectorXi inputs;

  // Transformed input tokens types
  Eigen::VectorXi input_tokens_types;

  // Transformed labels
  Eigen::VectorXi labels;

  // Transformed label tokens types
  Eigen::VectorXi label_tokens_types;

  // Self-attention mask
  Eigen::MatrixXi attention_mask;

  // Relative positions for attention
  Eigen::MatrixXi attention_relative_position;

  // Attention indicator
  Eigen::VectorXi attention_indicator;

  // Padding mask for the memory, 1 if the memory at index i is valid for the
  // *current* chunk. Redundant with memory_attention_mask, but useful for the
  // codepaths where memory_attention_mask is ignored. Also relieves the model
  // from dealing with this.
  Eigen::VectorXi memory_padding_mask;

  // Sequence-to-memory attention mask
  Eigen::MatrixXi memory_attention_mask;

  // Linear position in the transformed sequence corresponding to each index in
  // the memory. Used for testing.
  Eigen::VectorXi memory_position;

  // Tree depth of the input token (only computed when the tree is computed,
  // i.e. for TG)
  Eigen::VectorXi depth;

  // Indicator mapping indices from the current sequence to indices for the next
  // memory
  Eigen::MatrixXi update_memory_from_sequence;

  // Indicator mapping indices from the current memory to indices for the next
  // memory
  Eigen::MatrixXi update_memory_from_memory;
};

struct PerTokenAttentionInfo {
  std::vector<int32_t> positions_to_attend;
  std::vector<int32_t> relative_positions;
  int32_t depth;
  int32_t indicator;
  std::vector<int32_t> stack;
};

// "Masking rules" define how attention in Transformer Grammars should operate.
// It's implemented as a class that handles preparing the various model inputs,
// from the data (input tokens, input token types, label tokens, label token
// types), i.e.
// 1) if required, transform the input/labels
// 2) compute the attention mask, the relative position indication, the
//    attention function to select
// 3) return chunks, cf. above for what they contain

class AbstractMaskingRule {
 public:
  AbstractMaskingRule(size_t sequence_len, size_t memory_len,
                      bool use_relative_positions);

  // Delete the default constructor to prevent creating objects with
  // uninitialized members.
  AbstractMaskingRule() = delete;

  virtual ~AbstractMaskingRule() = default;

  // Get a list of chunks (as described above) from the data.
  virtual absl::StatusOr<std::vector<Chunk>> GetChunksForSequence(
      Eigen::VectorXi inputs, Eigen::VectorXi labels,
      Eigen::VectorXi input_tokens_types,
      Eigen::VectorXi label_tokens_types) const = 0;

  // Sequence length for the model.
  size_t GetSequenceLength() const { return sequence_len_; }

  // Memory length for the model.
  size_t GetMemoryLength() const { return memory_len_; }

  // Number of attention functions to use (e.g. a TXL has one, but stack-compose
  // proposals have 2)
  virtual size_t GetNumAttentionFunctions() const { return 1; }

  // Whether the model should use the relative positions returned by the masking
  // code. This is better set to false for TXL, so that the fast path with
  // shifted logits is used. This can also be set to false to use TXL-style
  // relative positions with a TG mask.
  bool UseRelativePositions() const { return use_relative_positions_; }

 protected:
  size_t sequence_len_;
  size_t memory_len_;
  bool use_relative_positions_;
};

// We support Transformer-XL as a trivial case of masking rules.
class TXLCausalMasking : public AbstractMaskingRule {
 public:
  TXLCausalMasking(size_t sequence_len, size_t memory_len);

  // Delete the default constructor to prevent creating objects with
  // uninitialized members.
  TXLCausalMasking() = delete;

  absl::StatusOr<std::vector<Chunk>> GetChunksForSequence(
      Eigen::VectorXi inputs, Eigen::VectorXi labels,
      Eigen::VectorXi input_tokens_types,
      Eigen::VectorXi label_tokens_types) const final;

  size_t GetNumAttentionFunctions() const override { return 1; }
};

// Stack-compose proposal with duplicated closing non-terminals.
class StackComposeDoubleClosingNT : public AbstractMaskingRule {
 public:
  StackComposeDoubleClosingNT(size_t sequence_len, size_t memory_len,
                              bool use_different_attention_fns,
                              float transparency_prob,
                              bool use_relative_positions,
                              bool gather_into_new_memory,
                              int32_t transparency_depth_threshold);

  // Delete the default constructor to prevent creating objects with
  // uninitialized members.
  StackComposeDoubleClosingNT() = delete;

  absl::StatusOr<std::vector<Chunk>> GetChunksForSequence(
      Eigen::VectorXi inputs, Eigen::VectorXi labels,
      Eigen::VectorXi input_tokens_types,
      Eigen::VectorXi label_tokens_types) const final;

  size_t GetNumAttentionFunctions() const override {
    return use_different_attention_fns_ ? 2 : 1;
  }

 private:
  // Transform inputs, labels, input tokens types, label tokens types.
  // Here it means duplicating the closing non-terminals, and predicting a label
  // for the second one only.
  std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>
  Transform(Eigen::VectorXi inputs, Eigen::VectorXi labels,
            Eigen::VectorXi input_tokens_types,
            Eigen::VectorXi label_tokens_types) const;

  // Compute the position to attend to, and with which relative positions, and
  // with which attention function (stack or compose), for each position in
  // the sequence.
  absl::StatusOr<std::vector<PerTokenAttentionInfo>> ComputeAttentionQuantities(
      Eigen::VectorXi transformed_input_tokens_types) const;

  // From the transformed inputs, and the attention quantities computed above,
  // return chunks for the model. A chunk is a struct of Eigen vectors and
  // matrices.
  std::vector<Chunk> MakeChunks(
      Eigen::VectorXi transformed_inputs, Eigen::VectorXi transformed_labels,
      Eigen::VectorXi transformed_input_tokens_types,
      Eigen::VectorXi transformed_label_tokens_types,
      std::vector<PerTokenAttentionInfo> attention_info_vec) const;

  bool use_different_attention_fns_;

  float transparency_prob_;

  bool gather_into_new_memory_;

  int32_t transparency_depth_threshold_;
};

}  // namespace tg_masking

}  // namespace deepmind

#endif  // TRANSFORMER_GRAMMARS_MODELS_MASKING_MASKING_RULES_H_
