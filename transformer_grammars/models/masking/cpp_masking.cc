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

#include <Eigen/Core>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/internal/raw_logging.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "masking_rules.h" // NOLINT
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/gil.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace deepmind {

namespace tg_masking {

using int32_array_t = pybind11::array_t<int32_t>;
using fstyle_int32_array_t = pybind11::array_t<int32_t, py::array::f_style>;

fstyle_int32_array_t NumpyFromVector(Eigen::VectorXi input) {
  pybind11::array::ShapeContainer out_shape{input.size()};
  fstyle_int32_array_t output(out_shape);
  std::memcpy(output.mutable_data(0), input.data(),
              input.size() * sizeof(int32_t));
  return output;
}

fstyle_int32_array_t NumpyFromMatrix(Eigen::MatrixXi input) {
  pybind11::array::ShapeContainer out_shape{input.rows(), input.cols()};
  fstyle_int32_array_t output(out_shape);
  std::memcpy(output.mutable_data(0), input.data(),
              input.size() * sizeof(int32_t));
  return output;
}

// Return chunks as Python tuple of NumPy arrays. Should we return them directly
// from the masking rules instead?
template <typename T>
auto ChunksForSequence(const T& rules, int32_array_t inputs,
                       int32_array_t input_tokens_types, int32_array_t labels,
                       int32_array_t label_tokens_types) {
  // NumPy arrays passed as input must have dimension 1
  auto inputs_ = inputs.unchecked<1>();
  auto input_tokens_types_ = input_tokens_types.unchecked<1>();
  auto labels_ = labels.unchecked<1>();
  auto label_tokens_types_ = label_tokens_types.unchecked<1>();

  // Compute the chunks from the input data, using the masking rules.
  absl::StatusOr<std::vector<Chunk>> chunks_result;
  {
    // Release the GIL in this scope.
    pybind11::gil_scoped_release release_gil;
    Eigen::VectorXi inputs_vec =
        Eigen::Map<const Eigen::VectorXi>(inputs_.data(0), inputs_.size());
    Eigen::VectorXi input_tokens_types_vec = Eigen::Map<const Eigen::VectorXi>(
        input_tokens_types_.data(0), input_tokens_types_.size());
    Eigen::VectorXi labels_vec =
        Eigen::Map<const Eigen::VectorXi>(labels_.data(0), labels_.size());
    Eigen::VectorXi label_tokens_types_vec = Eigen::Map<const Eigen::VectorXi>(
        label_tokens_types_.data(0), label_tokens_types_.size());
    chunks_result = rules.GetChunksForSequence(
        inputs_vec, labels_vec, input_tokens_types_vec, label_tokens_types_vec);
  }
  if (!chunks_result.ok()) {
    throw std::runtime_error(chunks_result.status().ToString());
  }
  const std::vector<Chunk>& chunks = chunks_result.value();

  // Prepare the output shapes.
  pybind11::array::ShapeContainer out_shape_scalar;  // []

  // Prepare the container for the results.
  std::vector<std::tuple<
      fstyle_int32_array_t, fstyle_int32_array_t, fstyle_int32_array_t,
      fstyle_int32_array_t, fstyle_int32_array_t, fstyle_int32_array_t,
      fstyle_int32_array_t, fstyle_int32_array_t, fstyle_int32_array_t,
      fstyle_int32_array_t, fstyle_int32_array_t, fstyle_int32_array_t,
      fstyle_int32_array_t, fstyle_int32_array_t, fstyle_int32_array_t>>
      result;
  for (size_t i = 0; i < chunks.size(); ++i) {
    const Chunk& chunk = chunks.at(i);
    // Allocate NumPy arrays with the desired shapes.
    // We copy here, we could do something smarter.
    fstyle_int32_array_t inputs_np = NumpyFromVector(chunk.inputs);
    fstyle_int32_array_t input_tokens_types_np =
        NumpyFromVector(chunk.input_tokens_types);
    fstyle_int32_array_t labels_np = NumpyFromVector(chunk.labels);
    fstyle_int32_array_t label_tokens_types_np =
        NumpyFromVector(chunk.label_tokens_types);
    fstyle_int32_array_t attention_mask_np =
        NumpyFromMatrix(chunk.attention_mask);
    fstyle_int32_array_t attention_relative_position_np =
        NumpyFromMatrix(chunk.attention_relative_position);
    fstyle_int32_array_t attention_indicator_np =
        NumpyFromVector(chunk.attention_indicator);
    fstyle_int32_array_t memory_padding_mask_np =
        NumpyFromVector(chunk.memory_padding_mask);
    fstyle_int32_array_t memory_attention_mask_np =
        NumpyFromMatrix(chunk.memory_attention_mask);
    fstyle_int32_array_t memory_position_np =
        NumpyFromVector(chunk.memory_position);
    fstyle_int32_array_t depth_np = NumpyFromVector(chunk.depth);
    fstyle_int32_array_t update_memory_from_sequence_np =
        NumpyFromMatrix(chunk.update_memory_from_sequence);
    fstyle_int32_array_t update_memory_from_memory_np =
        NumpyFromMatrix(chunk.update_memory_from_memory);
    fstyle_int32_array_t beginning_of_sequence_np =
        fstyle_int32_array_t(out_shape_scalar);
    fstyle_int32_array_t end_of_sequence_np =
        fstyle_int32_array_t(out_shape_scalar);
    beginning_of_sequence_np.mutable_data()[0] = (i == 0) ? 1 : 0;
    end_of_sequence_np.mutable_data()[0] = (i == chunks.size() - 1) ? 1 : 0;

    result.emplace_back(
        std::move(inputs_np), std::move(input_tokens_types_np),
        std::move(labels_np), std::move(label_tokens_types_np),
        std::move(attention_mask_np), std::move(attention_relative_position_np),
        std::move(attention_indicator_np), std::move(memory_attention_mask_np),
        std::move(memory_padding_mask_np), std::move(memory_position_np),
        std::move(depth_np), std::move(beginning_of_sequence_np),
        std::move(end_of_sequence_np),
        std::move(update_memory_from_sequence_np),
        std::move(update_memory_from_memory_np));
  }

  return result;
}

// Define a Python module named `cpp_masking`.
PYBIND11_MODULE(cpp_masking, m) {
  m.doc() = "TG/TXL mask, relative positions, etc. computation classes";

  // StackComposeDoubleClosingNT class
  {
    auto cls = py::class_<StackComposeDoubleClosingNT>(
        m, "StackComposeDoubleClosingNT");

    cls.def(py::init([](size_t sequence_length, size_t memory_length,
                        std::string relative_position,
                        bool use_different_attention_fns,
                        float transparency_prob, bool gather_into_new_memory,
                        int32_t transparency_depth_threshold) {
              bool use_relative_positions = true;
              if (relative_position.empty()) {
                use_relative_positions = false;
              } else if (relative_position != "delta_depth") {
                throw std::runtime_error(
                    "Unsupported relative position indication for "
                    "StackComposeDoubleClosingNT");
              }
              if (gather_into_new_memory && (!use_relative_positions)) {
                throw std::runtime_error(
                    "gather_into_new_memory and !use_relative_positions are "
                    "incompatible.");
              }
              if (sequence_length != memory_length) {
                throw std::runtime_error(
                    "sequence_length and memory_length must be equal.");
              }
              return StackComposeDoubleClosingNT(
                  sequence_length, memory_length, use_different_attention_fns,
                  transparency_prob, use_relative_positions,
                  gather_into_new_memory, transparency_depth_threshold);
            }),
            py::kw_only(), py::arg("sequence_length"), py::arg("memory_length"),
            py::arg("relative_pos") = "delta_depth",
            py::arg("use_different_attn_fns") = false,
            py::arg("transparency_prob") = 0.0,
            py::arg("gather_into_new_memory") = false,
            py::arg("transparency_depth_threshold") = -1,
            R"delimiter(
Initialises the stack/compose masking rule.

Args:
  sequence_length: Length of the sequence in each chunk.
  memory_length: Length of the TXL-style memory. Known limitation: must be the
    same as sequence_length.
  relative_position: String indicating the type of relative position indications
    to use. Must be one of "delta_depth" (difference in the depths of attending
    vs. attended token) or "" (no relative positions computed, defaults to
    relative shift in the TXL core).
  use_different_attention_fns: Use different attention functions for STACK and
    COMPOSE positions.
  transparency_prob: Probability that a COMPOSE position is actually transparent
    (i.e. do not pop from the stack).
  gather_into_new_memory: Instead of shifting the current sequence onto the
    memory and masking out the tokens that cannot be attended, gather from the
    current sequence and the current memory to form the new memory. This
    requires explicitly passing relative positions, so it only works with
    relative_position == "delta depth".
  transparency_depth_threshold: Make all nodes of depth less than or equal to
    this value transparent. Set to -1 to disable.

)delimiter");

    cls.def_property_readonly(
        "num_attention_functions",
        &StackComposeDoubleClosingNT::GetNumAttentionFunctions);

    cls.def_property_readonly(
        "use_relative_positions",
        &StackComposeDoubleClosingNT::UseRelativePositions);

    // We use the ChunksForSequence<T> function template.
    cls.def("chunks_for_sequence",
            [](const StackComposeDoubleClosingNT& rules, int32_array_t inputs,
               int32_array_t input_tokens_types, int32_array_t labels,
               int32_array_t label_tokens_types) {
              return ChunksForSequence<StackComposeDoubleClosingNT>(
                  rules, inputs, input_tokens_types, labels,
                  label_tokens_types);
            });
  }

  // TXLCausalMasking class
  {
    auto cls = py::class_<TXLCausalMasking>(m, "TXLCausalMasking");

    cls.def(py::init([](size_t sequence_length, size_t memory_length) {
              if (sequence_length != memory_length) {
                throw std::runtime_error(
                    "sequence_length and memory_length must be equal.");
              }
              return TXLCausalMasking(sequence_length, memory_length);
            }),
            py::kw_only(), py::arg("sequence_length"), py::arg("memory_length"),
            R"delimiter(
Initialises the TXL-style causal masking rule.

Args:
  sequence_length: Length of the sequence in each chunk.
  memory_length: Length of the TXL-style memory. Known limitation: must be the
    same as sequence_length.

)delimiter");

    cls.def_property_readonly("num_attention_functions",
                              &TXLCausalMasking::GetNumAttentionFunctions);

    cls.def_property_readonly("use_relative_positions",
                              &TXLCausalMasking::UseRelativePositions);

    // We use the ChunksForSequence<T> function template.
    cls.def("chunks_for_sequence",
            [](const TXLCausalMasking& rules, int32_array_t inputs,
               int32_array_t input_tokens_types, int32_array_t labels,
               int32_array_t label_tokens_types) {
              return ChunksForSequence<TXLCausalMasking>(
                  rules, inputs, input_tokens_types, labels,
                  label_tokens_types);
            });
  }
}

}  // namespace tg_masking

}  // namespace deepmind
