# Copyright 2021-2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_policy(SET CMP0048 NEW)

project(CppMasking VERSION 1.0.0)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

include_directories(../../../.dependencies/abseil-cpp)
include_directories(../../../.dependencies/pybind11/include)
include_directories(../../../.dependencies/eigen)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Abseil requires C++11
set(CMAKE_CXX_STANDARD 14)

# Process Abseil's CMake build system
add_subdirectory(../../../.dependencies/abseil-cpp abseil-cpp)

add_subdirectory(../../../.dependencies/pybind11 pybind11)

include_directories(.)
pybind11_add_module(cpp_masking cpp_masking.cc masking_rules.cc)

target_link_libraries(cpp_masking PRIVATE absl::flat_hash_set)
target_link_libraries(cpp_masking PRIVATE absl::flat_hash_map)
target_link_libraries(cpp_masking PRIVATE absl::status)
target_link_libraries(cpp_masking PRIVATE absl::statusor)
target_link_libraries(cpp_masking PRIVATE absl::strings)
