#!/bin/bash
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


set verbose
set -o errexit

# Get dependencies.
rm -rf .dependencies
mkdir .dependencies
cd .dependencies
git clone -b 20220623.1 https://github.com/abseil/abseil-cpp.git
git clone -b 3.4.0 https://gitlab.com/libeigen/eigen.git
git clone -b v2.10.2 https://github.com/pybind/pybind11.git

# SentencePiece
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
git clone -b v0.1.97 https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j
sudo make install
sudo ldconfig -v

# Go back to the package directory, install it and its dependencies.
cd ../../..
pip install --require-hashes -r requirements.txt
pip install -e . --no-deps --no-index

rm -rf .dependencies
