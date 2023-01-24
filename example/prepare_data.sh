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

DATA=data

# Convert Penn-style input data (one tree per line, with POS tags) to
# "Choe-Charniak" format (cf. README.md)

for SPLIT in train valid test
do
  python ../tools/convert_to_choe_charniak.py --input ${DATA}/${SPLIT}.txt --output ${DATA}/${SPLIT}.choecharniak
done

# Train a SentencePiece tokenizer

# We set the vocabulary size to 2048, because the example corpus is excessively
# small. This should usually be increased.
TOTAL_VOCAB_SIZE=2048

SPM=spm
mkdir -p ${SPM}

MODEL_PREFIX=${SPM}/spm
NON_TERMINALS=`perl -0pe 's/ /\n/g' < ${DATA}/train.choecharniak  | grep -E '\(|\)' | sort | uniq | perl -0pe 's/\n/,/g'`
spm_train \
  --input=${DATA}/train.choecharniak \
  --model_prefix=${MODEL_PREFIX} \
  --vocab_size=${TOTAL_VOCAB_SIZE} \
  --character_coverage=1.0 \
  --pad_id=0 \
  --bos_id=1 \
  --eos_id=2 \
  --unk_id=3 \
  --user_defined_symbols=${NON_TERMINALS::-1} \
  --max_sentence_length=100000 \
  --shuffle_input_sentence=true

# Tokenize the data from Choe-Charniak format to space-separated integers.

for SPLIT in train valid test prompt
do
spm_encode \
  --output_format=id \
  --model=${SPM}/spm.model \
  --input=${DATA}/${SPLIT}.choecharniak \
  --output=${DATA}/${SPLIT}.enc
done

# Remove the redundant whitespace, output as CSV.

for SPLIT in train valid test prompt
do
  python ../tools/postprocess_encoded_docs.py \
  --input ${DATA}/${SPLIT}.enc \
  --output ${DATA}/${SPLIT}.csv \
  --vocab ${SPM}/spm.vocab
done
