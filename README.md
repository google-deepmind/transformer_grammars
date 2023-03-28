# Transformer Grammars

Transformer Grammars are Transformer-like models of the joint structure and
sequence of words of a sentence or document. Specifically, they model the
sequence of actions describing a linearized tree. Their distinguishing feature
is that the attention mask used in the Transformer core is a function of the
structure itself, so that representations of constituents are composed
recursively. The approach is fully described in our paper Transformer Grammars:
Augmenting Transformer Language Models with Syntactic Inductive Biases at Scale,
Sartran et al., TACL (2022), available from MIT Press at
[this address](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00526/114315).

## Code organization

The code is organized as follows:

```text
transformer_grammars/
├─ transformer_grammars/    TG module used by the entrypoints
│  ├─ data/                 Dataset, tokenizer, input transformation, etc.
│  ├─ models/               Core model code
│  │  ├─ masking/           C++ masking code, for the attention mask, relative
│  │                        positions, etc.
│  ├─ training/             Training loop
├─ configs/                 Configs used for the paper
├─ example/                 Example data + scripts to train and use a model
├─ tools/                   Misc tools to prepare the data, cf. below
├─ train.py                 Entrypoint for training
├─ score.py                 Entrypoint for scoring
├─ sample.py                Entrypoint for sampling
```

## Installation

This code was tested on Google Cloud Compute Engine, using a N1 instance, NVIDIA
V100 GPU, and the disk image Debian 10 based Deep Learning VM with CUDA 11.3
preinstalled, M102. In particular, the Python version it contains is 3.7, for
which we install the corresponding `jaxlib` package.

1.  Download the code from the `transformer_grammars` repository.

```bash
git clone https://github.com/deepmind/transformer_grammars.git
cd transformer_grammars
```

2.  Create a virtual environment.

```bash
python -m venv .tgenv
source .tgenv/bin/activate
```

3.  Install the package (in development mode) and its dependencies.

```bash
./install.sh
```

This also builds the C++ extension that is required to compute the attention
mask, relative positions, memory update, etc.

4.  Run the test suite.

```bash
nosetests transformer_grammars
```

## Quick start: example

We provide in the `example/` directory parsed sentences from Dickens's A Tale of
Two Cities, prepared using spaCy for sentence segmentation (`en_core_web_md`)
and Benepar for parsing (`benepar_en3_large`), and split into
`{train,valid,test}.txt`. The following can be done from that directory:

-   Run the data preparation described above at once with `./prepare_data.sh`.
-   Train a model for a few steps with `./run_training.sh`.
-   Use it to score the test set with `./run_scoring.sh`.
-   Use it to generate samples with `./run_sampling.sh`.

NOTE: Such a small model trained for so few steps will give bad results -- this
is only meant as an end-to-end example of training and using a TG model. To
really train and use the model, please follow the instructions below.

## Training and using a TG model

### Data preparation

#### Expected input

The expected input format is one tree per line, with POS tags, e.g.

```text
(S (S (NP (NNP John) (NNP Blair) (CC &) (NNP Co.)) (VP (VBZ is) (ADVP (RB close) (PP (TO to) (NP (DT an) (NN agreement) (S (VP (TO to) (VP (VB sell) (NP (PRP$ its) (NX (NX (NN TV) (NN station) (NN advertising) (NN representation) (NN operation)) (CC and) (NX (NN program) (NN production) (NN unit)))) (PP (TO to) (NP (NP (DT an) (NN investor) (NN group)) (VP (VBN led) (PP (IN by) (NP (NP (NNP James) (NNP H.) (NNP Rosenfield)) (, ,) (NP (DT a) (JJ former) (NNP CBS) (NNP Inc.) (NN executive))))))))))))))) (, ,) (NP (NN industry) (NNS sources)) (VP (VBD said)) (. .))
```

We assume that 3 such files are available: `train.txt`, `valid.txt`, `test.txt`
in `$DATA`.

Whilst not specific to our work, we describe below how we prepared the example
data.

#### Convert to Choe-Charniak

Convert all files to "Choe-Charniak" format (i.e. one sequence of actions,
including opening and closing non-terminals, but excluding POS tags, per line)
using `tools/convert_to_choe_charniak.py`

```bash
for SPLIT in train valid test
do
    python tools/convert_to_choe_charniak.py --input $DATA/${SPLIT}.txt --output $DATA/${SPLIT}.cc
done
```

Now, there are two ways of transforming this data into sequences of integers for modelling:
- one uses SentencePiece (recommended), with which we can train a tokenization model
- the other involves learning a word-based vocabulary

Exactly one or the other needs to be done.

#### With SentencePiece

##### Training a tokenizer (SentencePiece)

Create a directory, set it to the `$TOKENIZER` environment variable.

Train a tokenizer on the training data with the following, adjusting the user
defined symbols to reflect the non-terminals in the data (can be obtained
automatically with `perl -0pe 's/ /\n/g' < $DATA/train.cc | grep -E '\(|\)' |
LC_ALL=C sort | uniq | perl -0pe 's/\n/,/g'`, but do check the list).

```bash
MODEL_PREFIX=$TOKENIZER/spm
NON_TERMINALS=`perl -0pe 's/ /\n/g' < $DATA/train.cc  | grep -E '\(|\)' | LC_ALL=C sort | uniq | perl -0pe 's/\n/,/g'`
spm_train \
--input=$DATA/train.cc \
--model_prefix=$MODEL_PREFIX \
--vocab_size=32768 \
--character_coverage=1.0 \
--pad_id=0 \
--bos_id=1 \
--eos_id=2 \
--unk_id=3 \
--user_defined_symbols=${NON_TERMINALS::-1} \
--max_sentence_length=100000 \
--shuffle_input_sentence=true
```

This produces two files, `$TOKENIZER/spm.model` and `$TOKENIZER/spm.vocab`. The
first one is the actual model, the second is the vocabulary, one token per line.

##### Tokenizing the data (SentencePiece)

Encode the train/validation/test data with:

```bash
for SPLIT in train valid test
do
    spm_encode \
    --output_format=id \
    --model=$TOKENIZER/spm.model \
    --input=$DATA/${SPLIT}.cc \
    --output=$DATA/${SPLIT}.enc
done
```

We want our user-defined symbols to implicitly represent whitespace before and
after them as necessary, but SentencePiece treats them literally. We therefore
end up with extraneous space tokens:

```text
(S (NP The blue bird NP) (VP sings VP) S)
```

is tokenized into

```text
▁ (S ▁ (NP ▁The ▁blue ▁ bird ▁ NP) ▁ (VP ▁sings ▁ VP) ▁ S)
```

even though we want

```text
(S (NP ▁The ▁blue ▁ bird NP) (VP ▁sings VP) S)
```

We fix this with a post-processing step:

```bash
for SPLIT in train valid test
do
    python tools/postprocess_encoded_docs.py \
    --input $DATA/${SPLIT}.enc \
    --output $DATA/${SPLIT}.csv \
    --vocab $TOKENIZER/spm.vocab
done
```

#### Word-based vocabulary

This is the alternative to SentencePiece. We used this for our word-level experiments on PTB only. Most applications will use SentencePiece.

##### Training a tokenizer (word-based)

Create a directory, set it to the `$TOKENIZER` environment variable.

Train a word-based, closed-vocabulary tokenizer, using:

```bash
python tools/build_dictionary.py \
--input $DATA/train.cc \
--output $TOKENIZER/word_based
```

This produces two files, `$TOKENIZER/word_based.txt` and
`$TOKENIZER/word_based.json`. The first one is the vocabulary, one token per
line; the second one contains metadata.

##### Tokenizing the data (word-based)

Encode the train/validation/test data with:

```bash
for SPLIT in train valid test
do
    python tools/encode_offline.py \
    --input $DATA/${SPLIT}.cc \
    --output $DATA/${SPLIT}.csv \
    --dictionary $TOKENIZER/word_based
done
```

Warning! This will not allow UNK tokens.

### Training a model

The `train.csv`, `valid.csv`, `test.csv` now contain the tokenized (possibly
post-processed, if using SentencePiece) sequences from the corresponding `.txt`
file. The training config should point to them (for training and for
evaluation), as well as to the tokenizer learnt.

Launch a training run with:

```bash
python train.py --config config.py
```

Checkpoints are regularly saved into `checkpoint.pkl`.

### Using a model for scoring

Given a model checkpoint in `checkpoint.pkl`, a dataset of tokenized
(post-processed if necessary) sequences (as prepared above) in
`$DATA/valid.csv`, and the corresponding tokenizer in `$TOKENIZER/spm.model`,
sequences can be scored with:

```bash
python score.py \
--checkpoint checkpoint.pkl \
--tokenizer $TOKENIZER/spm.model \
--input $DATA/valid.csv
```

It outputs, for each position in the sequence, the input token, the label, and
its associated log probability.

### Using a model for sampling

We did not investigate the samples obtained by TG in our work, but we provide an
example of a sampling script for illustration purposes. It is easy to sample
from a TG model in principle, though because our implementation separates the
computation of the attention mask, relative positions, memory update matrices
(done in C++ in the data loading pipeline) and the model core (in JAX), it
requires passing the sequence back and forth between the two.

Given a prompt (tokenized, post-processed if necessary) in `prompt.csv`, samples
can be generated with:

```bash
python sample.py \
--checkpoint checkpoint.pkl \
--tokenizer $TOKENIZER/spm.model \
--input $DATA/valid.csv
```

## Citing this work

If you use this work for research, we ask to you kindly cite our TACL article:

```
@article{10.1162/tacl_a_00526,
    author = {Sartran, Laurent and Barrett, Samuel and Kuncoro, Adhiguna and Stanojević, Miloš and Blunsom, Phil and Dyer, Chris},
    title = "{Transformer Grammars: Augmenting Transformer Language Models with Syntactic Inductive Biases at Scale}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {10},
    pages = {1423-1439},
    year = {2022},
    month = {12},
    abstract = "{We introduce Transformer Grammars (TGs), a novel class of Transformer language models that combine (i) the expressive power, scalability, and strong performance of Transformers and (ii) recursive syntactic compositions, which here are implemented through a special attention mask and deterministic transformation of the linearized tree. We find that TGs outperform various strong baselines on sentence-level language modeling perplexity, as well as on multiple syntax-sensitive language modeling evaluation metrics. Additionally, we find that the recursive syntactic composition bottleneck which represents each sentence as a single vector harms perplexity on document-level language modeling, providing evidence that a different kind of memory mechanism—one that is independent of composed syntactic representations—plays an important role in current successful models of long text.}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00526},
    url = {https://doi.org/10.1162/tacl\_a\_00526},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00526/2064617/tacl\_a\_00526.pdf},
}
```

## License and disclaimer

Copyright 2021-2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
