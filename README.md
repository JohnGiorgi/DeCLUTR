![build](https://github.com/JohnGiorgi/declutr/workflows/build/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/JohnGiorgi/DeCLUTR/branch/master/graph/badge.svg)](https://codecov.io/gh/JohnGiorgi/DeCLUTR)

# DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations

The corresponding code for our paper: [DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations](https://arxiv.org/abs/2006.03659v2). Results on [SentEval](https://github.com/facebookresearch/SentEval) are presented below (as averaged scores on the downstream and probing task dev sets), along with existing state-of-the-art methods.

| Model                                                                                                      | Requires labelled data? | Parameters | Embed. dim. | Downstream |  Probing  |    Avg.   |   Δ   |
|------------------------------------------------------------------------------------------------------------|:-----------------------:|:----------:|:-----------:|:----------:|:---------:|:---------:|:-----:|
| [InferSent V2](https://github.com/facebookresearch/InferSent)                                              |           Yes           |     38M    |     4096    |    76.46   |   72.58   |   74.52   | -1.40 |
| [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-large/5)                  |           Yes           |    147M    |     512     |  __79.13__ |   66.70   |   72.91   | -3.00 |
| [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)  ("roberta-base-nli-mean-tokens") |           Yes           |    125M    |     768     |    77.59   |   63.22   |   70.40   | -5.52 |
| Transformer-small ([DistilRoBERTa-base](https://huggingface.co/distilroberta-base))                        |            No           |     82M    |     768     |    72.69   | __74.27__ |   73.48   | -2.44 |
| Transformer-base ([RoBERTa-base](https://huggingface.co/roberta-base))                                     |            No           |    125M    |     768     |    72.22   |   73.38   |   72.80   | -3.12 |
| DeCLUTR-small ([DistilRoBERTa-base](https://huggingface.co/distilroberta-base))                            |            No           |     82M    |     768     |    76.43   |   73.82   |   75.13   | -0.79 |
| DeCLUTR-base ([RoBERTa-base](https://huggingface.co/roberta-base))                                         |            No           |    125M    |     768     |    78.17   |   73.67   | __75.92__ |   --  |

> Transformer-* is the same underlying architecture and pretrained weights as DeCLUTR-* _before_ continued training with our contrastive objective. Transformer-* and DeCLUTR-* use mean pooling on their token-level embeddings to produce a fixed-length sentence representation.

## Notebooks

The easiest way to get started is to follow along with one of our [notebooks](notebooks):

- Training your own model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnGiorgi/DeCLUTR/blob/master/notebooks/training.ipynb)
- Embedding text with a pretrained model (:soon:)

## Installation

This repository requires Python 3.6 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. See [here](https://github.com/allenai/allennlp#installing-via-pip) for detailed instructions.

### Installing the library and dependencies

First, clone the repository locally

```bash
git clone https://github.com/JohnGiorgi/DeCLUTR.git
```

Then, install

```bash
cd DeCLUTR
pip install --editable .
```

#### Gotchas

- For the time being, please install [AllenNLP](https://github.com/allenai/allennlp) [from source](https://github.com/allenai/allennlp#installing-from-source). Specifically, please install [this commit](https://github.com/allenai/allennlp/commit/9766eb407e7d83a0bf2150ad054a7c8e2da4ae2b).
- If you plan on training your own model, you should also install [PyTorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-zone) support by following the instructions for your system [here](https://pytorch.org/get-started/locally/).

## Usage

### Preparing a dataset

Datasets should be text files where each line contains a raw text sequence. You can specify different partitions in the [configs](configs) under `"train_data_path"`, `"validation_data_path"` and `"test_data_path"`.

We provide scripts to download some popular datasets and prepare them for training with our model. For example, to download [WikiText-103](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) (and match our minimal preprocessing), you can call

```bash
python scripts/preprocess_wikitext_103.py path/to/output/wikitext-103/train.txt --min-length 2048
```

#### Gotchas

- A training dataset should contain documents with a minimum of `num_anchors * max_span_len * 2` whitespace tokens. This is required to sample spans according to our sampling procedure. See the [dataset reader](declutr/dataset_readers/declutr.py) for more details on these hyperparameters.

### Training

To train the model, run the following command

```bash
allennlp train configs/contrastive_simple.jsonnet \
    --serialization-dir output \
    --overrides "{'train_data_path': 'path/to/input.txt'}" \
    --include-package declutr
```

During training, models, vocabulary, configuration, and log files will be saved to `output`. This can be changed to any path you like.

> Note, we provide a second config, [`contrastive.jsonnet`](configs/contrastive.jsonnet) that requires slightly more configuration but leads to slightly better scores.

#### Multi-GPU training

To train on more than one GPU, provide a list of CUDA devices in your call to `allennlp train`. For example, to train with four CUDA devices with IDs `0, 1, 2, 3`

```bash
allennlp train configs/contrastive_simple.jsonnet \
    --serialization-dir output \
    --overrides "{'train_data_path': 'path/to/input.txt', 'distributed.cuda_devices': [0, 1, 2, 3]}" \
    --include-package declutr
```

> You can also add this to a [config](configs) if you prefer.

#### Training with mixed-precision

If you want to train with [mixed-precision](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) (strongly recommended if your GPU supports it), you will need to [install Apex with CUDA and C++ extensions](https://github.com/NVIDIA/apex#quick-start). Once installed, you need only to set `"opt_level"` to `"O1"` in your training [config](configs), or, equivalently, pass the following flag to `allennlp train`

```bash
--overrides "{'trainer.opt_level': 'O1'}"
```

> You can also add this to a [config](configs) if you prefer.

#### Gotchas

- Mixed-precision training will cause an error with the [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning) library. See [here](https://github.com/JohnGiorgi/DeCLUTR/issues/60) for a discussion on the issue, along with the suggested fix.

### Embedding

You can embed text with a trained model in one of three ways:

1. [As a library](#as-a-library): import and initialize an object from this repo, which can be used to embed sentences/paragraphs.
2. [HuggingFace Transformers](#huggingface-transformers): load our pretrained model with the [HuggingFace transformers library](https://github.com/huggingface/transformers).
3. [Bulk embed](#bulk-embed-a-file): emded all text in a given text file with a simple command-line interface.

#### As a library

To use the model as a library, import `Encoder` and pass it some text (it accepts both strings and lists of strings)

```python
from declutr import Encoder

encoder = Encoder("path/to/serialized/model")
embeddings = encoder([
    "A smiling costumed woman is holding an umbrella.",
    "A happy woman in a fairy costume holds an umbrella."
])
```

these embeddings can then be used, for example, to compute the semantic similarity between some number of sentences or paragraphs

```python
from scipy.spatial.distance import cosine

semantic_sim = 1 - cosine(embeddings[0], embeddings[1])
```

> In the future, we will host pre-trained weights online, so that a model name can be passed to `Encoder` and the model will be automatically downloaded.

#### HuggingFace Transformers

Our pretrained models are also hosted with HuggingFace Transformers, so they can be used like any other model in that library. Here is a simple example:

```python
import torch
from scipy.spatial.distance import cosine

from transformers import AutoModelWithLMHead, AutoTokenizer

# Load the model
tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-small")
model = AutoModelWithLMHead.from_pretrained("johngiorgi/declutr-small")

# Prepare some text to embed
text = [
    "A smiling costumed woman is holding an umbrella.",
    "A happy woman in a fairy costume holds an umbrella.",
]
inputs = tokenizer.batch_encode_plus(
    text, pad_to_max_length=True, max_length=512, return_tensors="pt"
)

# Embed the text
with torch.no_grad():
    _, hidden_states = model(**inputs)
# Mean pool the token-level embeddings to get sentence-level embeddings
embeddings = torch.sum(
    hidden_states[-1] * inputs["attention_mask"].unsqueeze(-1), dim=1
) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)

# Compute a semantic similarity
semantic_sim = 1 - cosine(embeddings[0], embeddings[1])
```

Currently available models:

- [johngiorgi/declutr-small](https://huggingface.co/johngiorgi/declutr-small)
- johngiorgi/declutr-base (:soon:)

#### Bulk embed a file

To embed all text in a given file with a trained model, run the following command

```bash
allennlp predict output path/to/input.txt \
 --output-file output/embeddings.jsonl \
 --batch-size 32 \
 --cuda-device 0 \
 --use-dataset-reader \
 --include-package declutr
```

This will:

1. Load the model serialized to `output` with the "best" weights (i.e. the ones that achieved the lowest loss during training).
2. Use that model to embed the text in the provided input file (`path/to/input.txt`).
3. Save the embeddings to disk as a [JSON lines](http://jsonlines.org/) file (`output/embeddings.jsonl`)

The text embeddings are stored in the field `"embeddings"` in `output/embeddings.jsonl`.

### Evaluating with SentEval

[SentEval](https://github.com/facebookresearch/SentEval) is a library for evaluating the quality of sentence embeddings. We provide a script to evaluate our model against SentEval.

First, clone the SentEval repository and download the transfer task datasets (you only need to do this once)

```bash
git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/data/downstream/
./get_transfer_data.bash
cd ../../../
```

> See the [SentEval](https://github.com/facebookresearch/SentEval) repository for full details.

Then you can run our [script](scripts/run_senteval.py) to evaluate a trained model against SentEval

```bash
python scripts/run_senteval.py allennlp SentEval output 
 --output-filepath output/senteval_results.json \
 --cuda-device 0  \
 --include-package declutr
```

The results will be saved to `output/senteval_results.json`. This can be changed to any path you like.

> Pass the flag `--prototyping-config` to get a proxy of the results while dramatically reducing computation time.

For a list of commands, run

```bash
python scripts/run_senteval.py --help
```

For help with a specific command, e.g. `allennlp`, run

```
python scripts/run_senteval.py allennlp --help
```

#### Gotchas

- Evaluating the `"SNLI"` task of SentEval will fail without [this fix](https://github.com/facebookresearch/SentEval/pull/52).
