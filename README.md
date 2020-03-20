![build](https://github.com/JohnGiorgi/t2t/workflows/build/badge.svg?branch=master)

# Contrastive Learning of Textual Representations

A contrastive, self-supervised method for learning textual representations.

## Installation

This repository requires Python 3.7 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. See [here](https://github.com/allenai/allennlp#installing-via-pip) for detailed instructions.

### Installing the library and dependencies

First, clone the repository locally

```bash
git clone https://github.com/JohnGiorgi/t2t.git
```

Then, install

```bash
cd t2t
pip install --editable .
```

For the time being, please install [AllenNLP](https://github.com/allenai/allennlp) [from source](https://github.com/allenai/allennlp#installing-from-source). You should also install [PyTorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-zone) support by following the instructions for your system [here](https://pytorch.org/get-started/locally/).

#### Enabling mixed-precision training

If you want to train with [mixed-precision](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) (strongly recommended if your GPU supports it), you will need to [install Apex with CUDA and C++ extensions](https://github.com/NVIDIA/apex#quick-start). Once installed, you need only to set `"opt_level"` to `"O1"` in your training [config](configs).

## Usage

### Preparing a dataset

Datasets should be text files where each line contains a raw text sequence. You can specify different partitions in the [configs](configs) under `"train_data_path"`, `"validation_data_path"` and `"test_data_path"`.

### Training

To train the model, run the following command

```bash
allennlp train contrastive.jsonnet -s tmp --include-package t2t
```

During training, models, vocabulary, configuration and log files will be saved to `tmp`. This can be changed to any path you like.

### Embedding

To embed text with a trained model, run the following command

```bash
allennlp predict tmp path/to/input.txt \
 --output-file tmp/embeddings.jsonl \
 --batch-size 32 \
 --cuda-device 0 \
 --use-dataset-reader \
 --predictor "contrastive" \
 --include-package t2t
```

This will:

1. Load the model serialized to `tmp` with the weights from the epoch that achieved the best performance on the validation set.
2. Use that model to embed the text in the provided input file (`path/to/input.txt`).
3. Save the embeddings to disk as a [JSON lines](http://jsonlines.org/) file (`tmp/embeddings.jsonl`)

The text embeddings are stored in the field `"embeddings"` in `tmp/embeddings.jsonl`.

> If your model was trained with a `FeedForward` module, it will also contain a field named `"projections"`. A `FeedForward` module with a non-linear transformation [may improve the quality of representations learned by the encoder network](https://arxiv.org/abs/2002.05709).

### Evaluating with SentEval

[SentEval](https://github.com/facebookresearch/SentEval) is a library for evaluating the quality of sentence embeddings. We provide a script to easily evaluate our model against SentEval.

First, clone the SentEval repository and download the transfer task datasets (you only need to do this once)

```bash
git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/data/downstream/
./get_transfer_data.bash
```

> See the SentEval repository for full details.

Then you can run our [script](scripts/run_senteval.py) to evaluate a trained model against SentEval

```bash
python scripts/run_senteval.py allennlp SentEval tmp "contrastive" \
 --output-filepath tmp/senteval_results.json \
 --cuda-device 0  \
 --include-package t2t
```

The results will be saved to `tmp/senteval_results.json`. This can be changed to any path you like.

For a list of commands, run

```bash
python scripts/run_senteval.py --help
```

For help with a specific command, e.g. `allennlp`, run

```
python scripts/run_senteval.py allennlp --help
```
