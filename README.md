# Contrastive Learning of Textual Representations

A contrastive, self-supervised method for learning textual representations.

> This is a work in progress!

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

> For the time being, please install [AllenNLP](https://github.com/allenai/allennlp) [from source](https://github.com/allenai/allennlp#installing-from-source). You should also install [PyTorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-zone) support by following the instructions for your system [here](https://pytorch.org/get-started/locally/).

## Usage

### Preparing a dataset

Datasets should be text files where each line contains a raw text sequence. You can specify different partitions in the config (the default config is `contrastive.jsonnet`) under `"train_data_path"`, `"validation_data_path"` and `"test_data_path"`.

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
 --overrides '{"dataset_reader.sample_spans": false}' \
 --include-package t2t
```

This will:

1. Load the model serialized to `tmp` with the weights from the epoch that achieved the best performance on the validation set.
2. Use that model to embed the text in the provided input file (`path/to/input.txt`).
3. Save the embeddings to disk as a [JSON lines](http://jsonlines.org/) file (`tmp/embeddings.jsonl`)

The text embeddings are stored in the field `"embeddings"` in `tmp/embeddings.jsonl`.

> If your model was trained with a `FeedForward` module, it will also contain a field named `"projections"`. A `FeedForward` module with a non-linear transformation [may improve the quality of representations learned by the encoder network](https://arxiv.org/abs/2002.05709).