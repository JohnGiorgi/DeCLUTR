![build](https://github.com/JohnGiorgi/t2t/workflows/build/badge.svg?branch=master)

# Contrastive Self-supervision for Textual Representations

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

If you want to train with [mixed-precision](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) (strongly recommended if your GPU supports it), you will need to [install Apex with CUDA and C++ extensions](https://github.com/NVIDIA/apex#quick-start). Once installed, you need only to set `"opt_level"` to `"O1"` in your training [config](configs), or, equivalently, pass the following flag to `allennlp train`

```bash
--overrides '{"trainer.opt_level": "O1"}'
```

## Usage

### Preparing a dataset

Datasets should be text files where each line contains a raw text sequence. You can specify different partitions in the [configs](configs) under `"train_data_path"`, `"validation_data_path"` and `"test_data_path"`.

We provide scripts to download some popular datasets and prepare them for training with our model. For example, to download [WikiText-103](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) (with minimal preprocessing), you can call

```bash
python scripts/preprocess_wikitext_103.py path/to/output/wikitext-103/train.txt
```

### Training

To train the model, run the following command

```bash
allennlp train configs/contrastive.jsonnet \
    -s output \
    -o '{"train_data_path": "path/to/input.txt"}' \
    --include-package t2t
```

During training, models, vocabulary, configuration and log files will be saved to `output`. This can be changed to any path you like.

#### Multi-GPU training

To train on more than one GPU, provide a list of CUDA devices in your call to `allennlp train`. For example, to train with four CUDA devices with IDs `0, 1, 2, 3`

```bash
allennlp train configs/contrastive.jsonnet \
    -s output \
    -o '{"train_data_path": "path/to/input.txt", "distributed.cuda_devices": [0, 1, 2, 3]}' \
    --include-package t2t
```

> You can also add this to a [config](configs), if you prefer.

### Embedding

1. As a library (e.g. import and initialize an object which can be used to embed sentences/paragraphs).
2. Bulk embed all text in a given text file with a simple command-line interface.

#### As a library

To use the model "as a library," import `Encoder` and pass it some text (it accepts both strings and lists of strings)

```python
from t2t import Encoder

encoder = Encoder("path/to/serialized/model")
embeddings = encoder([
    "A smiling costumed woman is holding an umbrella.",
    "A happy woman in a fairy costume holds an umbrella."
])
```

these embeddings can then be used, for example, to compute the semantic similarity between some number of sentences or paragraphs

```python
from scipy.spatial.distance import cosine

semantic_sim = cosine(embeddings[0], embeddings[1])
```

> In the future, we will host pre-trained weights online, so that a model name can be passed to `Encoder` and the model will be automatically downloaded. 

#### Bulk embed a file

To embed all text in a given file with a trained model, run the following command

```bash
allennlp predict output path/to/input.txt \
 --output-file output/embeddings.jsonl \
 --batch-size 32 \
 --cuda-device 0 \
 --use-dataset-reader \
 --predictor "contrastive" \
 --include-package t2t
```

This will:

1. Load the model serialized to `output` with the "best" weights (i.e. the ones that achieved the lowest loss during training).
2. Use that model to embed the text in the provided input file (`path/to/input.txt`).
3. Save the embeddings to disk as a [JSON lines](http://jsonlines.org/) file (`output/embeddings.jsonl`)

The text embeddings are stored in the field `"embeddings"` in `output/embeddings.jsonl`.

> If your model was trained with a `FeedForward` module, it would also contain a field named `"projections"`. A `FeedForward` module with a non-linear transformation [may improve the quality of representations learned by the encoder network](https://arxiv.org/abs/2002.05709).

### Evaluating with SentEval

[SentEval](https://github.com/facebookresearch/SentEval) is a library for evaluating the quality of sentence embeddings. We provide a script to evaluate our model against SentEval.

First, clone the SentEval repository and download the transfer task datasets (you only need to do this once)

```bash
git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/data/downstream/
./get_transfer_data.bash
cd ../../../
```

> See the SentEval repository for full details.

Then you can run our [script](scripts/run_senteval.py) to evaluate a trained model against SentEval

```bash
python scripts/run_senteval.py allennlp SentEval output 
 --output-filepath output/senteval_results.json \
 --cuda-device 0  \
 --predictor-name "contrastive" \
 --include-package t2t
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
