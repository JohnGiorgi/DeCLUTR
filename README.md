# Contrastive Self-supervision for Sentence and Document Embedding

A contrastive, self-supervised method for sentence and document embedding.

## Installation

To clone the repository locally

```
git clone https://github.com/JohnGiorgi/t2t.git
```

The only requirement is [AllenNLP](https://github.com/allenai/allennlp). For the time being, please [install from source](https://github.com/allenai/allennlp#installing-from-source).

## Usage

### Preparing a Dataset

Datasets should be text files where each line contains a raw text sequence. You can specify different partitions in the config (the default config is `contrastive.jsonnet`) under `"train_data_path"`, `"validation_data_path"` and `"test_data_path"`.

### Training

To train the model, run the following command

```
allennlp train contrastive.jsonnet -s tmp --include-package t2t
```

During training, models, vocabulary, configuration and log files will be saved to `tmp`. This can be changed to any path you like.

### Embedding

To embed text with a trained model, run the following command

```
allennlp predict tmp path/to/input/file.txt \
 --output-file tmp/embeddings.jsonl \
 --weights-file tmp/best.th \
 --batch-size 32 \
 --cuda-device 0 \
 --use-dataset-reader \
 --dataset-reader-choice validation \
 --include-package t2t
```

This will:
* load the model serialized to `tmp` with the weights from the epoch that achieved the best performance on the validation set
* use that model to perform inference on the provided input file
* save the predictions to disk as a [JSON lines](http://jsonlines.org/) file (`tmp/embeddings.jsonl`)

The sentence and/or document embeddings are stored in the field `"embeddings"` in `embeddings.json`.