# t2t

A encoder-decoder model trained to reproduce the source text in order to learn useful document-level embeddings.

## Installation

To clone the repository locally

```
git clone https://github.com/JohnGiorgi/t2t.git
```

The only requirement is [AllenNLP](https://github.com/allenai/allennlp). For the time being, please [install from source](https://github.com/allenai/allennlp#installing-from-source).

## Usage

### Training

To train the model, run the following command

```
allennlp train t2t.jsonnet \
    -s ./tmp \
    --include-package t2t.modules.seq2seq_encoders.pretrained_transformer
```

During training, models, vocabulary configuration and log files will be saved to `"./tmp"`. This can be changed to any path you like.

### Inference

To perform inference with a trained model, run the following command

```
allennlp predict path/to/model.tar.gz path/to/input/file.tsv \
    --output-file path/to/predictions.json \
    --batch-size 16 \
    --cuda-device 0 \
    --use-dataset-reader \
    --dataset-reader-choice validation
    --predictor seq2seq \
    --include-package t2t.modules.seq2seq_encoders.pretrained_transformer \
```

The document embeddings will be stored as lists under the field `"embeddings"` in the `output-file`.

> It is important that you provided a validation iterator during training that _does not shuffle_ the data. Because the `output-file` of the `predict` command will not contain the input text, we rely on the fact that the input texts in the input file and the predictions in the output file are in the same order.