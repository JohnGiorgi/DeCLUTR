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
    -s tmp \
    --include-package t2t.modules.seq2seq_encoders.pretrained_transformer
```

During training, models, vocabulary configuration and log files will be saved to `"tmp"`. This can be changed to any path you like.

### Inference

To perform inference with a trained model, run the following command

```
allennlp predict tmp path/to/input/file.tsv \
    --output-file tmp/predictions.json \
    --weights-file tmp/best.th \
    --batch-size 16 \
    --cuda-device 0 \
    --use-dataset-reader \
    --dataset-reader-choice validation
    --predictor seq2seq \
    --include-package t2t.modules.seq2seq_encoders.pretrained_transformer \
```

This will:
* load the model serialized to `"./tmp"` with the weights from the epoch that achieved the best performance on the validation set
* use that model to perform inference on the provided input file
* save the predictions to disk as `tmp/predictions.json`