# t2t

A encoder-decoder model trained to reproduce the source text in order to learn useful document-level embeddings.

## Installation

To clone the repository locally

```
git clone https://github.com/JohnGiorgi/t2t.git
```

The only requirement is [AllenNLP](https://github.com/allenai/allennlp). For the time being, please [install from source](https://github.com/allenai/allennlp#installing-from-source).

## Usage

### Preparing a Dataset

Datasets should be `tsv` files, where each line contains a pair of document seperated by a tab character (the source and target texts). You can specify different partitions in the config (the default config is `t2t.jsonnet`) under `"train_data_path"`, `"validation_data_path"` and `"test_data_path"`.

### Training

To train the model, run the following command

```
allennlp train t2t.jsonnet \
    -s tmp \
    --include-package t2t.modules.seq2seq_encoders.pretrained_transformer
    --include-package t2t.models.encoder_decoders.composed_seq2seq_with_doc_embeddings
```

During training, models, vocabulary configuration and log files will be saved to `tmp`. This can be changed to any path you like.

### Inference

To perform inference with a trained model, run the following command

```
allennlp predict tmp path/to/input/file.tsv \
    --output-file tmp/predictions.jsonl \
    --weights-file tmp/best.th \
    --batch-size 32 \
    --cuda-device 0 \
    --use-dataset-reader \
    --dataset-reader-choice validation \
    --predictor seq2seq \
    --include-package t2t.modules.seq2seq_encoders.pretrained_transformer \
    --include-package t2t.models.encoder_decoders.composed_seq2seq_with_doc_embeddings
```

This will:
* load the model serialized to `tmp` with the weights from the epoch that achieved the best performance on the validation set
* use that model to perform inference on the provided input file
* save the predictions to disk as a [JSON lines](http://jsonlines.org/) file (`tmp/predictions.jsonl`)

The document embeddings are stored in the field `"doc_embeddings"` in `predictions.json`.