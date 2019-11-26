# t2t

A encoder-decoder model trained to reproduce the source text in order to learn useful document-level embeddings.

## Installation

To clone the repository locally

```
git clone https://github.com/JohnGiorgi/t2t.git
```

The only requirement is [AllenNLP](https://github.com/allenai/allennlp). Please [install from source](https://github.com/allenai/allennlp#installing-from-source).

## Usage

### Training

To train the model, run the following command

```
allennlp train t2t.jsonnet -s ./tmp --include-package t2t.models.pretrained_transformer_seq2seq --include-package t2t.modules.seq2seq_encoders.pretrained_transformer
```

During training, models, vocabulary configuration and log files will be saved to `'./tmp'`. This can be changed to any path you like.

### Inference

TODO.