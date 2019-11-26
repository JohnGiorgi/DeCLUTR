// This should be either a registered name in the Transformers library, or a path on disk to a
// serialized transformer model. Note, to avoid issues, please name the serialized model folder
// [bert|roberta|gpt2|distillbert|etc]-[base|large|etc]-[uncased|cased|etc]
local pretrained_transformer_model_name = "roberta-base";
// This will be used to set the max # of tokens and the max # of decoding steps
local max_sequence_length = 512;

{
    "dataset_reader": {
        "lazy": false,
        // TODO (John): Because our source and target text is identical, we should subclass this
        // dataloader to one which only expects one document per line.
        "type": "seq2seq",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": pretrained_transformer_model_name,
            "max_length": max_sequence_length,
        },
        "target_tokenizer": {
            "type": "spacy"
        },
        // TODO (John): For now, use different namespaces for source and target indexers. It may
        // make more sense to use the same namespaces in the future.
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": pretrained_transformer_model_name,
            },
        },
        "target_token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": "target_tokens"
            },
        },
        // This will break the pretrained_transformer token indexer. So remove for now.
        // In the future, we should assigned the special start and end sequence tokens to one of
        // BERTs unused vocab ids.
        // See: https://github.com/allenai/allennlp/issues/3435#issuecomment-558668277
        "source_add_start_token": false,
        "source_add_end_token": false,
    },
    "train_data_path": "datasets/pubmed/train.tsv",
    "validation_data_path": "datasets/pubmed/valid.tsv",
    "model": {
        "type": "pretrained_transformer",
        "encoder": {
            "type": "pretrained_transformer",
            "model_name": pretrained_transformer_model_name,
        },
        "target_namespace": "target_tokens",
        "max_decoding_steps": max_sequence_length,
        "beam_size": 8,
        "target_embedding_dim": 256,
        "use_bleu": true
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["source_tokens", "num_tokens"]],
        "batch_size": 1
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            // TODO (John): Because our decoder is trained from scratch, we will likely need a larger
            // learning rate. Idea, diff learning rates for encoder / decoder?
            "lr": 2e-5
        },
        "validation_metric": "+bleu",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 5,
        "cuda_device": 0,
        "grad_norm": 1.0,
    }
}