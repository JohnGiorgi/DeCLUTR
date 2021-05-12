// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model.
local transformer_model = std.extVar("TRANSFORMER_MODEL");

// This will be used to set the max/min # of tokens in the positive and negative examples.
local max_length = 512;

{
    "vocabulary": {
        "type": "empty"
    },
    "dataset_reader": {
        "type": "declutr",
        "lazy": true,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            // Account for special tokens (e.g. CLS and SEP), otherwise a cryptic error is thrown.
            "max_length": max_length - 2,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
            },
        },
    }, 
    "train_data_path": null,
    "model": {
        "type": "declutr",
        "text_field_embedder": {
            "type": "mlm",
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mlm",
                    "model_name": transformer_model,
                    "masked_language_modeling": true
                },
            },
        },
    },
    "data_loader": {
        "batch_size": 16,
        "num_workers": 1,
        "drop_last": true,
    },
    "trainer": {
        "type": "no_op"
    },
}