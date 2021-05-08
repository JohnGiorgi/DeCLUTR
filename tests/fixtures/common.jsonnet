// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model. 
local transformer_model = "distilroberta-base";
// This will be used to set the max/min # of tokens in the positive and negative examples.
local max_length = 16;
local min_length = 8;

{
    "vocabulary": {
        "type": "empty"
    },
    "dataset_reader": {
        "type": "declutr.dataset_reader.DeCLUTRDatasetReader",
        "lazy": true,
        "num_anchors": 2,
        "num_positives": 2,
        "max_span_len": max_length,
        "min_span_len": min_length,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            "max_length": max_length,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
            },
        },
    }, 
    "train_data_path": "tests/fixtures/data/openwebtext/train.txt",
    "validation_data_path": "tests/fixtures/data/openwebtext/valid.txt",
    "model": {
        "type": "declutr.DeCLUTR",
    },
    "data_loader": {
        "batch_size": 4,
        "num_workers": 1,
        "drop_last": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "weight_decay": 0.1,
            "parameter_groups": [
                // Apply weight decay to pre-trained params, excluding LayerNorm params and biases
                [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}],
            ],
        },
        "num_epochs": 1,
        "checkpointer": {
            "num_serialized_models_to_keep": -1,
        },
        "grad_norm": 1.0,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
        },
    },
}