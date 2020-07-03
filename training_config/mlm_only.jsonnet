// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model.
local transformer_model = std.extVar("TRANSFORMER_MODEL");

// This will be used to set the max/min # of tokens in the positive and negative examples.
local max_length = 512;
local min_length = 32;

{
    "dataset_reader": {
        "type": "declutr",
        "lazy": true,
        // Technically, we don't need to sample anchors or positives when training with MLM only.
        // However, to make this experiment as comparable as possible to the "Contrastive only"
        // and "Both" experiments, we sample the same number of anchors and MLM on all of them.
        "num_anchors": 2,
        "num_positives": 1,
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
        "batch_size": 4,
        "drop_last": true,
    },
    "trainer": {
        // If Apex is installed, chose one of its opt_levels here to use mixed-precision training.
        "opt_level": null,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "weight_decay": 0.0,
            "parameter_groups": [
                # Apply weight decay to pre-trained params, excluding LayerNorm params and biases
                # See: https://github.com/huggingface/transformers/blob/2184f87003c18ad8a172ecab9a821626522cf8e7/examples/run_ner.py#L105
                # Regex: https://regex101.com/r/ZUyDgR/3/tests
                [["(?=.*transformer_model)(?=.*\\.+)(?!.*(LayerNorm|bias)).*$"], {"weight_decay": 0.1}],
            ],
        },
        "num_epochs": 1,
        "checkpointer": {
            // A value of null or -1 will save the weights of the model at the end of every epoch
            "num_serialized_models_to_keep": -1,
        },
        "grad_norm": 1.0,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
        },
    },
}