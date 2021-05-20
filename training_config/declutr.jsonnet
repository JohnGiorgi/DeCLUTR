// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model.
local transformer_model = std.extVar("TRANSFORMER_MODEL");

// This will be used to set the max/min # of tokens in the positive and negative examples.
local max_length = 512;
local min_length = 32;

{
    "vocabulary": {
        "type": "empty"
    },
    "dataset_reader": {
        "type": "declutr",
        "lazy": true,
        "num_anchors": 2,
        "num_positives": 2,
        "max_span_len": max_length,
        "min_span_len": min_length,
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
        "loss": {
            "type": "nt_xent",
            "temperature": 0.05,
        },
        // There was a small bug in the original implementation that caused gradients derived from
        // the contrastive loss to be scaled by 1/N, where N is the number of GPUs used during
        // training. This has been fixed. To reproduce results from the paper, set this to false.
        // Note that this will have no effect if you are not using distributed training with more
        // than 1 GPU.
        "scale_fix": false
    },
    "data_loader": {
        "batch_size": 4,
        "num_workers": 1,
        "drop_last": true,
    },
    "trainer": {
        // Set use_amp to true to use automatic mixed-precision during training (if your GPU supports it)
        "use_amp": true,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "eps": 1e-06,
            "correct_bias": false,
            "weight_decay": 0.1,
            "parameter_groups": [
                // Apply weight decay to pre-trained params, excluding LayerNorm params and biases
                [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}],
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