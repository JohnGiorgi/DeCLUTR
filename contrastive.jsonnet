// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model. 
// Note, to avoid issues, please name the serialized model folder in roughly the same format as the
// Transformers library, e.g.
// [bert|roberta|gpt2|distillbert|etc]-[base|large|etc]-[uncased|cased|etc]
local pretrained_transformer_model_name = "distilroberta-base";
// This will be used to set the max # of tokens in the anchor, positive and negative examples.
local max_length = 512;
// TODO (John): Can we set this programatically?
// This corresponds to the config.hidden_size of pretrained_transformer_model_name
local token_embedding_size = 768;

{
    "dataset_reader": {
        "type": "contrastive",
        "sample_spans": true,
        "max_spans": 15,
        "min_span_width": 15,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": pretrained_transformer_model_name,
            "add_special_tokens": true,
            "max_length": max_length,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": pretrained_transformer_model_name,
            },
        },
        // If not null, a cache of already-processed data will be stored in this directory.
        // If a cache file exists at this directory, it will be loaded instead of re-processing the data.
        "cache_directory": null
    },
    "train_data_path": "datasets/wikitext-103/train.txt",
    "model": {
        "type": "constrastive",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": pretrained_transformer_model_name,
                },
            },
        },
        "seq2vec_encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": token_embedding_size,
            "averaged": true
        },
        "feedforward": {
            "input_dim": token_embedding_size,
            "num_layers": 2,
            "hidden_dims": [128, 128],
            "activations": ["relu", "linear"],
        },
        "loss": {
            "type": "nt-xent",
            "temperature": 0.1,
            "normalize_embeddings": true
        },
    },
    "iterator": {
        "type": "basic",
        // TODO (John): Ideally this would be much larger but there are OOM issues.
        "batch_size": 20,
    },
    "trainer": {
        "type": "mixed-precision",
        "opt_level": "O1",
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 2e-5,
            "weight_decay": 0.0,
            "parameter_groups": [
                # Apply weight decay to pre-trained parameters, exlcuding LayerNorm parameters and biases
                # See: https://github.com/huggingface/transformers/blob/2184f87003c18ad8a172ecab9a821626522cf8e7/examples/run_ner.py#L105
                # Regex: https://regex101.com/r/ZUyDgR/3/tests
                [["(?=.*transformer_model)(?=.*\\.+)(?!.*(LayerNorm|bias)).*$"], {"weight_decay": 0.01}],
            ],
        },
        "num_epochs": 100,
        "checkpointer": {
            "num_serialized_models_to_keep": 1,
        },
        "cuda_device": 0,
        "grad_norm": 1.0,
    }
}