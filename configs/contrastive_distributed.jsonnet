// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model. 
// Note, to avoid issues, please name the serialized model folder in roughly the same format as the
// Transformers library, e.g.
// [bert|roberta|gpt2|distillbert|etc]-[base|large|etc]-[uncased|cased|etc]
local pretrained_transformer_model_name = "distilroberta-base";
// This will be used to set the max # of tokens in the anchor, positive and negative examples.
local max_length = 512;
// This corresponds to PretrainedTransformerEmbedder.transformer_model.config.hidden_size
local token_embedding_size = 768;

{
    "dataset_reader": {
        "type": "contrastive",
        "lazy": true,
        "sample_spans": true,
        // This is (approximately an upper bound on sentence length in English
        "min_span_len": 30,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": pretrained_transformer_model_name,
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
    "train_data_path": "",
    "model": {
        "type": "constrastive",
        "text_field_embedder": {
            "type": "mlm",
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mlm",
                    "model_name": pretrained_transformer_model_name,
                    "masked_language_modeling": true
                },
            },
        },
        "seq2vec_encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": token_embedding_size,
            "averaged": true
        },
        "loss": {
            "type": "nt_xent",
            "temperature": 0.001,
        },
    },
    "data_loader": {
        "batch_size": 10,
        // TODO (John): Currently, num_workers must be < 1 or we will end up loading the same data more than once.
        // I need to modify the dataloader according to:
        // https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        // in order to support multi-processing.
        "num_workers": 1
    },
    "trainer": {
        // If you have installed Apex, you can chose one of its opt_levels here to use mixed precision training.
        "opt_level": null,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 2e-5,
            "weight_decay": 0.0,
            "parameter_groups": [
                # Apply weight decay to pre-trained parameters, exlcuding LayerNorm parameters and biases
                # See: https://github.com/huggingface/transformers/blob/2184f87003c18ad8a172ecab9a821626522cf8e7/examples/run_ner.py#L105
                # Regex: https://regex101.com/r/ZUyDgR/3/tests
                [["(?=.*transformer_model)(?=.*\\.+)(?!.*(LayerNorm|bias)).*$"], {"weight_decay": 0.1}],
            ],
        },
        "num_epochs": 1,
        "checkpointer": {
            "num_serialized_models_to_keep": 1,
        },
        "grad_norm": 1.0,
    },
    // This is where you would specify the CUDA devices to use for training
    "distributed" : {
        "cuda_devices": [0, 1],
    },
}