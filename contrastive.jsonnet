// Whether or not you are training the model. This changes two things:
//  1. If true, then spans are extracted during the dataloading process for training against the contrastive
//     objective. Otherwise text is loaded as is.
//  2. If true, a non-linear transformation is added between the text representation and the contrastive loss.
local training = false;
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

// During training, this nonlinear transformation (inserted between the encoder network and the contrastive loss)
// will be learned.
// During inference, it is discarded and the textual representation is obtained from the encoder network only,
// which is composed of: text_field_embedder, seq2seq_encoder (optional) and seq2vec_encoder, in that order.
local projection_head = {
    "input_dim": token_embedding_size,
    "num_layers": 2,
    "hidden_dims": [128, 128],
    "activations": ["relu", "linear"],
};

{
    "dataset_reader": {
        "type": "contrastive",
        "extract_spans": training,
        "max_spans": 250,
        "min_span_width": 20,
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
    "train_data_path": "",
    "validation_data_path": "",
    "model": {
        "type": "constrastive",
        "temperature": 0.1,
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
        "feedforward": (if training then projection_head),
    },
    "iterator": {
        "type": "basic",
        // TODO (John): Ideally this would be much larger but there are OOM issues.
        "batch_size": 14,
    },
    "validation_iterator": {
        "type": "basic",
        "batch_size": 26
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 2e-5,
            "weight_decay": 0.0,
            "parameter_groups": [
                # Apply weight decay to pre-trained parameters, exlcuding LayerNorm parameters and biases
                # See: https://github.com/huggingface/transformers/blob/2184f87003c18ad8a172ecab9a821626522cf8e7/examples/run_ner.py#L105
                # Regex: https://regex101.com/r/ZUyDgR/3/tests
                [["(?=.*transformer_model)(?=.*\\.+)(?!.*(LayerNorm|bias)).*$"], {"weight_decay": 0.01}],
                # TODO (John): Apply a different (smaller) learning rate to the seq2seq encoder as it is pre-trained
            ],
        },
        "patience": 5,
        "num_epochs": 50,
        "checkpointer": {
            "num_serialized_models_to_keep": 1,
        },
        "cuda_device": 0,
        "grad_norm": 1.0,
        // The effective batch size is batch_size * num_gradient_accumulation_steps
        "num_gradient_accumulation_steps": 1
    }
}
