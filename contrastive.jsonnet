// TODO (John): General description of the model

// This should be a registered name in the Transformers library 
// (see https://huggingface.co/transformers/pretrained_models.html) or a path on disk to a
// serialized transformer model. Note, to avoid issues, please name the serialized model folder in
// the same format as the Transformers library, e.g.
// [bert|roberta|gpt2|distillbert|etc]-[base|large|etc]-[uncased|cased|etc]
local pretrained_transformer_model_name = "albert-base-v2";
// This will be used to set the max # of tokens in the anchor, positive and negative examples
local max_length = 512;
// TODO (John): Can we set this programatically?
// This corresponds to the config.hidden_size of the pretrained_transformer_model_name
local token_embedding_size = 768;
// Size of the fixed-length document embedding to be learned by the model
local document_embedding_size = 512;

{
    "dataset_reader": {
        "type": "contrastive",
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
                "max_length": max_length
            },
        },
    },
    "train_data_path": "datasets/pubmed/train.tsv",
    "validation_data_path": "datasets/pubmed/valid.tsv",
    "model": {
        "type": "constrastive",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": pretrained_transformer_model_name,
                    "max_length": max_length
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
            "num_layers": 1,
            "hidden_dims": document_embedding_size,
            # TODO (John): Need to experiment to determine if a non-linearity helps here.
            "activations": "linear",
            "dropout": 0.0
        },
    },
    "iterator": {
        "type": "bucket",
        // TODO (John): Ideally this would be much larger but there are OOM issues.
        "batch_size": 4
    },
    "validation_iterator": {
        "type": "bucket",
        "batch_size": 8
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 2e-5,
            "weight_decay": 0.0,
            /*
            "parameter_groups": [
                # Apply weight decay to pre-trained parameters, exlcuding LayerNorm parameters and biases
                # See: https://github.com/huggingface/transformers/blob/2184f87003c18ad8a172ecab9a821626522cf8e7/examples/run_ner.py#L105
                # Regex: https://regex101.com/r/ZUyDgR/3
                [["(?=.*transformer_model)(?=.*\.+)(?!.*(LayerNorm|bias)).*$"], {"weight_decay": 0.01}],
                # Apply a different (smaller) learning rate to the seq2seq encoder as it is pre-trained
                # TODO (John): This will probably break because the parameters exist in two groups.
                [["token_embedders"], {"lr": 2e-5}],
            ],
            */
        },
        "patience": 5,
        "num_epochs": 25,
        "checkpointer": {
            "num_serialized_models_to_keep": 1,
        },
        "cuda_device": 0,
        "grad_norm": 1.0,
        // The effective batch size is batch_size * num_gradient_accumulation_steps
        "num_gradient_accumulation_steps": 4
    }
}
