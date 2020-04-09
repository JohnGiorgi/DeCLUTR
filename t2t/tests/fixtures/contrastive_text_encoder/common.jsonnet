// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model. 
local transformer_model = "distilroberta-base";
// The hidden size of the model, which can be found in its config as "hidden_size".
local transformer_dim = 768;
// This will be used to set the max # of tokens in the positive and negative examples.
local max_length = 512;

{
    "dataset_reader": {
        "type": "t2t.data.dataset_readers.contrastive.ContrastiveDatasetReader",
        "lazy": true,
        "sample_spans": true,
        // This is (approximately) an upper bound on sentence length in English
        "min_span_len": 30,
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
    "train_data_path": "t2t/tests/fixtures/data/wikitext-103/train.txt",
    "validation_data_path": "t2t/tests/fixtures/data/wikitext-103/valid.txt",
    "model": {
        "type": "t2t.models.contrastive_text_encoder.ContrastiveTextEncoder",
        "seq2vec_encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": transformer_dim,
            "averaged": true
        },
    },
    "data_loader": {
        "batch_size": 5,
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
            "num_serialized_models_to_keep": -1,
        },
        "cuda_device": -1,
        "grad_norm": 1.0,
    },
}