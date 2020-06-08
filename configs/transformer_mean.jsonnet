// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model.
local transformer_model = "distilroberta-base";
// The hidden size of the model, which can be found in its config as "hidden_size".
local transformer_dim = 768;
// This will be used to set the max/min # of tokens in the positive and negative examples.
local max_length = 512;

{
    "dataset_reader": {
        "type": "contrastive",
        "lazy": true,
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
        "type": "constrastive",
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
        // TODO (John): Currently, num_workers must be < 1 or we will end up loading the same data
        // more than once. I need to modify the dataloader according to:
        // https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        // in order to support multi-processing.
        "num_workers": 1,
        "drop_last": true,
    },
    "trainer": {
        "type": "no_op"
    },
}