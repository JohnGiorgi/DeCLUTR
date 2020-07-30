local COMMON = import 'common.jsonnet';
local transformer_model = "distilroberta-base";

{
    "vocabulary": COMMON['vocabulary'],
    "dataset_reader": COMMON['dataset_reader'],
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": COMMON['train_data_path'],
    "validation_data_path": COMMON['validation_data_path'],
    "model": {
        "type": "declutr.DeCLUTR",
        "text_field_embedder": {
            "type": "declutr.modules.text_field_embedders.mlm_text_field_embedder.MLMTextFieldEmbedder",
            "token_embedders": {
                "tokens": {
                    "type": "declutr.modules.token_embedders.pretrained_transformer_embedder_mlm.PretrainedTransformerEmbedderMLM",
                    "model_name": transformer_model,
                    "last_layer_only": false,
                    "masked_language_modeling": true
                },
            },
        },
        "loss": {
            "type": "declutr.losses.pytorch_metric_learning.NTXentLoss",
            "temperature": 0.05,
        },
    },
    "data_loader": COMMON['data_loader'],
    "trainer": COMMON['trainer']
}