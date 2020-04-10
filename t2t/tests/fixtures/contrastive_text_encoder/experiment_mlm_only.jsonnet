local COMMON = import 'common.jsonnet';
local transformer_model = "distilroberta-base";

{
    "dataset_reader": COMMON['dataset_reader'],
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": COMMON['train_data_path'],
    "validation_data_path": COMMON['validation_data_path'],
    "model": {
        "type": "t2t.models.contrastive_text_encoder.ContrastiveTextEncoder",
        "text_field_embedder": {
            "type": "t2t.modules.text_field_embedders.mlm_text_field_embedder.MLMTextFieldEmbedder",
            "token_embedders": {
                "tokens": {
                    "type": "t2t.modules.token_embedders.pretrained_transformer_embedder_mlm.PretrainedTransformerEmbedderMLM",
                    "model_name": transformer_model,
                    "masked_language_modeling": true
                },
            },
        },
        "seq2vec_encoder": COMMON['model']['seq2vec_encoder'],
        "loss": null
    },
    "data_loader": COMMON['data_loader'],
    "trainer": COMMON['trainer']
}