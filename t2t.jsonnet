// Specifies an encoder-decoder model trained to reconstruct the input text in order to produce
// rich document embeddings. Some inspiration was taken from the following AllenNLP configs:
// * https://github.com/allenai/allennlp/blob/2850579831f392467276f1ab6d5cda3fdb45c3ba/allennlp/tests/fixtures/encoder_decoder/composed_seq2seq/experiment_transformer.json
// * https://github.com/allenai/allennlp/blob/2850579831f392467276f1ab6d5cda3fdb45c3ba/allennlp/tests/fixtures/encoder_decoder/composed_seq2seq/experiment_lstm.json

// This should be a registered name in the Transformers library 
// (see https://huggingface.co/transformers/pretrained_models.html) or a path on disk to a
// serialized transformer model. Note, to avoid issues, please name the serialized model folder in
// the same format as the Transformers library, e.g.
// [bert|roberta|gpt2|distillbert|etc]-[base|large|etc]-[uncased|cased|etc]
local pretrained_transformer_model_name = "bert-base-uncased";
// This will be used to set the max # of source tokens and the max # of decoding steps
local max_sequence_length = 512;
// This corresponds to the config.hidden_size of the pretrained_transformer_model_name
// TODO (John): Can we set this programatically?
local transformer_embedding_size = 768;
// Whether or not tokens should be lowercased. Should match pretrained_transformer_model_name.
local do_lowercase = true;

{
    "dataset_reader": {
        // TODO (John): Because our source and target text is identical, we should subclass this
        // dataloader to one which only expects one document per line.
        "type": "seq2seq",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": pretrained_transformer_model_name,
            "max_length": max_sequence_length,
        },
        "target_tokenizer": {
            "type": "spacy"
        },
        // TODO (John): For now, we are using different token namespaces for the source and target
        // text. Not sure this makes sense, as the vocabs are the same. Experiement with merging.
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": pretrained_transformer_model_name,
                "namespace": "source_tokens"
            },
        },
        "target_token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": do_lowercase,
                "namespace": "target_tokens"
            },
        },
        // TODO (John): This will break the pretrained_transformer token indexer. So remove for now.
        // In the future, we might want to consider assigning the special start and end sequence
        // tokens to one of BERTs unused vocab ids.
        // See: https://github.com/allenai/allennlp/issues/3435#issuecomment-558668277
        "source_add_start_token": false,
        "source_add_end_token": false,
        "source_max_tokens": max_sequence_length,
        "target_max_tokens": max_sequence_length - 2,  // Account for start/end tokens
    },
    "train_data_path": "datasets/pubmed/train.tsv",
    "validation_data_path": "datasets/pubmed/valid.tsv",
    "model": {
        "type": "composed_seq2seq",
        "source_text_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pass_through",
                    "hidden_dim": transformer_embedding_size  
                },
            },
        },
        "encoder": {
            "type": "pretrained_transformer",
            "model_name": pretrained_transformer_model_name,
        },
        "decoder": {
            "decoder_net": {
                // These hyperparameters should be chosen to minimize the decoders capacity, while 
                // still allowing it to learn coherent reconstructions
                "type": "stacked_self_attention",
                "decoding_dim": transformer_embedding_size,
                "target_embedding_dim": transformer_embedding_size,
                "feedforward_hidden_dim": 1024,
                "num_layers": 1,
                "num_attention_heads": 1,
                "positional_encoding_max_steps": max_sequence_length
            },
            "max_decoding_steps": max_sequence_length,
            "target_embedder": {
                "embedding_dim": transformer_embedding_size,
                "vocab_namespace": "target_tokens"
            },
            "target_namespace": "target_tokens",
            // Use greedy decoding
            "beam_size": 1
        },
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["source_tokens", "num_tokens"]],
        "batch_size": 4
    },
    // By using this iterator at train time, we can provide it to the AllenNLP predict command
    // to ensure that instances are not shuffled at inference time.
    "validation_iterator": {
        "type": "basic",
        "batch_size": 16
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            // TODO (John): Because our decoder is trained from scratch, we will likely need a larger
            // learning rate. Idea: different learning rates for encoder / decoder?
            "lr": 5e-5
        },
        // "validation_metric": "-loss",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 10,
        "cuda_device": 0,
        "grad_norm": 1.0
    }
}