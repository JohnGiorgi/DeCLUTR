from typing import Optional

import torch
from overrides import overrides
from transformers import AutoConfig, AutoModelWithLMHead

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("pretrained_transformer_mlm")
class PretrainedTransformerEmbedderMLM(PretrainedTransformerEmbedder):
    """
    This is a wrapper around `PretrainedTransformerEmbedder` that allows us to train against a
    masked language modelling objective while we are embedding text. I don't like that we had to
    modify this class and hope in the future that we can replace it with a model from the
    https://github.com/allenai/allennlp-models repo.

    Registered as a `TokenEmbedder` with name "pretrained_transformer_mlm".
    """

    def __init__(
        self, model_name: str, max_length: int = None, masked_language_modeling: bool = True
    ) -> None:
        super().__init__(model_name, max_length)
        self.masked_language_modeling = masked_language_modeling
        if self.masked_language_modeling:
            self.tokenizer = PretrainedTransformerTokenizer(model_name)
            # Models with LM heads may not include hidden states in their outputs by default.
            config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
            self.transformer_model = AutoModelWithLMHead.from_pretrained(model_name, config=config)

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
        masked_lm_labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: torch.LongTensor
            Shape: [
                batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces
            ].
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: torch.BoolTensor
            Shape: [batch_size, num_wordpieces].
        type_ids: Optional[torch.LongTensor]
            Shape: [
                batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces
            ].
        segment_concat_mask: Optional[torch.BoolTensor]
            Shape: [batch_size, num_segment_concat_wordpieces].
        masked_lm_labels: torch.LongTensor
            Shape: [
                batch_size, num_wordpieces
            ].

        # Returns:

        Shape: [batch_size, num_wordpieces, embedding_size].
        """

        # Some of the huggingface transformers don't support type ids at all and crash when you
        # supply them. For others, you can supply a tensor of zeros, and if you don't, they act as
        # if you did. There is no practical difference to the caller, so here we pretend that one
        # case is the same as another case.
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError("Found type ids too large for the chosen transformer model.")
                assert token_ids.shape == type_ids.shape

        fold_long_sequences = self._max_length is not None and token_ids.size(1) > self._max_length
        if fold_long_sequences:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids
            )

        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]

        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        parameters = {"input_ids": token_ids, "attention_mask": transformer_mask.float()}
        masked_lm_loss = None
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids
        if self.masked_language_modeling:
            # Even if masked_language_modeling is True, we may not be masked language modeling on
            # the current batch. We still need to check if masked language modeling labels are
            # present in the input.
            if masked_lm_labels is not None:
                parameters["masked_lm_labels"] = masked_lm_labels
                masked_lm_loss, _, hidden_states = self.transformer_model(**parameters)
            else:
                _, hidden_states = self.transformer_model(**parameters)
            embeddings = hidden_states[-1]
        else:
            embeddings = self.transformer_model(**parameters)[0]

        if fold_long_sequences:
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces
            )

        return masked_lm_loss, embeddings
