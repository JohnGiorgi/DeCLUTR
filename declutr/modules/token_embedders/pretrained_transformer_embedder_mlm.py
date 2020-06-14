from typing import Optional, Tuple, Union

import torch
from overrides import overrides
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("pretrained_transformer_mlm")
class PretrainedTransformerEmbedderMLM(PretrainedTransformerEmbedder):
    """
    This is a wrapper around `PretrainedTransformerEmbedder` that allows us to train against a
    masked language modelling objective while we are embedding text.

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers
        naturally act as embedders such as BERT. However, other models consist of encoder and
        decoder, in which case we just want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    masked_language_modeling: `bool`, optional (default = `True`)
        If this is `True` and `masked_lm_labels is not None` in the call to `forward`, the model
        will be trained against a masked language modelling objective and the resulting loss will be
        returned along with the output tensor.
    output_hidden_states: `bool`, optional (default = `False`)
        If this is `True`, then the returned output tensor will contain the word-level
        embeddings and the output from every layer in the transformer. Corresponds to the
        `output_hidden_states` variable in the HuggingFace Transformers Config, see:
        https://huggingface.co/transformers/main_classes/configuration.html


    Registered as a `TokenEmbedder` with name "pretrained_transformer_mlm".
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
        masked_language_modeling: bool = True,
        output_hidden_states: bool = False,
    ) -> None:
        TokenEmbedder.__init__(self)  # Call the base class constructor

        self.masked_language_modeling = masked_language_modeling
        self._output_hidden_states = output_hidden_states

        tokenizer = PretrainedTransformerTokenizer(model_name)

        if self._output_hidden_states or self.masked_language_modeling:
            self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        else:
            self.config = AutoConfig.from_pretrained(model_name)

        if self.masked_language_modeling:
            # We only need access to the tokenizer if we are masked language modeling
            self.tokenizer = tokenizer
            # The only differences when masked language modeling are:
            # 1) `output_hidden_states` must be True to get access to token embeddings.
            # 2) We need to use `AutoModelWithLMHead` to get the correct model
            self.transformer_model = AutoModelWithLMHead.from_pretrained(
                model_name, config=self.config
            )

        else:
            self.transformer_model = AutoModel.from_pretrained(model_name, config=self.config)

        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)
        self._max_length = max_length
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size
        self._train_parameters = train_parameters

        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
        masked_lm_labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor, torch.Tensor], torch.Tensor]:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
        segment_concat_mask: `Optional[torch.BoolTensor]`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.
        masked_lm_labels: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces]`.

        # Returns:

        If `self.masked_language_modeling`, returns a `Tuple` of the masked language modeling loss
        and a `torch.Tensor` of shape: `[batch_size, num_wordpieces, embedding_size]`.

        If `self._output_hidden_states`, the returned tensor will be of shape:
        `[num_layers + 1, batch_size, num_wordpieces, embedding_size]`. See `output_hidden_states`
        in https://huggingface.co/transformers/main_classes/configuration.html for more info.
        """

        with torch.set_grad_enabled(self._train_parameters):
            # Some of the huggingface transformers don't support type ids at all and crash when you
            # supply them. For others, you can supply a tensor of zeros, and if you don't, they act
            # as if you did. There is no practical difference to the caller, so here we pretend that
            # one case is the same as another case.
            if type_ids is not None:
                max_type_id = type_ids.max()
                if max_type_id == 0:
                    type_ids = None
                else:
                    if max_type_id >= self._number_of_token_type_embeddings():
                        raise ValueError(
                            "Found type ids too large for the chosen transformer model."
                        )
                    assert token_ids.shape == type_ids.shape

            fold_long_sequences = (
                self._max_length is not None and token_ids.size(1) > self._max_length
            )
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
                # Even if masked_language_modeling is True, we may not be masked language modeling
                # on the current batch. We still need to check if masked language modeling labels
                # are present in the input.
                if masked_lm_labels is not None:
                    parameters["masked_lm_labels"] = masked_lm_labels
                    masked_lm_loss, _, hidden_states = self.transformer_model(**parameters)
                else:
                    _, hidden_states = self.transformer_model(**parameters)
                embeddings = (
                    torch.stack(hidden_states) if self._output_hidden_states else hidden_states[-1]
                )
            else:
                embeddings = (
                    torch.stack(self.transformer_model(**parameters)[2])
                    if self._output_hidden_states
                    else self.transformer_model(**parameters)[0]
                )

            # I do not know if this code will break when `self._output_hidden_states=True`.
            # We never use it in this repo, but if it causes problems, raise a ValueError.
            if fold_long_sequences:
                embeddings = self._unfold_long_sequences(
                    embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces
                )

            return masked_lm_loss, embeddings
