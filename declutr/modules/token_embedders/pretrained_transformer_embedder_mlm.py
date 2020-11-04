from typing import Any, Dict, Optional, Tuple, Union

import torch
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from overrides import overrides
from transformers import AutoConfig, AutoModelForMaskedLM


@TokenEmbedder.register("pretrained_transformer_mlm")
class PretrainedTransformerEmbedderMLM(PretrainedTransformerEmbedder):
    """
    This is a wrapper around `PretrainedTransformerEmbedder` that allows us to train against a
    masked language modelling objective while we are embedding text.

    Registered as a `TokenEmbedder` with name "pretrained_transformer_mlm".

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
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    masked_language_modeling: `bool`, optional (default = `True`)
        If this is `True` and `masked_lm_labels is not None` in the call to `forward`, the model
        will be trained against a masked language modelling objective and the resulting loss will
        be returned along with the output tensor.
    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
        last_layer_only: bool = True,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        masked_language_modeling: bool = True,
    ) -> None:
        TokenEmbedder.__init__(self)  # Call the base class constructor
        tokenizer = PretrainedTransformerTokenizer(model_name, tokenizer_kwargs=tokenizer_kwargs)
        self.masked_language_modeling = masked_language_modeling

        if self.masked_language_modeling:
            self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
            # We only need access to the HF tokenizer if we are masked language modeling
            self.tokenizer = tokenizer.tokenizer
            # The only differences when masked language modeling are:
            # 1) `output_hidden_states` must be True to get access to token embeddings.
            # 2) We need to use `AutoModelForMaskedLM` to get the correct model
            self.transformer_model = AutoModelForMaskedLM.from_pretrained(
                model_name, config=self.config, **(transformer_kwargs or {})
            )
        # Eveything after the if statement (including the else) is copied directly from:
        # https://github.com/allenai/allennlp/blob/master/allennlp/modules/token_embedders/pretrained_transformer_embedder.py
        else:
            from allennlp.common import cached_transformers

            self.transformer_model = cached_transformers.get(
                model_name, True, override_weights_file, override_weights_strip_prefix
            )
            self.config = self.transformer_model.config

        if gradient_checkpointing is not None:
            self.transformer_model.config.update({"gradient_checkpointing": gradient_checkpointing})

        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)
        self._max_length = max_length

        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size

        self._scalar_mix: Optional[ScalarMix] = None
        if not last_layer_only:
            self._scalar_mix = ScalarMix(self.config.num_hidden_layers)
            self.config.output_hidden_states = True

        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

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
        and a `torch.Tensor` of shape: `[batch_size, num_wordpieces, embedding_size]`. Otherwise,
        returns only the `torch.Tensor` of shape: `[batch_size, num_wordpieces, embedding_size]`.
        """
        # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # There is no practical difference to the caller, so here we pretend that one case is the same
        # as another case.
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
        parameters = {"input_ids": token_ids, "attention_mask": transformer_mask.float()}  # type: ignore
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids
        if masked_lm_labels is not None and self.masked_language_modeling:
            parameters["labels"] = masked_lm_labels

        masked_lm_loss = None
        transformer_output = self.transformer_model(**parameters)

        if self.config.output_hidden_states:
            # Even if masked_language_modeling is True, we may not be masked language modeling on
            # the current batch. Check if masked language modeling labels are present in the input.
            if "labels" in parameters:
                masked_lm_loss = transformer_output[0]

            if self._scalar_mix:
                embeddings = self._scalar_mix(transformer_output[-1][1:])
            else:
                embeddings = transformer_output[-1][-1]
        else:
            embeddings = transformer_output[0]

        if fold_long_sequences:
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces
            )

        return masked_lm_loss, embeddings
