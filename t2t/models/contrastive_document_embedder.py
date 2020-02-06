from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, Seq2VecEncoder,
                              TextFieldEmbedder)
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask

# TODO (John): This should be registerable
from ..losses.n_pair_loss import NPairLoss


@Model.register("constrastive")
class ContrastiveDocumentEmbedder(Model):
    """
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._loss = NPairLoss()
        initializer(self)

    def forward(  # type: ignore
        self, anchor_tokens: TextFieldTensors, positive_tokens: TextFieldTensors = None
    ) -> Dict[str, torch.Tensor]:

        """
        """
        embedded_anchor_text, _ = self._forward_internal(anchor_tokens)

        # This is the document embedding
        output_dict = {"logits": embedded_anchor_text}

        if positive_tokens is not None:
            embedded_positive_text, _ = self._forward_internal(positive_tokens)

            loss = self._loss(embedded_anchor_text, embedded_positive_text)
            output_dict["loss"] = loss

        return output_dict

    def _forward_internal(self, tokens: TextFieldTensors):
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        return embedded_text, mask
