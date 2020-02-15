from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, Seq2VecEncoder,
                              TextFieldEmbedder)
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask

# TODO (John): This should be registerable so we can select hyperparameters in the config
from ..losses.n_pair_loss import NPairLoss


@Model.register("constrastive")
class ContrastiveTextEncoder(Model):
    """
    This `Model` implements a text encoder trained against a contrastive, self-supervised objective.
    After embedding the text into a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`.
    The resulting sequence is pooled using a `Seq2VecEncoder` and then passed to a `FeedFoward` layer, which
    projects the embeddings to a certain size. If a `Seq2SeqEncoder` is not provided, we will pass the embedded
    text directly to the `Seq2VecEncoder`. This model is based on modifications to the `BasicClassifier` in
    AllenNLP:  https://github.com/allenai/allennlp/blob/master/allennlp/models/basic_classifier.py.

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    feedforward : `FeedForward`, optional, (default = None).
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
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
        # Parameters

        anchor_tokens : TextFieldTensors
            From a `TextField`
        positive_tokens : torch.IntTensor, optional (default = None)
            From a `TextField`
        
        # Returns

        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape `(batch_size, self._seq2vec_encoder.get_output_dim())` if self._feedforward is None
            else `(batch_size, self._feedforward.get_output_dim())` representing the embedding for the given
            `tokens`.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_anchor_text = self._forward_internal(anchor_tokens)

        # This is the text embedding
        output_dict = {"embeddings": embedded_anchor_text}

        # To mine positives online, we need to
        # - add sentence segmentation to the dataset loader
        # - flatten the returned tokens when embedded the anchor
        # - for each anchor, choose a random sentence.
        # - embed these random sentences as positives.
        """
        if self.training:
            positive_tokens = random.choice(anchor_tokens)
            embedded_positive_text = self._forward_internal(positive_tokens)

            loss = self._loss(embedded_anchor_text, embedded_positive_text)
            output_dict["loss"] = loss
        """

        if positive_tokens is not None:
            embedded_positive_text = self._forward_internal(positive_tokens)

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

        return embedded_text
