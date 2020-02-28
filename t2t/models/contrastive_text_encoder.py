from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    FeedForward,
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TextFieldEmbedder,
)
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from t2t.losses import PyTorchMetricLearningLoss
from t2t.models.util import sample_anchor_positive_pairs


@Model.register("constrastive")
class ContrastiveTextEncoder(Model):
    """
    This `Model` implements a text encoder trained against a contrastive, self-supervised objective.
    After embedding the text into a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`.
    The resulting sequence is pooled using a `Seq2VecEncoder` and then passed to a `FeedFoward` layer, which
    projects the embeddings to a certain size. If a `Seq2SeqEncoder` is not provided, we will pass the embedded
    text directly to the `Seq2VecEncoder`.

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2seq_encoder : `Seq2SeqEncoder`, optional, (default=`None`)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    feedforward : `FeedForward`, optional, (default = None).
        An optional feedforward layer to apply after the seq2vec_encoder.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        loss: PyTorchMetricLearningLoss,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward

        self._loss = loss
        initializer(self)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : TextFieldTensors
            From a `TextField`

        # Returns

        An output dictionary consisting of:

        embeddings : torch.FloatTensor
            A tensor of shape `(batch_size, self._seq2vec_encoder.get_output_dim())`, which is the
            representation for the given `tokens` output by the encoder. The encoder is composed of:
            `self._text_field_embedder`, `self._seq2seq_encoder` (optional), and `self._seq2vec_encoder`, in that
            order.
        projections : torch.FloatTensor
            A tensor of shape `(batch_size, self._feedforward.get_output_dim())`, which is the non-linear
            projection of the learned representation for the given `anchor_tokens` output by the projection head.
            This field will only be included if `self._feedforward` is not `None`.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimized.
        """
        output_dict: Dict[str, torch.Tensor] = {}

        if tokens["tokens"]["token_ids"].dim() == 3:
            anchors, positives = sample_anchor_positive_pairs(tokens)
            # TODO (John): I don't think this works for several reasons:
            #   1. tokens is a dict
            #   2. tokens is used in sample_anchor_positive_pairs
            # Find a way to get these tensors off the GPU as they are taking up memory.
            del tokens
            torch.cuda.empty_cache()
        else:
            anchors, positives = tokens, None

        # This is the textual representation learned by a trained model that will be used for downstream tasks.
        embedded_anchor_text = self._forward_internal(anchors, output_dict)

        if positives is not None:
            embedded_positive_text = self._forward_internal(positives)
            embeddings, labels = self._loss.get_embeddings_and_labels(
                embedded_anchor_text, embedded_positive_text
            )
            output_dict["loss"] = self._loss(embeddings, labels)

        return output_dict

    def _forward_internal(
        self,
        tokens: TextFieldTensors,
        output_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)
        if output_dict is not None:
            output_dict["embeddings"] = embedded_text.clone().detach()

        # Representations produced by the non-linear projection are used for training with a contrastive loss.
        # When embedding text with a trained model, we want the representation produced by the encoder network.
        # See: https://arxiv.org/abs/2002.05709
        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)
            if output_dict is not None:
                output_dict["projections"] = embedded_text.clone().detach()

        return embedded_text
