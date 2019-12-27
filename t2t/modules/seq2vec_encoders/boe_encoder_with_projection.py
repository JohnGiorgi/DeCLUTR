from typing import Optional

import torch
from overrides import overrides
from torch.nn import Linear

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.util import get_lengths_from_binary_sequence_mask


@Seq2VecEncoder.register("boe_with_projection")
@Seq2VecEncoder.register("bag_of_embeddings_with_projection")
class BagOfEmbeddingsEncoderWithProjection(BagOfEmbeddingsEncoder):
    """
    A ``BagOfEmbeddingsEncoderWithProjection`` is a simple :class:`Seq2VecEncoder` which simply sums the
    embeddings of a sequence across the time dimension. The input to this module is of shape
    ``(batch_size, num_tokens, embedding_dim)``, and the output is of shape ``(batch_size, embedding_dim)``. If
    ``output_dim`` is not None, then the summed embeddings are linearly projected to this size, and the output is
    of shape ``(batch_size, output_dim)``.

    Parameters
    ----------
    embedding_dim: ``int``
        This is the input dimension to the encoder.
    averaged: ``bool``, optional (default=``False``)
        If ``True``, this module will average the embeddings across time, rather than simply summing (i.e. we will
        divide the summed embeddings by the length of the sentence).
    output_dim : ``Optional[int]``, optional (default=``None``)
        After pooling, we'll project the collected features into a vector of this size. If this value is ``None``,
        we will just return the result of the pooling, giving an output of shape
        ``(tokens.size(0), tokens.size(-1))``.
    """

    def __init__(self, embedding_dim: int, averaged: bool = False, output_dim: Optional[int] = None) -> None:
        super().__init__(embedding_dim, averaged)

        if output_dim:
            self._projection_layer = Linear(embedding_dim, output_dim)
            self._output_dim = output_dim
        else:
            self._projection_layer = None
            self._output_dim = embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Our input has shape `(batch_size, num_tokens, embedding_dim)`, so we sum out the
        # `num_tokens` dimension.
        summed = tokens.sum(1)

        if self._averaged:
            if mask is not None:
                lengths = get_lengths_from_binary_sequence_mask(mask)
                length_mask = lengths > 0

                # Set any length 0 to 1, to avoid dividing by zero.
                lengths = torch.max(lengths, lengths.new_ones(1))
            else:
                lengths = tokens.new_full((1,), fill_value=tokens.size(1))
                length_mask = None

            summed = summed / lengths.unsqueeze(-1).float()

            if length_mask is not None:
                summed = summed * (length_mask > 0).float().unsqueeze(-1)

        if self._projection_layer is not None:
            summed = self._projection_layer(summed)

        return summed
