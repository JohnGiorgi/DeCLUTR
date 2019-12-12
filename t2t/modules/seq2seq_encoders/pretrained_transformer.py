import torch
from overrides import overrides
from torch import nn

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from transformers.modeling_auto import AutoModel


@Seq2SeqEncoder.register("pretrained_transformer")
class PretrainedTransformerEncoder(Seq2SeqEncoder):
    """
    Implements an encoder using a pre-trained transformer model (e.g. BERT) from the 
    ``Transformers`` library. Because these models both embed and encode their inputs, when this
    class is used as the encoder of a encoder-decoder model it should be paired with the 
    ``PassThrough`` token embedder.

    Parameters
    ----------
    vocab : ``Vocabulary``
    model_name : ``str``
        The pre-trained ``Transformers`` model to be wrapped. The model is instantiated by calling:
        ``AutoModel.from_pretrained(model_name)``.
    """
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self._transformer_model = AutoModel.from_pretrained(model_name)
        self._output_dim = self.transformer_model.config.hidden_size

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        """
        Returns ``True`` if this encoder is bidirectional. If so, we assume the forward direction
        of the encoder is the first half of the final dimension, and the backward direction is the
        second half.
        """
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # type: ignore
        sequence_output, _ = self._transformer_model(inputs, attention_mask=mask)

        # TODO (John): We use a mean of the transformers last hidden states as our pooler
        # In the future, we will make this modular and experiment with learnable poolers.
        pooled_output = (torch.sum(sequence_output * mask.unsqueeze(-1), dim=1) / 
                         torch.clamp(torch.sum(mask, dim=1, keepdim=True), min=1e-9))
        # HACK (John): Decoder expects: (batch_size, max_input_sequence_length, encoder_input_dim)
        pooled_output = pooled_output.unsqueeze(1).expand_as(sequence_output)

        return pooled_output
