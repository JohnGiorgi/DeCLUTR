import torch
from overrides import overrides

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from transformers.modeling_auto import AutoModel


@Seq2SeqEncoder.register("pretrained_transformer")
class PretrainedTransformerEncoder(Seq2SeqEncoder):
    """
    Implements an encoder using a pre-trained transformer model (e.g. BERT) from the
    ``Transformers`` library. Because these models both embed and encode their inputs, when this
    class is used as the encoder of an encoder-decoder model it should be paired with the
    ``PassThrough`` token embedder.

    Parameters
    ----------
    model_name : ``str``
        The pre-trained ``Transformers`` model to be wrapped. The model is instantiated by calling:
        ``AutoModel.from_pretrained(model_name)``.
    pooler : ``Seq2VecEncoder``
        Pools the outputs from the pre-trained transformer across the time dimension.
    """
    def __init__(self, model_name: str, pooler: Seq2VecEncoder) -> None:
        super().__init__()
        self._transformer_model = AutoModel.from_pretrained(model_name)
        self._pooler = pooler
        self._output_dim = pooler.get_output_dim()

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
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None
    ) -> torch.Tensor:  # type: ignore

        sequence_output = self._transformer_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[0]  # The first output is always the hidden state of the final layer

        pooled_output = self._pooler(sequence_output, attention_mask)
        # HACK (John): Decoder expects: (batch_size, max_input_sequence_length, encoder_input_dim)
        pooled_output = pooled_output.unsqueeze(1).expand(input_ids.size(0), input_ids.size(1), -1)

        return pooled_output
