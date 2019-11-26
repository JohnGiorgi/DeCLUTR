import torch
from overrides import overrides

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from transformers.modeling_auto import AutoModel


@Seq2SeqEncoder.register("pretrained_transformer")
class PretrainedTransformerEncoder(Seq2SeqEncoder):
    """
    Implements an encoder from a pretrained transformer model (e.g. BERT).

    Parameters
    ----------
    vocab : ``Vocabulary``
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.transformer_model = AutoModel.from_pretrained(model_name)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size

    @overrides
    def get_output_dim(self):
        return self.output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        """
        Returns ``True`` if this encoder is bidirectional.  If so, we assume the forward direction
        of the encoder is the first half of the final dimension, and the backward direction is the
        second half.
        """
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.transformer_model(inputs, attention_mask=mask)[0]
