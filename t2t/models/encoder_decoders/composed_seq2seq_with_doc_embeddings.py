from typing import Dict

import torch
from overrides import overrides

from allennlp.models.encoder_decoders.composed_seq2seq import ComposedSeq2Seq
from allennlp.models.model import Model


@Model.register("composed_seq2seq_with_doc_embeddings")
class ComposedSeq2SeqWithDocEmbeddings(ComposedSeq2Seq):
    """
    A thin wrapper around ``ComposedSeq2Seq`` which adds the document embeddings learned by the model to its
    ``output_dict`` in the ``forward`` hook. See the ``ComposedSeq2Seq`` model in the AllenNLP documentation for
    full details.
    """

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: Dict[str, torch.LongTensor],
        target_tokens: Dict[str, torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Make foward pass on the encoder and decoder for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be passed through a
           `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the target tokens are
           also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output tensors from the decoder.
        """
        state = self._encode(source_tokens)
        output_dict = self._decoder(state, target_tokens)

        # During eval, copy the encoders output to CPU memory. This becomes our document embedding.
        # HACK (John): Document embeddings are expanded along the second dimension, drop it
        if not self.training:
            output_dict['doc_embeddings'] = state['encoder_outputs'][:, 0, :].cpu().squeeze(1)

        return output_dict
