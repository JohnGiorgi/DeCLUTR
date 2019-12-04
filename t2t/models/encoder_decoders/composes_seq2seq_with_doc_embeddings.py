from typing import Dict

import torch
from overrides import overrides

from allennlp.models.encoder_decoders.composed_seq2seq import ComposedSeq2Seq
from allennlp.models.model import Model


@Model.register("composed_seq2seq_with_doc_embeddings")
class ComposedSeq2SeqWithDocEmbeddings(ComposedSeq2Seq):
    """
    A thin wrapper around ``ComposedSeq2Seq`` which adds the document embeddings learned by the
    model to its ``output_dict`` in the ``forward`` hook. See the ``ComposedSeq2Seq model in the
    AllenNLP documentation for full details.
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
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output tensors from the decoder.
        """
        state = self._encode(source_tokens)
        output_dict = self._decoder(state, target_tokens)

        # HACK (John): Our current solution to extracting document embeddings
        with torch.no_grad():
            encoder_outputs, source_mask = state['encoder_outputs'], state['source_mask']

            doc_mask = source_mask.unsqueeze(-1).expand_as(encoder_outputs)
            doc_embeddings = (torch.sum(encoder_outputs * doc_mask, dim=1) /
                              # clamp prevents division by 0
                              torch.clamp(torch.sum(doc_mask, dim=1), min=1e-9))

        output_dict['doc_embeddings'] = doc_embeddings
