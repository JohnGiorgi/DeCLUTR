import logging
from typing import Dict
from typing import Optional

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers import Seq2SeqDatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("seq2sameseq")
class Seq2SameSeqDatasetReader(Seq2SeqDatasetReader):
    """
    A thin wrapper around ``Seq2SeqDatasetReader`` that is expected to be used with an
    encoder-decoder that is trained to reconstruct the source sequence. Each line in the input file
    should contain a sequence string. Please see ``Seq2SeqDatasetReader`` for usage instructions and
    parameters.
    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = True,
        source_add_end_token: bool = True,
        delimiter: str = None,
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(
            source_tokenizer, target_tokenizer, source_token_indexers, target_token_indexers,
            source_add_start_token, source_add_end_token, delimiter, source_max_tokens,
            target_max_tokens, lazy
        )

    @overrides
    def _read(self, file_path: str):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(data_file):
                source_sequence, target_sequence = row, row
                yield self.text_to_instance(source_sequence, target_sequence)
        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )
