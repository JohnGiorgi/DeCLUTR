import csv
import logging
from typing import Dict, Optional

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("contrastive")
class ContrastiveDatasetReader(DatasetReader):
    """
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        delimiter: str = "\t",
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._delimiter = delimiter
        self._max_tokens = max_tokens
        self._anchor_max_exceeded = 0
        self._positive_max_exceeded = 0

    @overrides
    def _read(self, file_path: str):
        # Reset exceeded counts
        self._anchor_max_exceeded = 0
        self._positive_max_exceeded = 0
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for row in csv.reader(data_file, delimiter=self._delimiter):
                # A positive sequence will only be provided when training
                if len(row) == 2:
                    anchor_sequence, positive_sequence = row
                else:
                    anchor_sequence, positive_sequence = row, None
                yield self.text_to_instance(anchor_sequence, positive_sequence)
        if self._max_tokens and self._anchor_max_exceeded:
            logger.info(
                "In %d instances, the anchor token length exceeded the max limit (%d) and were truncated.",
                self._anchor_max_exceeded,
                self._max_tokens,
            )
        if self._max_tokens and self._positive_max_exceeded:
            logger.info(
                "In %d instances, the positive token length exceeded the max limit (%d) and were truncated.",
                self._positive_max_exceeded,
                self.max_tokens,
            )

    @overrides
    def text_to_instance(
        self, anchor_string: str, positive_string: str = None
    ) -> Instance:  # type: ignore

        tokenized_anchor = self._tokenizer.tokenize(anchor_string)
        if self._max_tokens and len(tokenized_anchor) > self._max_tokens:
            self._anchor_max_exceeded += 1
            tokenized_anchor = tokenized_anchor[: self._max_tokens]
        anchor_field = TextField(tokenized_anchor, self._token_indexers)
        if positive_string is not None:
            tokenized_positive = self._tokenizer.tokenize(positive_string)
            if self._max_tokens and len(tokenized_positive) > self._max_tokens:
                self._positive_max_exceeded += 1
                tokenized_positive = tokenized_positive[: self._max_tokens]
            positive_field = TextField(tokenized_positive, self._token_indexers)
            return Instance({"anchor_tokens": anchor_field, "positive_tokens": positive_field})
        else:
            return Instance({"anchor_tokens": anchor_field})
