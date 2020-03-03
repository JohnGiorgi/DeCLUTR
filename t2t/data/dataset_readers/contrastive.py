import logging
from typing import Dict, Iterable, List, Optional

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ListField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer
from overrides import overrides

from t2t.data.dataset_readers.dataset_utils.span_utils import sample_spans

logger = logging.getLogger(__name__)


@DatasetReader.register("contrastive")
class ContrastiveDatasetReader(DatasetReader):
    """
    Read a txt file containing one instance per line, and create a dataset suitable for a `ContrastiveTextEncoder`
    model.

    The output of `read` is a list of `Instance` s with the field:
        tokens : `ListField[TextField]`

    # Parameters

    tokenizer : `Tokenizer`, optional (default = `{"tokens": SpacyTokenizer()}`)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text. See :class:`TokenIndexer`.
    sample_spans : ``boolean``, optional (default = True)
        If True, spans of text will be sampled from each input, tokenized and indexed.
    max_spans : ``int``, optional (default = None)
        The total number of spans to sample for each input. Defaults to sampling all possible spans. Has no
        effect if `sample_spans` is `False`.
    min_span_width : `int`, optional (default = 1)
        The minimum length of spans which should be sampled. Defaults to 1. Has no effect if `sample_spans` is
        `False`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        sample_spans: bool = True,
        min_span_width: int = 1,
        max_spans: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._sample_spans = sample_spans
        self._max_spans = max_spans
        self._min_span_width = min_span_width

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for idx, text in enumerate(data_file):
                # We use whitespace tokenization when sampling spans, so we also use it here to check that a
                # valid min_span_width was given.
                num_tokens = len(text.split())
                if num_tokens < self._min_span_width:
                    raise ConfigurationError(
                        (
                            f"min_span_width is {self._min_span_width} but instance on line {idx + 1} has len"
                            f" {num_tokens}"
                        )
                    )
                yield self.text_to_instance(text)

    @overrides
    def text_to_instance(self, text: str) -> Instance:  # type: ignore
        """
        # Parameters

        text : `str`, required.
            The text to process.

        # Returns

        An `Instance` containing the following fields:
            tokens : ``Union[TextField, ListField[TextField]]`
                If `self._sample_spans`, returns a `ListField` containing `self._max_spans` number of random,
                tokenized spans from `text`.
                Else, returns a `TextField` containing tokenized `text`.
        """

        fields: Dict[str, Field] = {}
        if self._sample_spans:
            spans: List[Field] = []
            for span in sample_spans(text, self._max_spans, min_span_width=self._min_span_width):
                tokens = self._tokenizer.tokenize(span)
                spans.append(TextField(tokens, self._token_indexers))
            fields["tokens"] = ListField(spans)
        else:
            tokens = self._tokenizer.tokenize(text)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        return Instance(fields)
