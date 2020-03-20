import logging
import random
from typing import Dict, Iterable, List, Optional

import torch.distributed as dist
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ListField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer
from t2t.data.dataset_readers.dataset_utils.contrastive_utils import sample_spans

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
    sample_spans : `bool`, optional (default = True)
        If True, two spans will be sampled from each input, tokenized and indexed.
    min_span_width : `int`, optional (default = 1)
        The minimum length of spans which should be sampled. Defaults to 1. Has no effect if
        `sample_spans is False`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        sample_spans: Optional[bool] = True,
        min_span_width: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._sample_spans = sample_spans
        self._min_span_width = min_span_width

        # HACK (John): I need to temporarily disable user warnings because this objects __len__ function returns
        # 1, which confuses PyTorch.
        import warnings

        warnings.filterwarnings("ignore")

    @property
    def sample_spans(self):
        return self._sample_spans

    @sample_spans.setter
    def sample_spans(self, sample_spans):
        self._sample_spans = sample_spans

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            # In the v1.0 AllenNLP pre-release, theres a small catch that dataset readers used in the distributed
            # setting need to shard instances to separate processes internally such that one epoch strictly
            # corresponds to one pass over the data. This may get fixed in the v1.0 release.
            # See here: https://github.com/allenai/allennlp/releases.
            distributed = dist.is_initialized()
            if distributed:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            # If we are sampling spans (i.e. we are training) we need to shuffle the data so that we don't yield
            # instances in the same order every epoch. Our current solution is to read the entire file into memory.
            # This is a little expensive (roughly 1G per 1 million paragraphs), so a better solution might be
            # required down the line.
            if self._sample_spans:
                data_file = list(enumerate(data_file))
                random.shuffle(data_file)
                data_file = iter(data_file)
            else:
                data_file = enumerate(data_file)

            for idx, text in data_file:
                if distributed and idx % world_size != rank:
                    continue
                # We use whitespace tokenization when sampling spans, so we also use it here to check that a
                # valid min_span_width was given.
                num_tokens = len(text.split())
                if self._sample_spans and num_tokens < self._min_span_width:
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
            tokens : `Union[TextField, ListField[TextField]]`
                If `self._sample_spans`, returns a `ListField` containing `self._sample_spans` number of random,
                tokenized spans from `text`.
                Else, returns a `TextField` containing tokenized `text`.
        """
        fields: Dict[str, Field] = {}
        if self._sample_spans:
            spans: List[Field] = []
            for span in sample_spans(text, 2, self._min_span_width):
                tokens = self._tokenizer.tokenize(span)
                spans.append(TextField(tokens, self._token_indexers))
            fields["tokens"] = ListField(spans)
        else:
            tokens = self._tokenizer.tokenize(text)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        return Instance(fields)
