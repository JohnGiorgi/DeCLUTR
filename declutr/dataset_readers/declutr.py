import logging
import random
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional

import torch
from overrides import overrides

from allennlp.common import util
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ListField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer
from declutr.dataset_readers.dataset_utils import contrastive_utils

logger = logging.getLogger(__name__)


@DatasetReader.register("declutr")
class DeCLUTRDatasetReader(DatasetReader):
    """
    Read a text file containing one instance per line, and create a dataset suitable for a
    `DeCLUTR` model.

    The output of `read` is a list of `Instance` s with the field:
        tokens : `ListField[TextField]`
    if `num_spans > 0`, else:
        tokens : `TextField`

    Registered as a `DatasetReader` with name "declutr".

    # Parameters

   token_indexers : `Dict[str, TokenIndexer]`, optional
        optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : `Tokenizer`, optional (default = `{"tokens": SpacyTokenizer()}`)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    num_anchors : `int`, optional
        The number of spans to sample from each instance to serve as anchors.
    num_positives : `int`, optional
        The number of spans to sample from each instance to serve as positive examples (per anchor).
        Has no effect if `num_anchors` is not provided.
    max_span_len : `int`, optional
        The maximum length of spans (after tokenization) which should be sampled. Has no effect if
        `num_spans` is not provided.
    min_span_len : `int`, optional
        The minimum length of spans (after tokenization) which should be sampled. Has no effect if
        `num_spans` is not provided.
    sampling_strategy : `str`, optional (default = None)
        One of "subsuming" or "adjacent". If "subsuming," positive spans are always subsumed by the
        anchor. If "adjacent", positive spans are always adjacent to the anchor. If not provided,
        positives may be subsumed, adjacent to, or overlapping with the anchor. Has no effect if
        `num_spans` is not provided.
    """

    def __init__(
        self,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        tokenizer: Optional[Tokenizer] = None,
        num_anchors: Optional[int] = None,
        num_positives: Optional[int] = None,
        max_span_len: Optional[int] = None,
        min_span_len: Optional[int] = None,
        sampling_strategy: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._num_anchors = num_anchors
        if self._num_anchors is not None:
            if num_positives is None:
                raise ValueError("num_positives must be provided if num_anchors is not None.")
            if max_span_len is None:
                raise ValueError("max_span_len must be provided if num_anchors is not None.")
            if min_span_len is None:
                raise ValueError("min_span_len must be provided if num_anchors is not None.")
        self._num_positives = num_positives
        self._max_span_len = max_span_len
        self._min_span_len = min_span_len
        self.sample_spans = bool(self._num_anchors)
        self._sampling_strategy = (
            sampling_strategy.lower() if sampling_strategy is not None else sampling_strategy
        )
        if (
            self.sample_spans
            and self._sampling_strategy is not None
            and self._sampling_strategy not in ["subsuming", "adjacent"]
        ):
            raise ValueError(
                (
                    'sampling_strategy must be one of ["subsuming", "adjacent"].'
                    f" Got {self._sampling_strategy}."
                )
            )

        # In the v1.0 AllenNLP pre-release, theres a small catch that dataset readers used in the
        # distributed setting need to shard instances to separate processes internally such that one
        # epoch strictly corresponds to one pass over the data. This may get fixed in the v1.0
        # release. See here: https://github.com/allenai/allennlp/releases.
        if util.is_distributed():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        # HACK (John): I need to temporarily disable user warnings because this objects __len__
        # function returns 1, which confuses PyTorch.
        import warnings

        warnings.filterwarnings("ignore")

    @property
    def sample_spans(self) -> None:
        return self._sample_spans

    @sample_spans.setter
    def sample_spans(self, sample_spans: bool) -> None:
        self._sample_spans = sample_spans

    @contextmanager
    def no_sample(self) -> None:
        """Context manager that disables sampling of spans. Useful at test time when we want to
        embed unseen text.
        """
        prev = self.sample_spans
        self.sample_spans = False
        yield self
        self.sample_spans = prev

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # If we are sampling spans (i.e. we are training) we need to shuffle the data so that
            # we don't yield instances in the same order every epoch. Our current solution is to
            # read the entire file into memory. This is a little expensive (roughly 1G per 1 million
            # docs), so a better solution might be required down the line.
            if self.sample_spans:
                data_file = list(enumerate(data_file))
                random.shuffle(data_file)
                data_file = iter(data_file)
            else:
                data_file = enumerate(data_file)

            for i, text in data_file:
                if i % self._world_size == self._rank:
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
                If `self.sample_spans`, returns a `ListField` containing two random, tokenized
                spans from `text`. Else, returns a `TextField` containing tokenized `text`.
        """
        fields: Dict[str, Field] = {}
        if self.sample_spans:
            # Choose the anchor/positives at random (spans are not contigous)
            anchor_text, positive_text = contrastive_utils.sample_anchor_positives(
                text=text,
                num_anchors=self._num_anchors,
                num_positives=self._num_positives,
                max_span_len=self._max_span_len,
                min_span_len=self._min_span_len,
                sampling_strategy=self._sampling_strategy,
            )
            anchors: List[Field] = []
            for text in anchor_text:
                tokens = self._tokenizer.tokenize(text)
                anchors.append(TextField(tokens, self._token_indexers))
            fields["anchors"] = ListField(anchors)
            positives: List[Field] = []
            for text in positive_text:
                tokens = self._tokenizer.tokenize(text)
                positives.append(TextField(tokens, self._token_indexers))
            fields["positives"] = ListField(positives)
        else:
            tokens = self._tokenizer.tokenize(text)
            fields["anchors"] = TextField(tokens, self._token_indexers)
        return Instance(fields)
