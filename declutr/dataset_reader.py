import logging
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ListField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer
from overrides import overrides

from declutr.common.contrastive_utils import sample_anchor_positive_pairs
from declutr.common.data_utils import sanitize

logger = logging.getLogger(__name__)


@DatasetReader.register("declutr")
class DeCLUTRDatasetReader(DatasetReader):
    """
    Read a text file containing one instance per line, and create a dataset suitable for a
    `DeCLUTR` model.

    The output of `read` is a list of `Instance` s with the field:
        tokens : `ListField[TextField]`
    if `num_anchors > 0`, else:
        tokens : `TextField`

    Registered as a `DatasetReader` with name "declutr".

    # Parameters

    tokenizer : `Tokenizer`, optional (default = `{"tokens": SpacyTokenizer()}`)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        We use this to define the input representation for the text. See :class:`TokenIndexer`.
    num_anchors : `int`, optional
        The number of spans to sample from each instance to serve as anchors.
    num_positives : `int`, optional
        The number of spans to sample from each instance to serve as positive examples (per anchor).
        Has no effect if `num_anchors` is not provided.
    max_span_len : `int`, optional
        The maximum length of spans (after tokenization) which should be sampled. Has no effect if
        `num_anchors` is not provided.
    min_span_len : `int`, optional
        The minimum length of spans (after tokenization) which should be sampled. Has no effect if
        `num_anchors` is not provided.
    sampling_strategy : `str`, optional (default = None)
        One of "subsuming" or "adjacent". If "subsuming," positive spans are always subsumed by the
        anchor. If "adjacent", positive spans are always adjacent to the anchor. If not provided,
        positives may be subsumed, adjacent to, or overlapping with the anchor. Has no effect if
        `num_anchors` is not provided.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        num_anchors: int = None,
        num_positives: int = None,
        max_span_len: int = None,
        min_span_len: int = None,
        sampling_strategy: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        # If the user provided us with a number of anchors to sample, we automatically
        # check that the other expected values are provided and valid.
        if num_anchors is not None:
            self._num_anchors = num_anchors
            self.sample_spans = True
            if num_positives is None:
                raise ValueError("num_positives must be provided if num_anchors is not None.")
            if max_span_len is None:
                raise ValueError("max_span_len must be provided if num_anchors is not None.")
            if min_span_len is None:
                raise ValueError("min_span_len must be provided if num_anchors is not None.")
            self._num_positives = num_positives
            self._max_span_len = max_span_len
            self._min_span_len = min_span_len
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
        else:
            self.sample_spans = False

    @property
    def sample_spans(self) -> bool:
        return self._sample_spans

    @sample_spans.setter
    def sample_spans(self, sample_spans: bool) -> None:
        self._sample_spans = sample_spans

    @contextmanager
    def no_sample(self) -> Iterator[None]:
        """A context manager that temporarily disables sampling of spans. Useful at test time when
        we want to embed unseen text.
        """
        prev = self.sample_spans
        self.sample_spans = False
        yield
        self.sample_spans = prev

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as f:
            for line in f:
                yield self.text_to_instance(line)

    @overrides
    def text_to_instance(self, text: str) -> Instance:  # type: ignore
        """
        # Parameters

        text : `str`, required.
            The text to process.

        # Returns

        An `Instance` containing the following fields:
            - anchors (`Union[TextField, ListField[TextField]]`) :
                If `self.sample_spans`, this will be a `ListField[TextField]` object, containing
                each anchor span sampled from `text`. Otherwise, this will be a `TextField` object
                containing the tokenized `text`.
            - positives (`ListField[TextField]`) :
                If `self.sample_spans`, this will be a `ListField[TextField]` object, containing
                each positive span sampled from `text`. Otherwise this field will not be included
                in the returned `Instance`.
        """
        # Some very minimal preprocessing to remove whitespace, newlines and tabs.
        # We peform it here as it will cover both training and predicting with the model.
        # We DON'T lowercase by default, but rather allow `self._tokenizer` to decide.
        text = sanitize(text, lowercase=False)

        fields: Dict[str, Field] = {}
        if self.sample_spans:
            # Choose the anchor/positives at random.
            anchor_text, positive_text = sample_anchor_positive_pairs(
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
                anchors.append(TextField(tokens))
            fields["anchors"] = ListField(anchors)
            positives: List[Field] = []
            for text in positive_text:
                tokens = self._tokenizer.tokenize(text)
                positives.append(TextField(tokens))
            fields["positives"] = ListField(positives)
        else:
            tokens = self._tokenizer.tokenize(text)
            fields["anchors"] = TextField(tokens)
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        # If this is a ListField, we need to set the token_indexers of each TextField independently.
        if isinstance(instance["anchors"], ListField):
            for text_field in instance["anchors"].field_list:
                text_field.token_indexers = self._token_indexers
            for text_field in instance["positives"].field_list:
                text_field.token_indexers = self._token_indexers
        # Otherwise, spans were not sampled, and instances contain a single TextField at "anchors".
        else:
            instance["anchors"].token_indexers = self._token_indexers
