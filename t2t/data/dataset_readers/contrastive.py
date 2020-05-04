import logging
import random
from contextlib import contextmanager
from random import randint
from typing import Callable, Dict, Iterable, List, Optional

import torch
from overrides import overrides

from allennlp.common import util
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ListField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("contrastive")
class ContrastiveDatasetReader(DatasetReader):
    """
    Read a text file containing one instance per line, and create a dataset suitable for a
    `ContrastiveTextEncoder` model.

    The output of `read` is a list of `Instance` s with the field:
        tokens : `ListField[TextField]`
    if `sample_spans`, else:
        tokens : `TextField`


    Registered as a `DatasetReader` with name "contrastive".

    # Parameters

   token_indexers : `Dict[str, TokenIndexer]`, optional
        optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : `Tokenizer`, optional (default = `{"tokens": SpacyTokenizer()}`)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    sample_spans : `bool`, optional (default = True)
        If True, two spans will be sampled from each input, tokenized and indexed.
    max_span_len : `int`, optional
        The maximum length of spans which should be sampled. Has no effect if
        `sample_spans is False`.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        num_chunks: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._num_chunks = num_chunks

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
    def num_chunks(self):
        return self._num_chunks

    @num_chunks.setter
    def num_chunks(self, num_chunks):
        self._num_chunks = num_chunks

    @contextmanager
    def no_chunk(self):
        """Context manager that disables chunking."""
        prev = self.num_chunks
        self.num_chunks = None
        yield self
        self.num_chunks = prev

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # If we are sampling spans (i.e. we are training) we need to shuffle the data so that
            # we don't yield instances in the same order every epoch. Our current solution is to
            # read the entire file into memory. This is a little expensive (roughly 1G per 1 million
            # docs), so a better solution might be required down the line.
            if self.num_chunks:
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
                If `self.num_chunks`, returns a `ListField` containing two random, tokenized
                spans from `text`. Else, returns a `TextField` containing tokenized `text`.
        """
        fields: Dict[str, Field] = {}
        if self.num_chunks:
            # Anchors are randomly sampled "contexts" of length self._tokenizer.tokenizer.max_len
            context = self._get_context(text, self._tokenizer.tokenizer.max_len)
            tokens = self._tokenizer.tokenize(context)
            fields["anchors"] = TextField(tokens, self._token_indexers)
            # Positives are contiguous "chunks" of this context
            chunks: List[Field] = []
            for chunk in self._chunk_context(context, self.num_chunks):
                tokens = self._tokenizer.tokenize(chunk)
                chunks.append(TextField(tokens, self._token_indexers))
            fields["positives"] = ListField(chunks)
        else:
            tokens = self._tokenizer.tokenize(text)
            fields["anchors"] = TextField(tokens, self._token_indexers)
        return Instance(fields)

    def _get_context(
        self, text: str, length: int, tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> str:
        tokens = tokenizer(text) if tokenizer is not None else text.split()
        num_tokens = len(tokens)
        start = randint(0, num_tokens - length)
        end = start + length
        return " ".join(tokens[start:end])

    def _chunk_context(
        self, text: str, num_chunks: int, tokenizer: Optional[Callable[[str], List[str]]] = None
    ) -> List[str]:
        tokens = tokenizer(text) if tokenizer is not None else text.split()
        num_tokens = len(tokens)
        chunk_size = num_tokens // num_chunks
        return [" ".join(tokens[i : i + chunk_size]) for i in range(0, num_tokens, chunk_size)]
