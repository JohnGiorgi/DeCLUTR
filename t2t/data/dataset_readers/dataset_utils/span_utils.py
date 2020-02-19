from typing import Iterable, Optional

import numpy as np

from allennlp.data.dataset_readers.dataset_utils.span_utils import \
    enumerate_spans


def sample_spans(text: str, max_spans: Optional[int] = None, **kwargs) -> Iterable[str]:
    """Returns a generator that yields random spans from `text`.

    # Parameters

    text : ``str``, required.
        The string to extract spans from.
    max_spans : ``int``, optional (default = None)
        The total number of spans to return. Defaults to returning all possible spans.
    **kwargs : ``Dict``
        Additional keyword arguments to be passed to `enumerate_spans`.
    """
    # No need for fancier tokenization here. Whitespace tokenization will suffice to generate spans.
    tokens = text.split()
    span_indices = enumerate_spans(tokens, **kwargs)
    np.random.shuffle(span_indices)
    if max_spans is not None:
        span_indices = span_indices[:max_spans]

    for start, end in span_indices:
        # Add 1 to end index because spans from enumerate_spans are inclusive.
        yield " ".join(tokens[start: end + 1])
