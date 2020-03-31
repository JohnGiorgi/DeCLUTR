from random import randint
from typing import Callable, Iterable, Optional

import numpy as np


def sample_spans(
    text: str, num_spans: int, min_span_len: int, tokenizer: Optional[Callable] = None, **kwargs,
) -> Iterable[str]:
    """Returns a generator that yields random spans from `text`.

    # Parameters

    text : `str`, required
        The string to extract spans from.
    num_spans : `int`, required
        The total number of spans to return.
    min_span_len : `int`, required
        The minimum length of spans, after whitespace tokenization, to sample.
    tokenizer : `Callable`, optional
        Optional tokenizer to use before sampling spans. If `None`, `text.split()` is used.
    """

    # Whitespace tokenization makes it much more straightforward to do whole word masking but a user can
    # also provide their own tokenization scheme if they want.
    tokens = tokenizer(text) if tokenizer is not None else text.split()
    num_tokens = len(tokens)
    if min_span_len > num_tokens:
        tok_method = "tokenizer(text)" if tokenizer else "text.split()"
        raise ValueError(
            f"min_span_len ({min_span_len}) must be less than or equal to len({tok_method}) ({num_tokens})."
        )

    for _ in range(num_spans):
        # 1. Sample span length from a beta distribution, which is skewed towards longer spans.
        span_length = int(np.random.beta(4, 2) * (num_tokens - min_span_len) + min_span_len)
        # 2. Sample the start index of the span uniformly
        start = randint(0, num_tokens - span_length)
        end = start + span_length

        yield " ".join(tokens[start:end])
