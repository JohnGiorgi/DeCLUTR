from random import randint
from typing import Callable, List, Optional, Tuple

import numpy as np


def sample_anchor_positives(
    text: str,
    max_span_len: int,
    min_span_len: int,
    num_spans: Optional[int] = 1,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Tuple[str, List[str]]:
    """Returns a tuple of anchor (`str`) and `num_spans` positive (`List[str]`) spans sampled from
    `text`.

    # Parameters

    text : `str`, required
        The string to extract anchor and positive spans from.
    max_span_len : `int`, required
        The maximum length of spans, after tokenization, to sample.
    min_span_len : `int`, optional
        The minimum length of spans, after tokenization, to sample.
    num_spans : `int`, optional (default = 1)
        The number of spans to sample from `text` to serve as positive examples.
    tokenizer : `Callable`, optional
        Optional tokenizer to use before sampling spans. If `None`, `text.split()` is used.
    """
    # Whitespace tokenization is much more straightforward (don't need to worry about chopping up
    # subword tokens) but a user can also provide their own tokenization scheme if they want.
    tokens = tokenizer(text) if tokenizer is not None else text.split()
    num_tokens = len(tokens)
    tok_method = "tokenizer(text)" if tokenizer else "text.split()"

    if num_tokens < 1:
        raise ValueError(
            (f"len({tok_method}) should be at least 1 (ideally much longer), got {num_tokens}.")
        )
    if min_span_len > max_span_len:
        raise ValueError(
            f"min_span_len must be less than max_span_len ({max_span_len}), got {min_span_len}."
        )
    if max_span_len > num_tokens:
        raise ValueError(
            (
                f"max_span_len must be less than or equal to"
                f" len({tok_method}) ({num_tokens}), got {max_span_len}."
            )
        )

    # Sample the anchor length from a beta distribution skewed towards longer spans, the intuition
    # being that longer spans have the best chance of being representative of the document they are
    # sampled from.
    anchor_length = int(np.random.beta(4, 2) * (max_span_len - min_span_len) + min_span_len)
    anchor_start = randint(0, num_tokens - anchor_length)
    anchor_end = anchor_start + anchor_length
    anchor = " ".join(tokens[anchor_start:anchor_end])

    # Sample positives from around the anchor. The intuition being that text that appears close
    # together is the same document is likely to be semantically similar. These may be adjacent or
    # overlap with each other and the anchor. Their length is sampled from a beta distribution
    # skewed towards shorter spans.
    positives = []
    for _ in range(num_spans):
        positive_length = int(np.random.beta(2, 4) * (max_span_len - min_span_len) + min_span_len)
        # Be careful not to run off the edges of the document, as this error will pass silently.
        positive_start = randint(
            max(0, anchor_start - positive_length), min(anchor_end, num_tokens - positive_length),
        )
        positive_end = positive_start + positive_length
        positives.append(" ".join(tokens[positive_start:positive_end]))

    return anchor, positives
