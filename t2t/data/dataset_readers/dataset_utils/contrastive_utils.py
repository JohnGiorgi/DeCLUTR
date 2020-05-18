from enum import Enum
from random import choice, randint
from typing import Callable, List, Optional, Tuple

import numpy as np


class Strategy(str, Enum):
    subsuming = "subsuming"
    adjacent = "adjacent"


def sample_anchor_positives(
    text: str,
    max_span_len: int,
    min_span_len: int,
    num_spans: Optional[int] = 1,
    strategy: Optional[Strategy] = None,
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
    strategy : `Strategy`, optional (default = None)
        One of "subsuming" or "adjacent". If "subsuming", positive spans are always subsumed by
        the anchor. If "adjacent", positive spans are always adjacent to the anchor. If not
        provided, positives may be subsumed, adjacent to, or overlapping with the anchor.
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
    # together is the same document is likely to be semantically similar. By default, these may be
    # adjacent or overlap with each other and the anchor, but this can be selected manually with
    # the `strategy` parameter. Their length is sampled from a beta distribution skewed towards
    # shorter spans.
    positives = []
    for _ in range(num_spans):
        positive_length = int(np.random.beta(2, 4) * (max_span_len - min_span_len) + min_span_len)
        if strategy and strategy == strategy.subsuming:
            positive_start = randint(anchor_start, anchor_end - positive_length)
        elif strategy and strategy == strategy.adjacent:
            # There are two types of adjacent positives, those that boarder the beginning of the
            # anchor and those that boarder the end. These may not be valid in cases where the
            # positive length is longer than the available space in the document.
            valid_starts = []
            if anchor_start - positive_length > 0:
                valid_starts.append(anchor_start - positive_length)
            elif anchor_end + positive_length <= num_tokens:
                valid_starts.append(anchor_end + positive_length)
            else:
                raise ValueError(
                    (
                        f"Cannot select adjacent only spans for a span length of {positive_length}"
                        f" and a number of tokens len({tok_method}) ({num_tokens})."
                    )
                )
            positive_start = choice(valid_starts)
        else:
            # Be careful not to run off the edges of the document, as this error will pass silently.
            positive_start = randint(
                max(0, anchor_start - positive_length),
                min(anchor_end, num_tokens - positive_length),
            )

    positive_end = positive_start + positive_length
    positives.append(" ".join(tokens[positive_start:positive_end]))

    return anchor, positives
