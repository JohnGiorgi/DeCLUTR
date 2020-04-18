import random
from random import randint
from typing import Callable, Optional, List, Tuple

import numpy as np


def sample_spans(
    text: str,
    max_span_len: int,
    min_span_len: Optional[int] = None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Tuple[str, str]:
    """Returns two randomly sampled spans from `text`. The sampling procedure is as follows:

    First, a "context window" of length `max_span_len * 2` is randomly chosen from `text`.
    Then, with equal probability, we sample either:

        1) Two adjacent spans from the context window of length `max_span_len`
        2) One "global" span from the context window of length `max_span_len` and one "local"
           span from the global span of length `min_span_len`.

    # Parameters

    text : `str`, required
        The string to extract spans from.
    max_span_len : `int`, required
        The maximum length of spans, after whitespace tokenization, to sample. This number decides
        the length of the adjacent and global span.
    min_span_len : `int`, optional
        The minimum length of spans, after whitespace tokenization, to sample. This number decides
        the length of the local span. Defaults to `int(0.25 * max_span_len)`.
    tokenizer : `Callable`, optional
        Optional tokenizer to use before sampling spans. If `None`, `text.split()` is used.
    """
    # If not provided, take the min_span_len to be 1/4 of the max_span_len
    min_span_len = int(0.25 * max_span_len) if min_span_len is None else min_span_len
    # The context window is taken to be 2X the max_span_len. The basic idea is that
    # text that is closer together in a document is more likely to be semantically related,
    # which increases the chance of sampling "clean" (semantically similar) positives.
    context_window_len = 2 * max_span_len

    # Whitespace tokenization makes it much more straightforward to do whole word masking but a
    # user can also provide their own tokenization scheme if they want.
    tokens = tokenizer(text) if tokenizer is not None else text.split()
    num_tokens = len(tokens)
    tok_method = "tokenizer(text)" if tokenizer else "text.split()"
    if min_span_len > max_span_len:
        raise ValueError(
            f"min_span_len ({min_span_len}) must be less than max_span_len ({max_span_len})."
        )
    if context_window_len > num_tokens:
        raise ValueError(
            (
                f"context_window_len ({context_window_len}) must be less than or equal to"
                f" len({tok_method}) ({num_tokens})."
            )
        )

    # Sample a "context window" from the full text
    start = randint(0, num_tokens - context_window_len)
    end = start + context_window_len
    tokens = tokens[start:end]

    adjacent = random.choice([True, False])
    # With 50% probability, sample two adjacent views from the context window
    if adjacent:
        start = 0
        end = max_span_len
        adjacent_view_1 = " ".join(tokens[start:end])
        adjacent_view_2 = " ".join(tokens[end : end + max_span_len])
        return adjacent_view_1, adjacent_view_2
    # With 50% probability, sample "global" and "local" views from the context window
    else:
        # Global view
        start = np.random.randint(0, context_window_len - max_span_len)
        end = start + max_span_len
        global_view = " ".join(tokens[start:end])
        # Local view
        start = np.random.randint(start, end - min_span_len)
        end = start + min_span_len
        local_view = " ".join(tokens[start:end])
        return global_view, local_view
