import copy
from math import floor
from random import randint
from typing import Callable, Iterable, List, Optional

import numpy as np


def sample_spans(
    text: str,
    num_spans: int,
    min_span_len: int,
    tokenizer: Optional[Callable] = None,
    span_masking: bool = False,
    **kwargs,
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
    span_masking : `bool`, optional
        If True, **kwargs will be passed to `mask_spans`, which will mask random, contiguous subsequences from the
        sampled spans before they are returned.
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
        # 3. Optionally, perform span masking (or "whole word masking")
        if span_masking:
            span = mask_spans(tokens[start:end], **kwargs)
        else:
            span = tokens[start:end]

        yield " ".join(span)


def mask_spans(
    tokens: List[str], mask_token: str, masking_budget: int = 0.15, max_span_len: int = 10, p: float = 0.2
) -> List[str]:
    """Implements a masking procedure similar to: "SpanBERT: Improving Pre-training by Representing and Predicting
    Spans". In particular, the default masking budget, minimum span length to mask, and p parameter of the
    geometric distribution to sample from are all taken from SpanBERT. For now, we do NOT replace 10% tokens with
    random tokens. See: https://arxiv.org/abs/1907.10529 for more details.
    """
    tokens = copy.deepcopy(tokens)  # avoid modifying the incoming text in place
    num_tokens = len(tokens)

    if max_span_len > num_tokens:
        raise ValueError(
            (f"max_span_len ({max_span_len}) must be less than or equal to len(text.split())" f" ({num_tokens}).")
        )

    budget = 0  # percent of words to be masked
    while budget < masking_budget:
        # 1. Determine what span lengths are within our budget. If the budget is spent, early-exit
        len_within_budget = floor(num_tokens * (masking_budget - budget))
        if len_within_budget < 1:
            break
        # 2. Sample span length from a geometric distribution, which is skewed towards short spans.
        # mean(span_length) = 3.8
        span_length = min(len_within_budget, np.random.geometric(p=p), max_span_len)
        # 3. Sample the start index of the span uniformly
        start = randint(0, num_tokens - span_length)
        end = start + span_length
        # 4. Mask the sampled span with the given mask token and update the budget
        tokens[start:end] = [mask_token] * span_length
        budget += span_length / num_tokens

    return tokens
