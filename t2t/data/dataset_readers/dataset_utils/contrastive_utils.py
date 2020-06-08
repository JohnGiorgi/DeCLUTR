from random import choice, randint
from typing import Callable, List, Optional, Tuple

import numpy as np

from allennlp.common.logging import AllenNlpLogger

logger = AllenNlpLogger(__name__)


def sample_anchor_positives(
    text: str,
    num_anchors: int,
    num_positives: int,
    max_span_len: int,
    min_span_len: int,
    sampling_strategy: Optional[str] = None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Tuple[List[str], List[str]]:
    """Returns a tuple of anchor (`str`) and `num_spans` positive (`List[str]`) spans sampled from
    `text`.

    # Parameters

    text : `str`, required
        The string to extract anchor and positive spans from.
    max_span_len : `int`, required
        The maximum length of spans, after tokenization, to sample.
    min_span_len : `int`, optional
        The minimum length of spans, after tokenization, to sample.
    num_anchors : `int`, optional (default = 1)
        The number of spans to sample from `text` to serve as anchors.
    num_positives : `int`, optional (default = 1)
        The number of spans to sample from `text` to serve as positives (per anchor).
    sampling_strategy : `str`, optional (default = None)
        One of "subsuming" or "adjacent". If "subsuming," positive spans are always subsumed by the
        anchor. If "adjacent", positive spans are always adjacent to the anchor. If not provided,
        positives may be subsumed, adjacent to, or overlapping with the anchor. Has no effect if
        `num_spans` is not provided.
    tokenizer : `Callable`, optional
        Optional tokenizer to use before sampling spans. If `None`, `text.split()` is used.
    """
    # Whitespace tokenization is much more straightforward (don't need to worry about chopping up
    # subword tokens), but a user can also provide their own tokenization scheme if they want.
    tokens = tokenizer(text) if tokenizer is not None else text.split()
    num_tokens = len(tokens)
    tok_method = "tokenizer(text)" if tokenizer else "text.split()"

    # This is mostly arbitrary, but it prevents the Hypothesis tests from breaking. And it makes
    # little sense to sample from extremely short documents.
    if num_tokens < 10:
        raise ValueError(
            (f"len({tok_method}) should be at least 10 (ideally much longer), got {num_tokens}.")
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

    anchors, positives = [], []
    # Valid anchor starts are token indicies which begin a token span of at least max_span_len
    valid_anchor_starts = list(range(0, num_tokens - max_span_len + 1, max_span_len))
    for _ in range(num_anchors):
        # Sample the anchor length from a beta distribution skewed towards longer spans, the
        # intuition being that longer spans have the best chance of being representative of the
        # document they are sampled from.
        anchor_length = int(np.random.beta(4, 2) * (max_span_len - min_span_len) + min_span_len)
        # Sample an anchor start position from the list of valid positions.
        anchor_start_idx = randint(0, len(valid_anchor_starts) - 1)
        anchor_start = valid_anchor_starts[anchor_start_idx]
        # Remove the sampled start and its neighbors from consideration.
        del valid_anchor_starts[max(0, anchor_start_idx - 1) : anchor_start_idx + 2]
        # Grab the anchor with its sampled start and length.
        anchor_end = anchor_start + anchor_length
        anchors.append(" ".join(tokens[anchor_start:anchor_end]))

        # Sample positives from around the anchor. The intuition being that text that appears close
        # together is the same document is likely to be semantically similar.
        for _ in range(num_positives):
            # Their length is sampled from a beta distribution skewed towards shorter spans. The idea
            # is to promote diversity and minimize the amount of overlapping text.
            positive_length = int(
                np.random.beta(2, 4) * (max_span_len - min_span_len) + min_span_len
            )
            # A user can specify a subsuming or adjacent only sampling strategy.
            if sampling_strategy == "subsuming":
                # To be strictly subsuming, we cannot allow the positive_length > anchor_length.
                if positive_length > anchor_length:
                    logger.warning_once(
                        (
                            f"Positive length was longer than anchor length. Temporarily reducing max"
                            f" span length of positives. This message will not be displayed again."
                        )
                    )
                    positive_length = int(
                        np.random.beta(2, 4) * (anchor_length - min_span_len) + min_span_len
                    )
                positive_start = randint(anchor_start, anchor_end - positive_length)
            elif sampling_strategy == "adjacent":
                # We have to restrict positives to a length that will allow them to be adjacent to
                # the anchor without running off the edge of the document. If documents are sufficiently
                # long, this won't be a problem and the max_positive_length will equal max_span_len.
                max_positive_len = min(max_span_len, max(anchor_start, num_tokens - anchor_end))
                if positive_length > max_positive_len:
                    logger.warning_once(
                        (
                            f"There is no room to sample an adjacent positive span. Temporarily"
                            f" reducing the maximum span length of positives. This message will not be"
                            f" displayed again."
                        )
                    )
                positive_length = int(
                    np.random.beta(2, 4) * (max_positive_len - min_span_len) + min_span_len
                )
                # There are two types of adjacent positives, those that border the beginning of the
                # anchor and those that border the end. The checks above guarantee at least one of these
                # is valid.
                valid_starts = []
                if anchor_start - positive_length > 0:
                    valid_starts.append(anchor_start - positive_length)
                if anchor_end + positive_length <= num_tokens:
                    valid_starts.append(anchor_end)
                positive_start = choice(valid_starts)
            else:
                # By default, spans may be adjacent or overlap with each other and the anchor.
                # Be careful not to run off the edges of the document, as this error will pass silently.
                positive_start = randint(
                    max(0, anchor_start - positive_length),
                    min(anchor_end, num_tokens - positive_length),
                )

            positive_end = positive_start + positive_length
            positives.append(" ".join(tokens[positive_start:positive_end]))

    return anchors, positives
