from typing import Callable, List, Optional, Tuple

import numpy as np
import itertools
import more_itertools as mit

from allennlp.common.logging import AllenNlpLogger

logger = AllenNlpLogger(__name__)


def sample_anchor_positive_pairs(
    text: str,
    num_anchors: int,
    num_positives: int,
    max_span_len: int,
    min_span_len: int,
    sampling_strategy: Optional[str] = None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Tuple[List[str], List[str]]:
    """Returns a tuple of `num_anchors` anchor spans and `num_positives` positive spans sampled from
    `text`.

    # Parameters

    text : `str`, required
        The string to extract anchor and positive spans from.
    num_anchors : `int`, required
        The number of spans to sample from `text` to serve as anchors.
    num_positives : `int`, required
        The number of spans to sample from `text` to serve as positives (per anchor).
    max_span_len : `int`, required
        The maximum length of spans, after tokenization, to sample.
    min_span_len : `int`, required
        The minimum length of spans, after tokenization, to sample.
    sampling_strategy : `str`, optional (default = None)
        One of "subsuming" or "adjacent". If "subsuming," positive spans are always subsumed by the
        anchor. If "adjacent", positive spans are always adjacent to the anchor. If not provided,
        positives may be subsumed, adjacent to, or overlapping with the anchor. Has no effect if
        `num_spans` is not provided.
    tokenizer : `Callable`, optional (default = None)
        Optional tokenizer to use before sampling spans. If `None`, `text.split()` is used.
    """
    # Tokenize the incoming text. Whitespace tokenization is much more straightforward
    # (we don't need to worry about chopping up subword tokens), but a user can also provide their
    # own tokenization scheme if they want.
    tokens = tokenizer(text) if tokenizer is not None else text.split()
    tok_method = "tokenizer(text)" if tokenizer else "text.split()"
    num_tokens = len(tokens)

    # Several checks on the parameters based to this function. The first check on length is mostly
    # arbitrary, but it prevents the Hypothesis tests from breaking. And it makes little sense to
    # sample from extremely short documents.
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

    # Valid anchor starts are token indices which begin a token span of at least max_span_len.
    anchors, positives = [], []
    valid_anchor_starts = list(range(0, num_tokens - max_span_len + 1))
    for _ in range(num_anchors):
        # Sample the anchor length from a beta distribution skewed towards longer spans, the
        # intuition being that longer spans have the best chance of being representative of the
        # document they are sampled from.
        anchor_len = int(np.random.beta(4, 2) * (max_span_len - min_span_len) + min_span_len)
        # Sample an anchor start position from the list of valid positions. Once sampled, remove it
        # (and its immediate neighbours) from consideration.
        anchor_start_idx = np.random.randint(len(valid_anchor_starts))
        anchor_start = valid_anchor_starts[anchor_start_idx]
        # Remove anchor and surrounding buffer from available starts
        anchor_buffer = range(max(0, anchor_start - max_span_len + 1), anchor_start + anchor_len + max_span_len + 1)
        valid_anchor_starts = [x for x in valid_anchor_starts if x not in anchor_buffer]
        # Group consecutive start point and remove those that are too small for future anchors
        valid_anchor_starts = [list(group) for group in mit.consecutive_groups(valid_anchor_starts)]
        valid_anchor_starts = [x for x in valid_anchor_starts if (len(x) >= max_span_len or (num_tokens - max_span_len) in x)]
        valid_anchor_starts = list(itertools.chain.from_iterable(valid_anchor_starts))
        # Grab the anchor with its sampled start and length.
        anchor_end = anchor_start + anchor_len
        anchors.append(" ".join(tokens[anchor_start:anchor_end]))

        # Sample positives from around the anchor. The intuition being that text that appears close
        # together is the same document is likely to be semantically similar.
        for _ in range(num_positives):
            # Sample positive length from a beta distribution skewed towards shorter spans. The
            # idea is to promote diversity and minimize the amount of overlapping text.
            positive_len = int(np.random.beta(2, 4) * (max_span_len - min_span_len) + min_span_len)
            # A user can specify a subsuming or adjacent only sampling strategy.
            if sampling_strategy == "subsuming":
                # To be strictly subsuming, we cannot allow the positive_len > anchor_len.
                if positive_len > anchor_len:
                    logger.warning_once(
                        (
                            f"Positive length was longer than anchor length. Temporarily reducing"
                            f" max length of positives. This message will not be displayed again."
                        )
                    )
                    positive_len = int(
                        np.random.beta(2, 4) * (anchor_len - min_span_len) + min_span_len
                    )
                positive_start = np.random.randint(
                    anchor_start, anchor_end - positive_len + 1  # randint is high-exclusive
                )
            elif sampling_strategy == "adjacent":
                # Restrict positives to a length that will allow them to be adjacent to the anchor
                # without running off the edge of the document. If documents are sufficiently
                # long, this won't be a problem and max_positive_len will equal max_span_len.
                max_positive_len = min(max_span_len, max(anchor_start, num_tokens - anchor_end))
                if positive_len > max_positive_len:
                    logger.warning_once(
                        (
                            f"There is no room to sample an adjacent positive span. Temporarily"
                            f" reducing the maximum span length of positives. This message will not"
                            f" be displayed again."
                        )
                    )
                positive_len = int(
                    np.random.beta(2, 4) * (max_positive_len - min_span_len) + min_span_len
                )
                # There are two types of adjacent positives, those that border the beginning of the
                # anchor and those that border the end. The checks above guarantee at least one of
                # these is valid. Here we just choose from the valid positive starts at random.
                valid_starts = []
                if anchor_start - positive_len > 0:
                    valid_starts.append(anchor_start - positive_len)
                if anchor_end + positive_len <= num_tokens:
                    valid_starts.append(anchor_end)
                positive_start = np.random.choice(valid_starts)
            else:
                # By default, spans may be adjacent or overlap with each other and the anchor.
                # Careful not to run off the edges of the document (this error may pass silently).
                positive_start = np.random.random_integers(
                    max(0, anchor_start - positive_len),
                    min(anchor_end, num_tokens - positive_len) + 1,  # randint is high-exclusive
                )

            positive_end = positive_start + positive_len
            positives.append(" ".join(tokens[positive_start:positive_end]))

    return anchors, positives
