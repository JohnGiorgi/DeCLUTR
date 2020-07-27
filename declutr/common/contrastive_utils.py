from typing import Callable, List, Optional, Tuple

import numpy as np

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
    """Returns a `Tuple` of `List`s, containing `num_anchors` anchor spans and `num_positives`
    positive spans sampled from `text`.

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
    sampling_strategy : `str`, optional (default = `None`)
        One of `"subsuming"` or `"adjacent"`. If `"subsuming"`, positive spans are always subsumed
        by the anchor. If `"adjacent"`, positive spans are always adjacent to the anchor. If not
        provided, positives may be subsumed, adjacent to, or overlapping with the anchor.
    tokenizer : `Callable`, optional (default = `None`)
        Optional tokenizer to use before sampling spans. If `None`, `text.split()` is used.
    """
    # Tokenize the incoming text. Whitespace tokenization is much more straightforward
    # (we don't need to worry about chopping up subword tokens), but a user can also provide
    # their own tokenization scheme if they want.
    tokens = tokenizer(text) if tokenizer is not None else text.split()
    tok_method = "tokenizer(text)" if tokenizer else "text.split()"
    num_tokens = len(tokens)

    if num_tokens < num_anchors * max_span_len * 2:
        raise ValueError(
            f"len({tok_method}) should be at least {num_anchors * max_span_len * 2}"
            f" (num_anchors * max_span_len * 2), got {num_tokens}."
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
    valid_anchor_starts = list(range(0, num_tokens - max_span_len + 1, max_span_len))
    for i in range(num_anchors):
        # Sample the anchor length from a beta distribution skewed towards longer spans, the
        # intuition being that longer spans have the best chance of being representative of the
        # document they are sampled from.
        anchor_len = int(np.random.beta(4, 2) * (max_span_len - min_span_len) + min_span_len)
        # This check prevents an edge case were we run out of valid_anchor_starts.
        if len(valid_anchor_starts) // (num_anchors - i) < num_anchors - i:
            anchor_start_idx = np.random.choice([0, len(valid_anchor_starts) - 1])
        else:
            anchor_start_idx = np.random.randint(len(valid_anchor_starts))
        # When num_anchors = 1, this is equivalent to uniformly sampling that starting position.
        anchor_start = np.random.randint(
            valid_anchor_starts[anchor_start_idx],
            # randint is high-exclusive
            valid_anchor_starts[anchor_start_idx] + max_span_len - anchor_len + 1,
        )
        # Once sampled, remove an anchor (and its immediate neighbours) from consideration.
        del valid_anchor_starts[max(0, anchor_start_idx - 1) : anchor_start_idx + 2]
        anchor_end = anchor_start + anchor_len
        anchors.append(" ".join(tokens[anchor_start:anchor_end]))

        # Sample positives from around the anchor. The intuition being that text that appears
        # close together is the same document is likely to be semantically similar.
        for _ in range(num_positives):
            # A user can specify a subsuming or adjacent only sampling strategy.
            if sampling_strategy == "subsuming":
                # To be strictly subsuming, we cannot allow the positive_len > anchor_len.
                positive_len = int(
                    np.random.beta(2, 4) * (anchor_len - min_span_len) + min_span_len
                )
                # randint is high-exclusive
                positive_start = np.random.randint(anchor_start, anchor_end - positive_len + 1)
            elif sampling_strategy == "adjacent":
                # Restrict positives to a length that will allow them to be adjacent to the anchor
                # without running off the edge of the document. If the anchor has sufficent room on
                # either side, this won't be a problem and max_positive_len will equal max_span_len.
                max_positive_len = min(max_span_len, max(anchor_start, num_tokens - anchor_end))
                if max_positive_len < max_span_len:
                    logger.warning_once(
                        (
                            "There is no room to sample an adjacent positive span. Temporarily"
                            " reducing the maximum span length of positives. This message will not"
                            " be displayed again."
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
                # Sample positive length from a beta distribution skewed towards shorter spans. The
                # idea is to promote diversity and minimize the amount of overlapping text.
                positive_len = int(
                    np.random.beta(2, 4) * (max_span_len - min_span_len) + min_span_len
                )
                # By default, spans may be adjacent or overlap with each other and the anchor.
                # Careful not to run off the edges of the document (this error may pass silently).
                positive_start = np.random.randint(
                    max(0, anchor_start - positive_len),
                    min(anchor_end, num_tokens - positive_len) + 1,  # randint is high-exclusive
                )

            positive_end = positive_start + positive_len
            positives.append(" ".join(tokens[positive_start:positive_end]))

    return anchors, positives
