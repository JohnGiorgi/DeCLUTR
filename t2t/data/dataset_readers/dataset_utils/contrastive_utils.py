from random import randint
from typing import Iterable


def sample_spans(text: str, num_spans: int, min_span_width: int) -> Iterable[str]:
    """Returns a generator that yields random spans from `text`.

    # Parameters

    text : `str`, required
        The string to extract spans from.
    num_spans : `int`, required
        The total number of spans to return.
    min_span_width : `int`, required
        The minimum length of spans, after whitespace tokenization, to sample.
    """

    # No need for fancier tokenization here. Whitespace tokenization will suffice to generate spans.
    tokens = text.split()

    if min_span_width > len(tokens):
        raise ValueError(
            (
                f"min_span_width ({min_span_width}) must be less than or equal to len(text.split())"
                f" ({len(tokens)})."
            )
        )

    sampled = set()

    # TODO (John): There is some probability that this will never complete. Add a check to see if num_spans is
    # valid.
    while True:
        if len(sampled) == num_spans:
            break

        start = randint(0, len(tokens) - min_span_width)
        end = randint(max(start + min_span_width, len(tokens)), len(tokens))

        # Never sample identical spans
        if (start, end) in sampled:
            continue
        sampled.add((start, end))

        yield " ".join(tokens[start:end])
