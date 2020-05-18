from itertools import zip_longest
from random import randint
from typing import Union

import pytest
from hypothesis import given
from hypothesis.strategies import integers, sampled_from, text

from t2t.data.dataset_readers.dataset_utils.contrastive_utils import sample_anchor_positives


class TestContrastiveUtils:
    def tokenize(self, text):
        return text.split()

    @given(
        text=text(),
        num_spans=integers(min_value=1, max_value=4),
        sampling_strategy=sampled_from(["subsuming", "adjacent", None]),
    )
    def test_sample_spans(self, text: str, num_spans: int, sampling_strategy: Union[str, None]):
        num_tokens = len(self.tokenize(text))
        # These represent sensible defaults
        max_span_len = num_tokens // 2
        min_span_len = randint(1, num_tokens // 2) if num_tokens // 2 > 1 else 1

        # The sampling procedure often breaks if we don't have at least ten tokens, so we set
        # a strict lower bound.
        if num_tokens < 10:
            with pytest.raises(ValueError):
                _, _ = sample_anchor_positives(
                    text,
                    max_span_len=max_span_len,
                    min_span_len=min_span_len,
                    num_spans=num_spans,
                    sampling_strategy=sampling_strategy,
                )
        else:
            anchor, positives = sample_anchor_positives(
                text,
                max_span_len=max_span_len,
                min_span_len=min_span_len,
                num_spans=num_spans,
                sampling_strategy=sampling_strategy,
            )
            assert len(positives) == num_spans
            for anc, pos in zip_longest(anchor, positives, fillvalue=anchor):
                anc_len, pos_len = len(self.tokenize(anc)), len(self.tokenize(pos))

                # Do basic tests for both anchors and positives
                assert anc_len <= max_span_len
                assert anc_len >= min_span_len
                assert pos_len <= max_span_len
                assert pos_len >= min_span_len
                # The tokenization process may lead to certain characters (such as escape
                # characters) being dropped, so repeat the tokenization process before performing
                # this check (otherwise a bunch of tests fail).
                assert anc in " ".join(self.tokenize(text))
                assert pos in " ".join(self.tokenize(text))

                # Test that specific sampling strategies are obeyed
                if sampling_strategy == "subsuming":
                    assert pos in " ".join(self.tokenize(anc))
                elif sampling_strategy == "adjacent":
                    assert pos not in " ".join(self.tokenize(anc))

    def test_sample_spans_raises_value_error_invalid_min_span_length(self):
        text = "They may take our lives, but they'll never take our freedom!"
        num_tokens = len(self.tokenize(text))

        max_span_len = num_tokens - 1  # This is guaranteed to be valid
        min_span_len = max_span_len + 1  # This is guaranteed to be invalid

        with pytest.raises(ValueError):
            _, _ = sample_anchor_positives(
                text, max_span_len=max_span_len, min_span_len=min_span_len
            )

    def test_sample_spans_raises_value_error_invalid_max_span_length(self):
        text = "They may take our lives, but they'll never take our freedom!"
        num_tokens = len(self.tokenize(text))

        max_span_len = num_tokens + 1  # This is guaranteed to be invalid
        min_span_len = max_span_len - 1  # This is guaranteed to be valid

        with pytest.raises(ValueError):
            _, _ = sample_anchor_positives(
                text, max_span_len=max_span_len, min_span_len=min_span_len
            )
