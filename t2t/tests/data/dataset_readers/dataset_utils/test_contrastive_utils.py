from random import randint

import pytest
from hypothesis import given
from hypothesis.strategies import text

from t2t.data.dataset_readers.dataset_utils.contrastive_utils import sample_anchor_positives


class TestContrastiveUtils:
    def tokenize(self, text):
        return text.split()

    @given(text())
    def test_sample_spans(self, text: str):
        num_tokens = len(self.tokenize(text))

        # It doesn't make sense for either of these to be less than 1.
        max_span_len = max(num_tokens // 2, 1)
        min_span_len = max(int(0.25 * max_span_len), 1)
        # Somewhat arbitrary but we want to know that the tests don't fail for num_spans > 1
        num_spans = randint(1, 4)

        # The sampling procedure will break if we don't have at least one token
        if num_tokens < 1:
            with pytest.raises(ValueError):
                _, _ = sample_anchor_positives(
                    text, max_span_len=max_span_len, min_span_len=min_span_len, num_spans=num_spans
                )
        else:
            anchor, positives = sample_anchor_positives(text, max_span_len, min_span_len)
            # Anchors and positives are sampled in nearly the same way, so just test them together
            spans = [anchor] + positives
            for span in spans:
                span_len = len(self.tokenize(span))

                assert span_len <= max_span_len
                assert span_len >= min_span_len
                # The tokenization process may lead to certain characters (such as escape
                # characters) being dropped, so repeat the tokenization process before performing
                # this check (otherwise a bunch of tests fail).
                assert span in " ".join(self.tokenize(text))

    def test_sample_spans_raises_value_error_invalid_min_span_length(self):
        text = "They may take our lives, but they'll never take our freedom!"
        num_tokens = len(self.tokenize(text))

        max_span_len = num_tokens // 2
        min_span_len = max_span_len + 1  # This is guaranteed to be invalid

        with pytest.raises(ValueError):
            _, _ = sample_anchor_positives(
                text, max_span_len=max_span_len, min_span_len=min_span_len
            )

    def test_sample_spans_raises_value_error_invalid_max_span_length(self):
        text = "They may take our lives, but they'll never take our freedom!"
        num_tokens = len(self.tokenize(text))

        max_span_len = num_tokens + 1  # This is guaranteed to be invalid
        min_span_len = max_span_len // 2

        with pytest.raises(ValueError):
            _, _ = sample_anchor_positives(
                text, max_span_len=max_span_len, min_span_len=min_span_len
            )
