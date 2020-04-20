import pytest
from hypothesis import given
from hypothesis.strategies import text

from t2t.data.dataset_readers.dataset_utils.contrastive_utils import sample_spans


class TestContrastiveUtils:
    @given(text())
    def test_sample_spans(self, text):
        tokenizer = lambda x: x.split()
        num_tokens = len(tokenizer(text))

        # It doesn't make sense for either of these to be less than 1,
        # so cap them at 1.
        max_span_len = max(num_tokens // 2, 1)
        min_span_len = max(int(0.25 * max_span_len), 1)

        # We cannot define a context window for text with less than 3 tokens so sample_spans
        # should raise a ValueError.
        if len(text.split()) <= 3:
            with pytest.raises(ValueError):
                _ = sample_spans(text, max_span_len, min_span_len)
        else:
            for span in sample_spans(text, max_span_len, min_span_len):
                span_len = len(tokenizer(span))

                assert span_len <= max_span_len
                assert span_len >= min_span_len
                # The tokenization process may lead to certain characters (such as escape
                # characters) being dropped, so repeat the tokenization process before performing
                # this check (otherwise a bunch of tests fail).
                assert span in " ".join(tokenizer(text))

    def test_sample_spans_raises_value_error_invalid_min_span_length(self):
        text = "They may take our lives, but they'll never take our freedom!"
        tokenizer = lambda x: x.split()
        num_tokens = len(tokenizer(text))

        max_span_len = num_tokens // 2
        min_span_len = max_span_len + 1  # This is guaranteed to be invalid

        with pytest.raises(ValueError):
            sample_spans(text, max_span_len, min_span_len)

    def test_sample_spans_raises_value_error_invalid_max_span_length(self):
        text = "They may take our lives, but they'll never take our freedom!"
        tokenizer = lambda x: x.split()
        num_tokens = len(tokenizer(text))

        max_span_len = num_tokens  # This is guaranteed to be invalid
        min_span_len = max_span_len // 2

        with pytest.raises(ValueError):
            sample_spans(text, max_span_len, min_span_len)
