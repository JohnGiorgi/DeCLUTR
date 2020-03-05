from t2t.data.dataset_readers.dataset_utils import contrastive_utils
import pytest


class TestContrastiveUtils():
    def test_sample_spans_returns_valid_spans(self):
        num_spans = 5
        min_span_width = 5
        text = "They may take our lives, but they'll never take our freedom!"
        text_length = len(text.split())

        for span in contrastive_utils.sample_spans(text, num_spans, min_span_width):
            span_length = len(span.split())

            assert span_length <= text_length
            assert span_length >= min_span_width
            assert span in text

    def test_sample_spans_raises_error_for_invalid_min_span_width(self):
        num_spans = 5
        text = "They may take our lives, but they'll never take our freedom!"
        text_length = len(text.split())

        min_span_width = text_length + 1  # This is guaranteed to be invalid

        with pytest.raises(ValueError):
            next(contrastive_utils.sample_spans(text, num_spans, min_span_width))
