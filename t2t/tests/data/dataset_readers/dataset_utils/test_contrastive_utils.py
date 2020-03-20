from t2t.data.dataset_readers.dataset_utils import contrastive_utils
import pytest


class TestContrastiveUtils:
    def test_sample_spans_returns_valid_spans(self):
        num_spans = 5
        min_span_width = 5
        sentence = "They may take our lives, but they'll never take our freedom!"

        spans = list(contrastive_utils.sample_spans(sentence, num_spans, min_span_width))
        assert len(spans) == num_spans

        for span in spans:
            assert len(span) <= len(sentence)
            assert len(span) >= min_span_width
            assert " ".join(span) in " ".join(sentence)

    def test_sample_spans_raises_error_for_invalid_min_span_width(self):
        num_spans = 5
        sentence = "They may take our lives, but they'll never take our freedom!"

        min_span_width = len(sentence) + 1  # This is guaranteed to be invalid

        with pytest.raises(ValueError):
            next(contrastive_utils.sample_spans(sentence, num_spans, min_span_width))
