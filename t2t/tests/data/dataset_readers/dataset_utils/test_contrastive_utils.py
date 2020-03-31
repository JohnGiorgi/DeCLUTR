from t2t.data.dataset_readers.dataset_utils.contrastive_utils import sample_spans
import pytest


class TestContrastiveUtils:
    def test_sample_spans_returns_valid_spans(self):
        num_spans = 5
        min_span_len = 5
        sentence = "They may take our lives, but they'll never take our freedom!"

        spans = list(sample_spans(sentence, num_spans, min_span_len))
        assert len(spans) == num_spans

        for span in spans:
            assert len(span) <= len(sentence)
            assert len(span) >= min_span_len
            assert " ".join(span) in " ".join(sentence)

    def test_sample_spans_raises_error_for_invalid_min_span_len(self):
        num_spans = 5
        sentence = "They may take our lives, but they'll never take our freedom!"

        min_span_len = len(sentence) + 1  # This is guaranteed to be invalid

        with pytest.raises(ValueError):
            next(sample_spans(sentence, num_spans, min_span_len))
