from t2t.data.dataset_readers.dataset_utils import contrastive_utils
import pytest


class TestContrastiveUtils:
    def test_sample_spans_returns_valid_spans(self):
        num_spans = 5
        min_span_len = 5
        sentence = "They may take our lives, but they'll never take our freedom!"

        spans = list(contrastive_utils.sample_spans(sentence, num_spans, min_span_len))
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
            next(contrastive_utils.sample_spans(sentence, num_spans, min_span_len))

    def test_mask_spans(self):
        tokens = "They may take our lives, but they'll never take our freedom!".split()
        mask_token = "[MASK]"
        masking_budget = 0.15
        max_span_len = 10
        p = 0.2

        masked_tokens = contrastive_utils.mask_spans(tokens, mask_token, masking_budget, max_span_len, p)

        # Check that the number of tokens did not change
        assert len(masked_tokens) == len(tokens)
        # Check that the % of masked tokens is within the budget
        assert masked_tokens.count(mask_token) / len(tokens) <= masking_budget

    def test_mask_spans_value_error(self):
        text = "They may take our lives, but they'll never take our freedom!".split()
        mask_token = "[MASK]"
        max_span_len = len(text) + 1  # This is guaranteed to be invalid

        with pytest.raises(ValueError):
            contrastive_utils.mask_spans(text, mask_token, max_span_len=max_span_len)
