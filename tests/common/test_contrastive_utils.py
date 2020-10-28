import random
from typing import List, Union

import pytest
from hypothesis import given
from hypothesis.strategies import integers, sampled_from

from declutr.common.contrastive_utils import sample_anchor_positive_pairs


class TestContrastiveUtils:
    def tokenize(self, text) -> List[str]:
        return text.split()

    @given(
        num_anchors=integers(min_value=1, max_value=4),
        num_positives=integers(min_value=1, max_value=4),
        sampling_strategy=sampled_from(["subsuming", "adjacent", None]),
    )
    def test_sample_spans(
        self,
        inputs: List[str],
        num_anchors: int,
        num_positives: int,
        sampling_strategy: Union[str, None],
    ) -> None:

        for text in inputs:
            tokens = self.tokenize(text)
            num_tokens = len(tokens)

            # Really short examples make the tests unreliable.
            if num_tokens < 7:
                continue

            # These represent sensible defaults
            max_span_len = num_tokens // 4
            min_span_len = random.randint(1, max_span_len) if max_span_len > 1 else 1

            if num_tokens < num_anchors * max_span_len * 2:
                with pytest.raises(ValueError):
                    _, _ = sample_anchor_positive_pairs(
                        text,
                        num_anchors=num_anchors,
                        num_positives=num_positives,
                        max_span_len=max_span_len,
                        min_span_len=min_span_len,
                        sampling_strategy=sampling_strategy,
                    )
            else:
                anchors, positives = sample_anchor_positive_pairs(
                    text,
                    num_anchors=num_anchors,
                    num_positives=num_positives,
                    max_span_len=max_span_len,
                    min_span_len=min_span_len,
                    sampling_strategy=sampling_strategy,
                )
                assert len(anchors) == num_anchors
                assert len(positives) == num_anchors * num_positives
                for i, anchor in enumerate(anchors):
                    # Several simple checks for valid anchors.
                    anchor_tokens = self.tokenize(anchor)
                    anchor_length = len(anchor_tokens)
                    assert anchor_length <= max_span_len
                    assert anchor_length >= min_span_len
                    # The tokenization process may lead to certain characters (such as escape
                    # characters) being dropped, so repeat the tokenization process before
                    # performing this check (otherwise a bunch of tests fail).
                    assert anchor in " ".join(tokens)
                    for j in range(i * num_positives, i * num_positives + num_positives):
                        # Several simple checks for valid positives.
                        positive = positives[j]
                        positive_tokens = self.tokenize(positive)
                        positive_length = len(positive_tokens)
                        assert positive_length <= max_span_len
                        assert positive_length >= min_span_len
                        assert positive in " ".join(tokens)
                        # Test that specific sampling strategies are obeyed.
                        if sampling_strategy == "subsuming":
                            assert positive in " ".join(anchor_tokens)
                        elif sampling_strategy == "adjacent":
                            assert positive not in " ".join(anchor_tokens)

    @given(
        num_anchors=integers(min_value=1, max_value=4),
        num_positives=integers(min_value=1, max_value=4),
    )
    def test_sample_spans_raises_value_error_invalid_min_span_length(
        self, num_anchors: int, num_positives: int
    ) -> None:
        text = "They may take our lives, but they'll never take our freedom!"
        num_tokens = len(self.tokenize(text))

        max_span_len = num_tokens - 1  # This is guaranteed to be valid.
        min_span_len = max_span_len + 1  # This is guaranteed to be invalid.

        with pytest.raises(ValueError):
            _, _ = sample_anchor_positive_pairs(
                text,
                num_anchors=num_anchors,
                num_positives=num_positives,
                max_span_len=max_span_len,
                min_span_len=min_span_len,
            )

    @given(
        num_anchors=integers(min_value=1, max_value=4),
        num_positives=integers(min_value=1, max_value=4),
    )
    def test_sample_spans_raises_value_error_invalid_max_span_length(
        self, num_anchors: int, num_positives: int
    ) -> None:
        text = "They may take our lives, but they'll never take our freedom!"
        num_tokens = len(self.tokenize(text))

        max_span_len = num_tokens + 1  # This is guaranteed to be invalid.
        min_span_len = max_span_len - 1  # This is guaranteed to be valid.

        with pytest.raises(ValueError):
            _, _ = sample_anchor_positive_pairs(
                text,
                num_anchors=num_anchors,
                num_positives=num_positives,
                max_span_len=max_span_len,
                min_span_len=min_span_len,
            )
