import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, text

from declutr.dataset_reader import DeCLUTRDatasetReader


class TestDeCLUTRDatasetReader:
    # Not clear why turning off the deadline is neccesary? Errors out otherwise.
    @settings(deadline=None)
    @given(
        num_anchors=integers(min_value=0, max_value=4),
        num_positives=integers(min_value=1, max_value=4),
        max_span_len=integers(min_value=32, max_value=64),
        min_span_len=integers(min_value=16, max_value=32),
    )
    def test_no_sample_context_manager(
        self, num_anchors: int, num_positives: int, max_span_len: int, min_span_len: int
    ) -> None:
        dataset_reader = DeCLUTRDatasetReader(
            num_anchors=num_anchors,
            num_positives=num_positives,
            max_span_len=max_span_len,
            min_span_len=min_span_len,
        )

        # While in the scope of the context manager, sample_spans should be false.
        # After existing the context manger, it should return to whatever value it was at
        # before entering the contxt manager.
        previous = dataset_reader.sample_spans
        with dataset_reader.no_sample():
            assert not dataset_reader.sample_spans
        assert dataset_reader.sample_spans == previous

    @given(
        num_anchors=integers(min_value=0, max_value=4),
        num_positives=integers(min_value=1, max_value=4),
        max_span_len=integers(min_value=32, max_value=64),
        min_span_len=integers(min_value=16, max_value=32),
    )
    def test_init_raises_value_error_sampling_missing_arguments(
        self, num_anchors: int, num_positives: int, max_span_len: int, min_span_len: int
    ) -> None:
        if num_anchors:  # should only raise the error when num_anchors is truthy
            with pytest.raises(ValueError):
                _ = DeCLUTRDatasetReader(
                    num_anchors=num_anchors,
                    num_positives=num_positives,
                    max_span_len=None,
                    min_span_len=min_span_len,
                )
            with pytest.raises(ValueError):
                _ = DeCLUTRDatasetReader(
                    num_anchors=num_anchors,
                    num_positives=num_positives,
                    max_span_len=max_span_len,
                    min_span_len=None,
                )
            with pytest.raises(ValueError):
                _ = DeCLUTRDatasetReader(
                    num_anchors=num_anchors,
                    num_positives=num_positives,
                    max_span_len=None,
                    min_span_len=None,
                )
            with pytest.raises(ValueError):
                _ = DeCLUTRDatasetReader(
                    num_anchors=num_anchors,
                    num_positives=None,
                    max_span_len=max_span_len,
                    min_span_len=min_span_len,
                )

    @given(
        num_anchors=integers(min_value=0, max_value=4),
        num_positives=integers(min_value=1, max_value=4),
        max_span_len=integers(min_value=32, max_value=64),
        min_span_len=integers(min_value=16, max_value=32),
        sampling_strategy=text(),
    )
    def test_init_raises_value_error_invalid_sampling_strategy(
        self,
        num_anchors: int,
        num_positives: int,
        max_span_len: int,
        min_span_len: int,
        sampling_strategy: str,
    ) -> None:
        if num_anchors:  # should only raise the error when num_spans is truthy
            with pytest.raises(ValueError):
                _ = DeCLUTRDatasetReader(
                    num_anchors=num_anchors,
                    num_positives=num_positives,
                    max_span_len=max_span_len,
                    min_span_len=min_span_len,
                    sampling_strategy=sampling_strategy,
                )
