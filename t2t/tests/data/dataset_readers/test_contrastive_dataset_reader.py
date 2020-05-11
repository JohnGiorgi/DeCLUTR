import pytest

from t2t.data.dataset_readers import ContrastiveDatasetReader


class TestContrastiveDatasetReader:
    def test_no_sample_context_manager(self):
        # The value of these arguments are arbitrary, they just need to be > 0 so that
        # ContrastiveDatasetReader.sample_spans is set to True.
        dataset_reader = ContrastiveDatasetReader(num_spans=1, max_span_len=512, min_span_len=32)

        # While in the scope of the context manager, sample_spans should be false.
        # After existing the context manger, it should return to True.
        assert dataset_reader.sample_spans
        with dataset_reader.no_sample():
            assert not dataset_reader.sample_spans
        assert dataset_reader.sample_spans

    def test_init_raises_value_error_no_max_min_span_length(self):
        with pytest.raises(ValueError):
            _ = ContrastiveDatasetReader(num_spans=1, max_span_len=None, min_span_len=32)
        with pytest.raises(ValueError):
            _ = ContrastiveDatasetReader(num_spans=1, max_span_len=512, min_span_len=None)
        with pytest.raises(ValueError):
            _ = ContrastiveDatasetReader(num_spans=1, max_span_len=None, min_span_len=None)
