from t2t.data.dataset_readers import ContrastiveDatasetReader
from t2t.predictors import no_sample


class TestContrastivePredictor:
    def test_no_sample_context_manager(self):
        dataset_readers = ContrastiveDatasetReader(sample_spans=True)

        # While in the scope of the context manager, sample_spans should be false.
        # After existing the context manger, it should return to True.
        with no_sample(dataset_readers):
            assert not dataset_readers.sample_spans

        assert dataset_readers._sample_spans
