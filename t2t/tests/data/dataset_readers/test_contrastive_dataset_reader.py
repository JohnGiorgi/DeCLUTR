from t2t.data.dataset_readers import ContrastiveDatasetReader


class TestContrastiveDatasetReader:
    def test_no_sample_context_manager(self):
        dataset_reader = ContrastiveDatasetReader(sample_spans=True)

        # While in the scope of the context manager, sample_spans should be false.
        # After existing the context manger, it should return to True.
        assert dataset_reader.sample_spans
        with dataset_reader.no_sample():
            assert not dataset_reader.sample_spans
        assert dataset_reader.sample_spans
