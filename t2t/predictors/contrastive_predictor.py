from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("contrastive")
class ContrastivePredictor(Predictor):
    """Predictor wrapper for the ContrastiveTextEncoder"""

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict["text"]
        # Context manager ensures that the sample_spans property of our DatasetReader is False
        with no_sample(self._dataset_reader):
            return self._dataset_reader.text_to_instance(text=text)


class no_sample:
    """Context manager that disables sampling of spans for the given `dataset_reader`."""

    def __init__(self, dataset_reader: DatasetReader):
        self.dataset_reader = dataset_reader
        self.prev = self.dataset_reader.sample_spans

    def __enter__(self):
        self.dataset_reader.sample_spans = False

    def __exit__(self, *args):
        self.dataset_reader.sample_spans = self.prev
