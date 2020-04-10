from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from typing import Union, List
from allennlp.common import util as common_util
import numpy as np


class Encoder:
    """A simple interface to the model for the purposes of embedding sentences/paragraphs."""

    def __init__(self, path_to_allennlp_archive: str, **kwargs) -> None:
        common_util.import_module_and_submodules("t2t")
        archive = load_archive(path_to_allennlp_archive, **kwargs)
        self._predictor = Predictor.from_archive(archive, predictor_name="contrastive")

    def __call__(self, inputs: Union[str, List[str]]):
        if isinstance(inputs, str):
            inputs = [inputs]
        json_formatted_inputs = [{"text": input_} for input_ in inputs]
        outputs = self._predictor.predict_batch_json(json_formatted_inputs)
        embeddings = [output["embeddings"] for output in outputs]
        embeddings = np.vstack(embeddings)
        return embeddings
