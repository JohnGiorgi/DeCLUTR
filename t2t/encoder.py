from typing import List, Union

import numpy as np

from allennlp.common import util as common_util
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class Encoder:
    """A simple interface to the model for the purposes of embedding sentences/paragraphs."""

    def __init__(self, path_to_allennlp_archive: str, **kwargs) -> None:
        common_util.import_module_and_submodules("t2t")
        archive = load_archive(path_to_allennlp_archive, **kwargs)
        self._predictor = Predictor.from_archive(archive, predictor_name="contrastive")
        self._output_dict_field = "embeddings"

    def __call__(self, inputs: Union[str, List[str]]) -> np.ndarray:
        if isinstance(inputs, str):
            inputs = [inputs]
        json_formatted_inputs = [{"text": input_} for input_ in inputs]
        outputs = self._predictor.predict_batch_json(json_formatted_inputs)
        embeddings = [output[self._output_dict_field] for output in outputs]
        embeddings = np.vstack(embeddings)
        return embeddings
