from typing import List, Union

import numpy as np
import torch

from allennlp.common import util as common_util
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class Encoder:
    """A simple interface to the model for the purposes of embedding sentences/paragraphs.

    # Parameters

   path_to_allennlp_archive : `str`, optional
        Path to a serialized AllenNLP archive..
    sphereize : `bool`
        If `True` embeddings will be l2-normalized and shifted by the centroid. Defaults to `False`.
    """

    def __init__(self, path_to_allennlp_archive: str, sphereize: bool = False, **kwargs) -> None:
        common_util.import_module_and_submodules("declutr")
        archive = load_archive(path_to_allennlp_archive, **kwargs)
        self._predictor = Predictor.from_archive(archive, predictor_name="contrastive")
        self._output_dict_field = "embeddings"
        self._sphereize = sphereize

    @torch.no_grad()
    def __call__(self, inputs: Union[str, List[str]]) -> np.ndarray:
        if isinstance(inputs, str):
            inputs = [inputs]
        json_formatted_inputs = [{"text": input_} for input_ in inputs]
        outputs = self._predictor.predict_batch_json(json_formatted_inputs)
        embeddings = [output[self._output_dict_field] for output in outputs]
        embeddings = torch.as_tensor(embeddings)
        if self._sphereize:
            centroid = torch.mean(embeddings, dim=0)
            embeddings -= centroid
            embeddings /= torch.norm(embeddings, dim=1, keepdim=True)

        return embeddings.numpy()
