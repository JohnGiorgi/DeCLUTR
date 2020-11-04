import warnings
from operator import itemgetter
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import torch
from allennlp.common import util as common_util
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from validators.url import url

from declutr.common.data_utils import sanitize

PRETRAINED_MODELS = {
    "declutr-small": "https://github.com/JohnGiorgi/DeCLUTR/releases/download/v0.1.0rc1/declutr-small.tar.gz",
    "declutr-base": "https://github.com/JohnGiorgi/DeCLUTR/releases/download/v0.1.0rc1/declutr-base.tar.gz",
}


class Encoder:
    """A simple interface to the model for the purposes of embedding sentences/paragraphs.

    # Example Usage

    ```python
    from declutr import Encoder

    # This can be a path on disk to a model you have trained yourself OR
    # the name of one of our pretrained models.
    pretrained_model_or_path = "declutr-small"

    encoder = Encoder(pretrained_model_or_path)
    embeddings = encoder([
        "A smiling costumed woman is holding an umbrella.",
        "A happy woman in a fairy costume holds an umbrella."
    ])
    ```

    # Parameters

    pretrained_model_name_or_path : `str`, required
        Path to a serialized AllenNLP archive or a model name from:
        `declutr.encoder.PRETRAINED_MODEL_URLS`
    sphereize : `bool`, optional (default = `False`)
        If `True` embeddings will be l2-normalized and shifted by the centroid. Defaults to `False`.
    **kwargs : `Dict`, optional
        Keyword arguments that will be passed to `allennlp.models.archival.load_archive`. This is
        useful, for example, to specify a CUDA device id with `cuda_device`. See:
        https://docs.allennlp.org/master/api/models/archival/#load_archive for more details.
    """

    _output_dict_field = "embeddings"

    def __init__(
        self, pretrained_model_name_or_path: str, sphereize: bool = False, **kwargs
    ) -> None:
        if pretrained_model_name_or_path in PRETRAINED_MODELS:
            pretrained_model_name_or_path = PRETRAINED_MODELS[pretrained_model_name_or_path]
        common_util.import_module_and_submodules("declutr")
        archive = load_archive(pretrained_model_name_or_path, **kwargs)
        self._predictor = Predictor.from_archive(archive, predictor_name="declutr")
        self._sphereize = sphereize

    @torch.no_grad()
    def __call__(
        self, inputs: Union[str, List[str]], batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Returns a numpy array of embeddings, one for each item in `inputs`.

        # Parameters

        inputs : `Union[str, List[str]]`, required
            The input text to embed. Can be a string, list of strings, or a filepath/URL to a text
            file with one input per line.
        batch_size : `int`, optional
            If given, the `inputs` will be batched before embedding.
        """
        if isinstance(inputs, str):
            if Path(inputs).is_file() or url(inputs):
                inputs = Path(cached_path(inputs)).read_text().split("\n")
            else:
                inputs = [inputs]

        if batch_size is None:
            unsort = False
            batch_size = len(inputs)
        else:
            # Sort the inputs by length, maintaining the original indices so we can un-sort
            # before returning the embeddings. This speeds up embedding by minimizing the
            # amount of computation performed on pads. Because this sorting happens before
            # tokenization, it is only a proxy of the true lengths of the inputs to the model.
            # In the future, it would be better to use the built-in bucket sort of AllenNLP,
            # which would lead to an even larger speedup.
            unsort = True
            sorted_indices, inputs = cast(
                Tuple[List[int], List[str]], zip(*sorted(enumerate(inputs), key=itemgetter(1)))
            )  # tell mypy explicitly the types of items in the unpacked tuple
            unsorted_indices, _ = zip(*sorted(enumerate(sorted_indices), key=itemgetter(1)))

        embeddings: torch.FloatTensor = []  # promise mypy we will behave
        for i in range(0, len(inputs), batch_size):
            batch_json = [{"text": sanitize(input_)} for input_ in inputs[i : i + batch_size]]
            outputs = self._predictor.predict_batch_json(batch_json)
            outputs = torch.as_tensor(
                # Accumulating the tensors on the GPU would quickly lead to OOM.
                [output[self._output_dict_field] for output in outputs],
                device="cpu",
            )
            embeddings.append(outputs)
        embeddings = torch.cat(embeddings)
        # Make sure to unsort the embeddings if they were sorted.
        if unsort:
            unsorted_indices = torch.as_tensor(unsorted_indices, dtype=torch.long)
            embeddings = torch.index_select(embeddings, dim=0, index=unsorted_indices)
        if self._sphereize:
            if embeddings.size(0) > 1:
                centroid = torch.mean(embeddings, dim=0)
                embeddings -= centroid
                embeddings /= torch.norm(embeddings, dim=1, keepdim=True)
            else:
                warnings.warn(
                    "sphereize==True but only a single input sentence was passed."
                    " Inputs will not be sphereized."
                )

        return embeddings.numpy()
