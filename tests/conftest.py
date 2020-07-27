from typing import List

import pytest
from allennlp.common import util as common_util
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import Archive, load_archive
from allennlp.predictors import Predictor

from declutr.encoder import PRETRAINED_MODELS, Encoder
from declutr.predictor import DeCLUTRPredictor


@pytest.fixture(params=["declutr-small", "declutr-base"], scope="module")
def archive(request) -> Archive:
    if request.param in PRETRAINED_MODELS:
        pretrained_model_name_or_path = PRETRAINED_MODELS[request.param]
    common_util.import_module_and_submodules("declutr")
    pretrained_model_name_or_path = cached_path(pretrained_model_name_or_path)
    return load_archive(pretrained_model_name_or_path)


@pytest.fixture(scope="module")
def predictor(archive) -> DeCLUTRPredictor:
    return Predictor.from_archive(archive, predictor_name="declutr")


@pytest.fixture(params=["declutr-small", "declutr-base"], scope="module")
def encoder(request) -> Encoder:
    return Encoder(request.param)


@pytest.fixture(scope="module")
def inputs() -> List[str]:
    # Some random examples taken from https://nlp.stanford.edu/projects/snli/
    return [
        "A man inspects the uniform of a figure in some East Asian country.",
        "The man is sleeping",
        "An older and younger man smiling.",
        "Two men are smiling and laughing at the cats playing on the floor.",
        "A black race car starts up in front of a crowd of people.",
        "A man is driving down a lonely road.",
        "A smiling costumed woman is holding an umbrella.",
        "A happy woman in a fairy costume holds an umbrella.",
    ]
