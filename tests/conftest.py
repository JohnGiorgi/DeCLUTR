from pathlib import Path
from typing import List

import pytest
from allennlp.common import util as common_util
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import Archive, load_archive
from allennlp.predictors import Predictor

from declutr.encoder import PRETRAINED_MODELS, Encoder
from declutr.predictor import DeCLUTRPredictor

# Note: Most of these are scoped as "module" to prevent a warning from hypothesis
# about fixtures being reset between function calls.


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
def inputs_filepath() -> str:
    # Some random examples taken from https://nlp.stanford.edu/projects/snli/
    return "tests/fixtures/data/encoder_inputs.txt"


@pytest.fixture(scope="module")
def inputs(inputs_filepath) -> List[str]:
    return Path(inputs_filepath).read_text().split("\n")
