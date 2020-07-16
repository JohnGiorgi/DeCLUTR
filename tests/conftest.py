import pytest
from allennlp.common import util as common_util
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from declutr.encoder import PRETRAINED_MODELS, Encoder


@pytest.fixture(params=["declutr-small"])
def encoder(request):
    return Encoder(request.param)


@pytest.fixture(params=["declutr-small"])
def archive(request):
    if request.param in PRETRAINED_MODELS:
        pretrained_model_name_or_path = PRETRAINED_MODELS[request.param]
    common_util.import_module_and_submodules("declutr")
    pretrained_model_name_or_path = cached_path(pretrained_model_name_or_path)
    return load_archive(pretrained_model_name_or_path)


@pytest.fixture
def predictor(archive):
    return Predictor.from_archive(archive, predictor_name="declutr")
