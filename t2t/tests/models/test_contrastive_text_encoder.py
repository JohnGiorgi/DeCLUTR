from pathlib import Path

import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.testing import ModelTestCase
from allennlp.models import Model


class TestContrastiveTextEncoder(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.FIXTURES_ROOT = Path(
            "t2t/tests/fixtures"  # We need to override the path set by AllenNLP
        )
        self.set_up_model(
            self.FIXTURES_ROOT / "contrastive_text_encoder" / "experiment.jsonnet",
            self.FIXTURES_ROOT / "data" / "wikitext-103" / "train.txt",
        )

    def test_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "loss" in output_dict.keys()
        # Embeddings are not added to the output dict when training
        assert "embeddings" not in output_dict.keys()

    def test_forward_pass_contrastive_only_runs_correctly(self):
        self.set_up_model(
            self.FIXTURES_ROOT / "contrastive_text_encoder" / "experiment_contrastive_only.jsonnet",
            self.FIXTURES_ROOT / "data" / "wikitext-103" / "train.txt",
        )
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "loss" in output_dict.keys()
        # Embeddings are not added to the output dict when training
        assert "embeddings" not in output_dict.keys()

    def test_forward_pass_mlm_only_runs_correctly(self):
        self.set_up_model(
            self.FIXTURES_ROOT / "contrastive_text_encoder" / "experiment_mlm_only.jsonnet",
            self.FIXTURES_ROOT / "data" / "wikitext-103" / "train.txt",
        )
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "loss" in output_dict.keys()
        # Embeddings are not added to the output dict when training
        assert "embeddings" not in output_dict.keys()

    def test_no_loss_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        params["model"]["loss"] = None
        params["model"]["text_field_embedder"]["token_embedders"]["tokens"][
            "masked_language_modeling"
        ] = False
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get("model"))
