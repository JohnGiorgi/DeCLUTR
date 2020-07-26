from pathlib import Path

import pytest

from allennlp.common.params import Params
from allennlp.common.testing import ModelTestCase
from allennlp.models import Model


class TestDeCLUTR(ModelTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        # We need to override the path set by AllenNLP
        self.FIXTURES_ROOT = Path("tests/fixtures")
        self.set_up_model(
            self.FIXTURES_ROOT / "experiment.jsonnet",
            self.FIXTURES_ROOT / "data" / "openwebtext" / "train.txt",
        )

    def test_forward_pass_runs_correctly(self) -> None:
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "loss" in output_dict.keys()
        # Embeddings are not added to the output dict when training
        assert "embeddings" not in output_dict.keys()

    def test_forward_pass_with_feedforward_runs_correctly(self) -> None:
        self.set_up_model(
            self.FIXTURES_ROOT / "experiment_feedforward.jsonnet",
            self.FIXTURES_ROOT / "data" / "openwebtext" / "train.txt",
        )
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "loss" in output_dict.keys()
        # Embeddings are not added to the output dict when training
        assert "embeddings" not in output_dict.keys()

    def test_forward_pass_contrastive_only_runs_correctly(self) -> None:
        self.set_up_model(
            self.FIXTURES_ROOT / "experiment_contrastive_only.jsonnet",
            self.FIXTURES_ROOT / "data" / "openwebtext" / "train.txt",
        )
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "loss" in output_dict.keys()
        # Embeddings are not added to the output dict when training
        assert "embeddings" not in output_dict.keys()

    def test_forward_pass_mlm_only_runs_correctly(self) -> None:
        self.set_up_model(
            self.FIXTURES_ROOT / "experiment_mlm_only.jsonnet",
            self.FIXTURES_ROOT / "data" / "openwebtext" / "train.txt",
        )
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "loss" in output_dict.keys()
        # Embeddings are not added to the output dict when training
        assert "embeddings" not in output_dict.keys()

    def test_forward_pass_scalar_mix_runs_correctly(self) -> None:
        self.set_up_model(
            self.FIXTURES_ROOT / "experiment_scalar_mix.jsonnet",
            self.FIXTURES_ROOT / "data" / "openwebtext" / "train.txt",
        )
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "loss" in output_dict.keys()
        # Embeddings are not added to the output dict when training
        assert "embeddings" not in output_dict.keys()

    def test_no_loss_throws_configuration_error(self) -> None:
        params = Params.from_file(self.param_file)
        params["model"]["loss"] = None
        params["model"]["text_field_embedder"]["token_embedders"]["tokens"][
            "masked_language_modeling"
        ] = False
        with pytest.raises(ValueError):
            Model.from_params(vocab=self.vocab, params=params.get("model"))

    @pytest.mark.skip(reason="failing for upstream reasons")
    def test_can_train_save_and_load(self) -> None:
        self.ensure_model_can_train_save_and_load(self.param_file)
