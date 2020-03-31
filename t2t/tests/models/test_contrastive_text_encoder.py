from allennlp.common.testing import ModelTestCase
from pathlib import Path


class TestContrastiveTextEncoder(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.FIXTURES_ROOT = Path(
            "t2t/tests/fixtures"
        )  # We need to override the path set by AllenNLP
        self.set_up_model(
            self.FIXTURES_ROOT / "contrastive_text_encoder" / "contrastive.jsonnet",
            self.FIXTURES_ROOT / "data" / "wikitext-103" / "train.txt",
        )

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert "loss" in output_dict.keys()

    def test_can_train_save_and_load(self):
        self.set_up_model(
            self.FIXTURES_ROOT / "contrastive_text_encoder" / "contrastive.jsonnet",
            self.FIXTURES_ROOT / "data" / "wikitext-103" / "train.txt",
        )
        self.ensure_model_can_train_save_and_load(self.param_file)
