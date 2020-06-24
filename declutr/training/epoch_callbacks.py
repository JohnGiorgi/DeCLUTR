from pathlib import Path
from typing import Any, Dict

from allennlp.training import EpochCallback, GradientDescentTrainer


@EpochCallback.register("hf_save_pretrained")
class HFSavePretrainedCallback(EpochCallback):
    """This callback will call
    `trainer.model.transformer_model.save_pretrained(trainer.serialization_dir)` at the end of
    every epoch. This is useful to save the weights/tokenizer/config of a HuggingFace Transformers
    model (encapsulated by a `PretrainedTransformerEmbedder` object) for uploading to
    https://huggingface.co/models.
    """

    def __call__(
        self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int
    ) -> None:
        if not trainer._serialization_dir:
            raise ValueError(
                "trainer._serialization_dir must be specified to use this EpochCallback."
            )

        hf_serialization_dir = Path(trainer._serialization_dir) / "hf_transformers"
        hf_serialization_dir.mkdir(parents=True, exist_ok=True)

        trainer.model._text_field_embedder._token_embedders[
            "tokens"
        ].transformer_model.save_pretrained(hf_serialization_dir)
        trainer.model._text_field_embedder._token_embedders[
            "tokens"
        ].tokenizer.tokenizer.save_pretrained(hf_serialization_dir)
