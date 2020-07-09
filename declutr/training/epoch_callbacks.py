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

    # Parameters

    model_name : `str`, optional (default=`'hf_transformer'`)
        Name of the folder to save the HuggingFace Transformers model. The full path is given
        by: "<trainer._serialization_dir>/model_name".
    """

    def __init__(self, model_name: str = "hf_transformer"):
        self._model_name = model_name

    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        if not trainer._serialization_dir:
            raise ValueError(
                "trainer._serialization_dir must be specified to use this EpochCallback."
            )

        hf_serialization_dir = Path(trainer._serialization_dir) / self._model_name
        hf_serialization_dir.mkdir(parents=True, exist_ok=True)

        trainer.model._text_field_embedder._token_embedders[
            "tokens"
        ].transformer_model.save_pretrained(hf_serialization_dir)
        # Casting as a string to avoid this error: https://github.com/huggingface/transformers/pull/4650
        # Can be removed after PR is merged and Transformers is updated.
        trainer.model._text_field_embedder._token_embedders["tokens"].tokenizer.save_pretrained(
            str(hf_serialization_dir)
        )
