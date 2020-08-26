from pathlib import Path

import typer
from allennlp.common import util as common_util
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
SAVING = "\U0001F4BE"
HUGGING_FACE = "\U0001F917"


def main(archive_file: str, save_directory: str) -> None:
    """Saves the model and tokenizer from an AllenNLP `archive_file` path pointing to a trained
    DeCLUTR model to a format that can be used with HuggingFace Transformers at `save_directory`."""
    save_directory = Path(save_directory)
    save_directory.parents[0].mkdir(parents=True, exist_ok=True)

    common_util.import_module_and_submodules("declutr")
    # cuda_device -1 places the model onto the CPU before saving. This avoids issues with
    # distributed models.
    overrides = "{'trainer.cuda_device': -1}"
    archive = load_archive(archive_file, overrides=overrides)
    predictor = Predictor.from_archive(archive, predictor_name="declutr")

    token_embedder = predictor._model._text_field_embedder._token_embedders["tokens"]
    model = token_embedder.transformer_model
    tokenizer = token_embedder.tokenizer

    # Casting as a string to avoid this error: https://github.com/huggingface/transformers/pull/4650
    # Can be removed after PR is merged and Transformers is updated.
    model.save_pretrained(str(save_directory))
    tokenizer.save_pretrained(str(save_directory))

    typer.secho(
        (
            f"{SAVING} {HUGGING_FACE} Transformers compatible model saved to: {save_directory}."
            " See https://huggingface.co/transformers/model_sharing.html for instructions on"
            f" hosting the model with {HUGGING_FACE} Transformers."
        ),
        bold=True,
    )


if __name__ == "__main__":
    typer.run(main)
