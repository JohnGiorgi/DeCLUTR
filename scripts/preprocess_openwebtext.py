import shutil
import tarfile
from pathlib import Path
from typing import List, Optional

import typer
from transformers import AutoTokenizer

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
WARNING = "\U000026A0"
SAVING = "\U0001F4BE"


def _write_output_to_disk(text: List[str], output_filepath: str) -> None:
    """Writes a list of documents, `text`, to the file `output_filepath`, one document per line.
    """
    # Create the directory path if it doesn't exist
    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)

    with open(output_filepath, "w") as f:
        f.write("\n".join(text))
    typer.secho(
        f"{SAVING} Preprocessed OpenWebText saved to: {output_filepath}",
        fg=typer.colors.WHITE,
        bold=True,
    )


def main(
    openwebtext_path: str,
    output_filepath: str,
    min_length: Optional[int] = None,
    max_documents: Optional[int] = None,
    pretrained_model_name_or_path: Optional[str] = None,
) -> None:
    """Lightly pre-processes an OpenWebText dump obtained from https://skylion007.github.io/OpenWebTextCorpus/.
    If `min_length` is not None, only documents whose shortest paragraph have at least this many tokens are
    retained. If `pretrained_model_name_or_path` is not None, the tokenizer will be loaded as
    `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` using the HuggingFace's Transformers library.
    Otherwise `.split()` is used. This argument has no effect if `min_length is None`.
    """
    openwebtext_path = Path(openwebtext_path)

    # Setup the pre-trained tokenizer, if specified
    if min_length is not None:
        if pretrained_model_name_or_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path).tokenize
        else:
            tokenizer = lambda x: x.split()  # noqa
    else:
        tokenizer = None

    early_exit = False
    documents = []
    skipped_files = 0
    with typer.progressbar(length=max_documents, label="Processing") as progress:
        for i, tar_filepath in enumerate(openwebtext_path.iterdir()):
            untared_filepath = Path(tar_filepath.stem)
            try:
                with tarfile.open(tar_filepath) as f:
                    f.extractall(untared_filepath)
            except (tarfile.ReadError, IsADirectoryError):
                skipped_files += 1
                continue

            for text_filepath in untared_filepath.iterdir():
                # Drop empty lines
                paragraphs = [
                    paragraph for paragraph in text_filepath.read_text().split("\n\n") if paragraph
                ]

                # We need at least two paragraphs to define a pair for the contrastive objective
                if len(paragraphs) < 2:
                    continue

                # Retain documents if the length of their shortest document is equal to or greater than
                # the minimum specified length
                if tokenizer is not None:
                    paragraphs_len = [len(tokenizer(paragraph)) for paragraph in paragraphs]
                    if min(paragraphs_len) >= min_length:
                        documents.append("\t".join(paragraphs))
                        progress.update(1)
                else:
                    documents.append("\t".join(paragraphs))
                    progress.update(1)
                if max_documents and len(documents) >= max_documents:
                    early_exit = True
                    break
            shutil.rmtree(untared_filepath)
            if early_exit:
                break

    if skipped_files > 0:
        typer.secho(
            f"{WARNING} {skipped_files} tar files were skipped because they couldn't be extracted.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    _write_output_to_disk(documents, output_filepath)


if __name__ == "__main__":
    typer.run(main)
