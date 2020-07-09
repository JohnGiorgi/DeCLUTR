#!/usr/bin/env python3
import shutil
import tarfile
from pathlib import Path
from typing import List, Optional

import typer

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
WARNING = "\U000026A0"
SAVING = "\U0001F4BE"
MINING = "\U000026CF"


def _sanitize(text: str, lowercase: bool) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing.
    """
    sanitized_text = " ".join(text.strip().split())
    if lowercase:
        return sanitized_text.lower()
    else:
        return sanitized_text


def _write_output_to_disk(text: List[str], output_filepath: str) -> None:
    """Writes a list of documents, `text`, to the file `output_filepath`, one document per line.
    """
    # Create the directory path if it doesn't exist
    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)

    with open(output_filepath, "w") as f:
        # In the future, it might make sense to both batch and shard:
        # 1) Batch, meaning write batches of documents to a file as opposed to 1 at a time
        # 2) Shard, meaning break a file up into shard_size // len(text) files, and return a
        #    directory instead. Loading a dataset like this is supported in AllenNLP (see:
        #    https://docs.allennlp.org/master/api/data/dataset_readers/sharded_dataset_reader/)
        with typer.progressbar(text, label="Writing to disk") as progress:
            for doc in progress:
                f.write(doc.strip() + "\n")
    typer.secho(
        f"{SAVING} {len(text)} preprocessed documents saved to: {output_filepath}", bold=True,
    )


def main(
    openwebtext_path: str,
    output_filepath: str,
    min_length: Optional[int] = None,
    lowercase: bool = True,
    max_documents: Optional[int] = None,
    pretrained_model_name_or_path: Optional[str] = None,
) -> None:
    """Lightly preprocesses an OpenWebText dump obtained from
    https://skylion007.github.io/OpenWebTextCorpus/. If `min-length is not None`, only documents
    with at least this many tokens are retained. If `pretrained_model_name_or_path` is not None,
    the tokenizer will be loaded as `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)`
    using the HuggingFace Transformers library. Otherwise `str.split()` is used. This argument has
    no effect if `min-length is None`.
    """
    openwebtext_path = Path(openwebtext_path)

    # Setup the pre-trained tokenizer, if specified
    if min_length is not None:
        if pretrained_model_name_or_path is not None:
            # Import transformers here to prevent ImportError errors if the
            # user doesn't want to use it.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path).tokenize
        else:
            tokenizer = lambda x: x.split()  # noqa
    else:
        tokenizer = None

    early_exit = False
    documents = []
    skipped_files = 0
    typer.secho(
        (
            f'{MINING} Scraping {max_documents or "all"} documents'
            f' {f"with a minimum token length of {min_length}" if min_length else ""}'
        ),
        bold=True,
    )

    with typer.progressbar(
        length=max_documents or len(list(openwebtext_path.iterdir())), label="Preprocessing text"
    ) as progress:
        for i, tar_filepath in enumerate(openwebtext_path.iterdir()):
            # Didn't bother debugging as it only happens for a tiny number (1-2) of tar archives.
            # Instead, catch the error and report to the user at the end how many we skipped.
            untared_filepath = Path(tar_filepath.stem)
            try:
                with tarfile.open(tar_filepath) as f:
                    f.extractall(untared_filepath)
            except (tarfile.ReadError, IsADirectoryError):
                skipped_files += 1
                continue

            for text_filepath in untared_filepath.iterdir():
                text = text_filepath.read_text()
                text = _sanitize(text, lowercase=lowercase)
                if not text:
                    continue

                # Retain documents if # of tokens is greater than the minimum specified length
                if tokenizer is not None:
                    num_tokens = len(tokenizer(text))
                    if num_tokens < min_length:
                        continue

                documents.append(text)
                if max_documents:
                    progress.update(1)

                if max_documents and len(documents) == max_documents:
                    early_exit = True
                    break

            shutil.rmtree(untared_filepath)
            if max_documents is None:
                progress.update(1)
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
