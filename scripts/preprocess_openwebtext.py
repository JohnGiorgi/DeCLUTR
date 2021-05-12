#!/usr/bin/env python3
import shutil
import tarfile
from pathlib import Path
from typing import Optional

import typer
from declutr.common.util import sanitize_text
from more_itertools import chunked

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
WARNING = "\U000026A0"
SAVING = "\U0001F4BE"
MINING = "\U000026CF"


def main(
    openwebtext_path: Path = typer.Argument(..., help="Path to a OpenWebText dump."),
    output_filepath: Path = typer.Argument(..., help="Filepath to save the preprocessed text"),
    min_length: Optional[int] = typer.Option(
        None, help="Minimum token length of documents to retain"
    ),
    lowercase: bool = typer.Option(True, help="Whether text should be lowercased"),
    max_documents: Optional[int] = typer.Option(
        None,
        help="Maximum number of documents to retain. Because of batching, this won't be exact.",
    ),
    pretrained_model_name_or_path: Optional[str] = typer.Option(
        None,
        help=(
            "Name of the HuggingFace Tokenizer to use when determining the token length of a"
            "document. Has no effect if min-length is None"
        ),
    ),
) -> None:
    """Lightly preprocesses an OpenWebText dump obtained from
    https://skylion007.github.io/OpenWebTextCorpus/. If `min-length is not None`, only documents
    with at least this many tokens are retained. If `pretrained_model_name_or_path` is not None,
    the tokenizer will be loaded as `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)`
    using the HuggingFace Transformers library. Otherwise `str.split()` is used. This argument has
    no effect if `min-length is None`.
    """
    openwebtext_path = Path(openwebtext_path)
    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)

    # Setup the pre-trained tokenizer, if specified
    if min_length is not None:
        if pretrained_model_name_or_path is not None:
            # Import transformers here to prevent ImportError errors if the
            # user doesn't want to use it.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=True)
        else:
            tokenizer = lambda x: x.split()  # noqa
    else:
        tokenizer = None

    processed_docs = 0
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
        for tar_filepath in openwebtext_path.iterdir():
            # Didn't bother debugging as it only happens for a tiny number (1-2) of tar archives.
            # Instead, catch the error and report to the user at the end how many we skipped.
            untared_filepath = Path(tar_filepath.stem)
            try:
                with tarfile.open(tar_filepath) as tf:
                    tf.extractall(untared_filepath)
            except (tarfile.ReadError, IsADirectoryError):
                skipped_files += 1
                continue

            for text_filepaths in chunked(untared_filepath.iterdir(), 128):
                docs = []
                for fp in text_filepaths:
                    # Some very minimal preprocessing to remove extra whitespace, newlines and tabs.
                    doc = sanitize_text(fp.read_text(), lowercase=lowercase)
                    # We add a space in front of the text in order to achieve consistant tokenization
                    # with certain tokenizers, e.g. the BPE tokenizer used by RoBERTa, GPT and others.
                    # See: https://github.com/huggingface/transformers/issues/1196
                    doc = f"{ doc.lstrip()}"
                    docs.append(doc)

                if tokenizer is not None:
                    if pretrained_model_name_or_path:
                        lengths = tokenizer(
                            docs, add_special_tokens=False, truncation=False, return_length=True
                        ).length
                    else:
                        lengths = [len(tokenizer(doc)) for doc in docs]
                    docs = [doc for doc, length in zip(docs, lengths) if length > min_length]

                with open(output_filepath, "a") as f:
                    f.write("\n".join(docs).strip() + "\n")

                if max_documents:
                    progress.update(len(docs))
                    processed_docs += len(docs)
                    if processed_docs >= max_documents:
                        break

            # We are using a for-else trick here, see: https://stackoverflow.com/a/3150107/6578628
            else:
                if max_documents is None:
                    progress.update(1)
                shutil.rmtree(untared_filepath)
                # Continue if the inner loop wasn't broken.
                continue
            shutil.rmtree(untared_filepath)
            # Inner loop was broken, break the outer.
            break

    if skipped_files:
        typer.secho(
            f"{WARNING} {skipped_files} tar files were skipped because they couldn't be extracted.",
            fg=typer.colors.YELLOW,
            bold=True,
        )


if __name__ == "__main__":
    typer.run(main)
