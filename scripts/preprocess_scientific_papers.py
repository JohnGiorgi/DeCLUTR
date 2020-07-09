#!/usr/bin/env python3
from pathlib import Path
import nlp
from typing import List, Optional
import itertools

import typer

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
WARNING = "\U000026A0"
SAVING = "\U0001F4BE"
MINING = "\U000026CF"


def _sanitize(text: str) -> str:
    """Cleans text by removing whitespace, newlines and tabs.
    """
    return " ".join(text.strip().split())


def _write_output_to_disk(text: List[str], output_filepath: str) -> None:
    """Writes a list of documents, `text`, to the file `output_filepath`, one document per line.
    """
    # Create the directory path if it doesn't exist
    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)

    with open(output_filepath, "w") as f:
        # TODO (John): In the future, it might make sense to both batch and shard:
        # 1) Batch, meaning write batches of documents to a file as opposed to 1 at a time
        # 2) Shard, meaning break a file up into shard_size // len(text) files, and return a
        #    directory instead. Loading a dataset like this is supported in AllenNLP (see:
        #    https://docs.allennlp.org/master/api/data/dataset_readers/sharded_dataset_reader/)
        with typer.progressbar(text, label="Writing to disk") as progress:
            for doc in progress:
                f.write(doc.strip() + "\n")
    typer.secho(
        f"{SAVING} {len(text)} preprocessed documents saved to: {output_filepath}",
        fg=typer.colors.WHITE,
        bold=True,
    )


def main(
    output_filepath: str,
    min_length: Optional[int] = None,
    max_documents: Optional[int] = None,
    pretrained_model_name_or_path: Optional[str] = None,
) -> None:
    """Lightly pre-processes an OpenWebText dump obtained from
    https://skylion007.github.io/OpenWebTextCorpus/. If `min_length` is not None, only documents
    with at least this many tokens are retained. If `pretrained_model_name_or_path` is not None, the
    tokenizer will be loaded as `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)`
    using the HuggingFace Transformers library. Otherwise `str.split()` is used. This argument has
    no effect if `min-length is None`.
    """
    # Collect the raw text from the "scientific_papers" dataset.
    pubmed = nlp.load_dataset("scientific_papers", "pubmed")
    arxiv = nlp.load_dataset("scientific_papers", "arxiv")
    # Create a genrator over both datasets to avoid storing things in memory.
    pubmed_text = (article["article"] for partition in pubmed.values() for article in partition)
    arxiv_text = (article["article"] for partition in arxiv.values() for article in partition)
    scientific_text = itertools.chain(pubmed_text, arxiv_text)

    # Setup the pre-trained tokenizer, if specified.
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

    documents = []
    typer.secho(
        (
            f' {MINING} Scraping {max_documents or "all"} documents'
            f' {f"with a minimum token length of {min_length}" if min_length else ""}'
        ),
        fg=typer.colors.WHITE,
        bold=True,
    )

    with typer.progressbar(scientific_text, label="Preprocessing text") as progress:
        for doc in progress:
            doc = _sanitize(doc)
            if not doc:
                continue

            # Retain documents if the length of their shortest document is
            # equal to or greater than the minimum specified length
            if tokenizer is not None:
                num_tokens = len(tokenizer(doc))
                if num_tokens < min_length:
                    continue
            documents.append(doc)
    _write_output_to_disk(documents, output_filepath)


if __name__ == "__main__":
    typer.run(main)
