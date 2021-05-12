#!/usr/bin/env python3
import io
import re
import zipfile
from pathlib import Path
from typing import List, Optional

import requests
import typer
from declutr.common.util import sanitize_text

WIKITEXT_103_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
SAVING = "\U0001F4BE"
DOWNLOAD = "\U00002B07"


def _write_output_to_disk(text: List[str], output_filepath: Path) -> None:
    """Writes a list of documents, `text`, to the file `output_filepath`, one document per line."""
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
        bold=True,
    )


def main(
    output_filepath: Path,
    segment_sentences: bool = False,
    lowercase: bool = False,
    min_length: Optional[int] = None,
    max_instances: Optional[int] = None,
    pretrained_model_name_or_path: Optional[str] = None,
) -> None:
    """Downloads and lightly preprocesses WikiText-103. If `min_length is not None`, only documents
    with at least this many tokens are retained. If `pretrained_model_name_or_path` is not None, the
    tokenizer will be loaded as `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)`
    using the HuggingFace Transformers library. Otherwise `str.split()` is used. This argument has
    no effect if `min-length is None`. If `segment_sentences` is provided, individual sentences
    will be returned instead of documents. You must have the `"en_core_web_sm"` spacy model
    installed to segment sentences.
    """
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

    # Setup spacy lang object if we are segmenting sentences
    if segment_sentences:
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # Download WikiText-103
    r = requests.get(WIKITEXT_103_URL, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    partition_filenames = z.namelist()[1:]
    typer.secho(f"{DOWNLOAD} Downloaded WikiText-103", bold=True)

    preprocessed_documents: List[str] = []
    for filename in partition_filenames:
        text = z.open(filename).read().decode("utf-8")

        # Strip out subtitles and split the text into documents
        no_subtitles = re.sub(r"(=\s){2,5}.*(=\s){2,5}", "", text)
        documents = re.split(r"=\s.*\s=", no_subtitles)

        if segment_sentences:
            documents = (sent.text for doc in documents for sent in nlp(doc).sents)  # type: ignore

        with typer.progressbar(
            documents, length=max_instances, label=typer.style("Preprocessing text", bold=True)
        ) as progress:
            for doc in progress:
                doc = sanitize_text(doc, lowercase=lowercase)
                if not doc:
                    continue

                # Retain documents if the length of their shortest document is
                # equal to or greater than the minimum specified length
                if tokenizer is not None:
                    num_tokens = len(tokenizer(doc))
                    if min_length and num_tokens < min_length:
                        continue

                if max_instances and len(preprocessed_documents) >= max_instances:
                    break
                preprocessed_documents.append(doc)
                progress.update(1)

    _write_output_to_disk(preprocessed_documents, output_filepath)


if __name__ == "__main__":
    typer.run(main)
