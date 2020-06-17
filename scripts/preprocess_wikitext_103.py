import io
import re
import zipfile
from pathlib import Path
from typing import List, Optional

import requests
import typer

WIKITEXT_103_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
SAVING = "\U0001F4BE"
DOWNLOAD = "\U00002B07"


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
        f.write("\n".join(text))
    typer.secho(
        f"{SAVING} Preprocessed WikiText-103 saved to: {output_filepath}",
        fg=typer.colors.WHITE,
        bold=True,
    )


def main(
    output_filepath: str,
    min_length: Optional[int] = None,
    pretrained_model_name_or_path: Optional[str] = None,
) -> None:
    """Downloads and lightly preprocesses WikiText-103. If `min_length` is not None, only documents
    with at least this many tokens are retained. If `pretrained_model_name_or_path` is not None, the
    tokenizer will be loaded as `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)`
    using the HuggingFace Transformers library. Otherwise `str.split()` is used. This argument has
    no effect if `min_length is None`.
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

    # Download WikiText-103
    r = requests.get(WIKITEXT_103_URL, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    partition_filenames = z.namelist()[1:]
    typer.secho(f"{DOWNLOAD} Downloaded WikiText-103", fg=typer.colors.WHITE, bold=True)

    preprocessed_documents = []
    for filename in partition_filenames:
        text = z.open(filename).read().decode("utf-8")

        # Strip out subtitles and split the text into documents
        no_subtitles = re.sub(r"(=\s){2,5}.*(=\s){2,5}", "", text)
        documents = re.split(r"=\s.*\s=", no_subtitles)

        with typer.progressbar(
            documents, label=typer.style(f"Preprocessing text", bold=True)
        ) as progress:
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

                preprocessed_documents.append(doc)

    _write_output_to_disk(preprocessed_documents, output_filepath)


if __name__ == "__main__":
    typer.run(main)
