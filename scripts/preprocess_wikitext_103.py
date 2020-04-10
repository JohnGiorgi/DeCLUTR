import io
import re
import zipfile
from pathlib import Path
from typing import List, Optional

import requests
import typer
from transformers import AutoTokenizer

WIKITEXT_103_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
SAVING = "\U0001F4BE"


def _sanitize(text: str) -> str:
    """Cleans text by dropping non-ASCII characters and removing whitespace, newlines and tabs.
    """
    # Convert to ASCII, dropping anything that can't be converted
    text = text.encode("ascii", "ignore").decode("utf-8")
    # Remove whitespace, newlines and tabs
    text = " ".join(text.strip().split())
    return text


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
    """Downloads and lightly pre-processes WikiText-103. If `min_lengthgth` is not None, only documents with at
    least this many tokens are retained. If `pretrained_model_name_or_path` is not None, the tokenizer will be
    loaded as `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` using the HuggingfFace Transformers
    library. Otherwise `.split()` is used. This argument has no effect if `min_lengthgth is None`.
    """
    # Setup the pre-trained tokenizer, if specified
    if pretrained_model_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    else:
        tokenizer = None

    # Download WikiText-103
    r = requests.get(WIKITEXT_103_URL, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    partition_filenames = z.namelist()[1:]

    preprocessed_text = []
    for filename in partition_filenames:
        text = z.open(filename).read().decode("utf-8")

        # Strip out subtitles and split the text into documents
        no_subtitles = re.sub(r"(=\s){2,5}.*(=\s){2,5}", "", text)
        documents = re.split(r"=\s.*\s=", no_subtitles)

        with typer.progressbar(
            documents, label=typer.style(f"Processing {filename}", bold=True)
        ) as progress:
            for doc in progress:
                doc = _sanitize(doc)

                if not doc:
                    continue

                # Retain the document if min_length is None, or min_length is not None and this document contains
                # greater than or equal to min_length tokens.
                if min_length is not None:
                    if tokenizer is not None:
                        tokens = tokenizer.tokenize(doc)
                    else:
                        tokens = doc.split()
                    if len(tokens) >= min_length:
                        preprocessed_text.append(doc)
                else:
                    preprocessed_text.append(doc)

    _write_output_to_disk(preprocessed_text, output_filepath)


if __name__ == "__main__":
    typer.run(main)
