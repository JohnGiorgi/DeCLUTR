#!/usr/bin/env python3
from pathlib import Path
from typing import Iterable, List, Optional

import typer
from snapy import LSH, MinHash

SEED = 3
# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
WARNING = "\U000026A0"
IO = "\U0001F4BE"
SEARCH = "\U0001F50E"
HASHING = "\U00000023\U0000FE0F\U000020E3"


def _normalize(text: str, max_length: Optional[int] = None) -> str:
    """Normalizes `text` by lowercasing, removing whitespace, newlines and tabs and (optionally)
    truncating it to `max_length` whitespace tokens.
    """
    normalized_text = text.lower().strip().split()
    max_length = max_length or len(normalized_text)
    return " ".join(normalized_text[:max_length])


def _yield_text(input_filepath: str) -> Iterable[str]:
    with open(input_filepath, "r") as f:
        for text in f:
            yield text


def _create_lsh(
    text: List[str],
    labels: List[int],
    n_gram: int,
    n_permutations: int,
    hash_bits: int,
    no_of_bands: int,
) -> LSH:
    """Returns a `snapy.lsh.LSH` object constructed from `text` to detect near duplicate texts.
    """

    minhash = MinHash(
        text, n_gram=n_gram, permutations=n_permutations, hash_bits=hash_bits, seed=SEED
    )
    lsh = LSH(minhash, labels, no_of_bands=no_of_bands)

    typer.secho(
        f"{HASHING}  Hashed the normalized text using Locality-Sensitive Hashing (LSH).", bold=True,
    )

    return lsh


def _get_duplicate_ids(text: List[str], lsh: LSH, min_jaccard: float) -> Iterable[str]:
    """Uses the given `lsh` object to find near duplicate text in `text`. Returns a list of
    indices into `text` which point to duplicate texts.
    """
    duplicate_ids = set()
    adjacency_list = lsh.adjacency_list(min_jaccard=min_jaccard)
    with typer.progressbar(adjacency_list.items(), label="Deduplicating text") as progress:
        for query_id, similar_ids in progress:
            # If query_id exists in duplicate_ids, we have already accounted for it.
            if query_id in duplicate_ids:
                continue
            duplicate_ids.update(similar_ids)
    typer.secho(
        f"{SEARCH} Found a total of {len(duplicate_ids)} duplicate texts.", bold=True,
    )
    return list(duplicate_ids)


def _write_output_to_disk(text: List[str], output_filepath: str, overwrite: bool = False) -> None:
    """Writes a list of documents, `text`, to the file `output_filepath`, one document per line.
    """
    # Create the directory path if it doesn't exist
    output_filepath: Path = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)

    num_docs = 0
    with open(output_filepath, "w") as f:
        # TODO (John): In the future, it might make sense to both batch and shard:
        # 1) Batch, meaning write batches of documents to a file as opposed to 1 at a time
        # 2) Shard, meaning break a file up into shard_size // len(text) files, and return a
        #    directory instead. Loading a dataset like this is supported in AllenNLP (see:
        #    https://docs.allennlp.org/master/api/data/dataset_readers/sharded_dataset_reader/)
        with typer.progressbar(text, label="Writing to disk") as progress:
            for doc in progress:
                f.write(doc.strip() + "\n")
                num_docs += 1
    typer.secho(
        f'{IO} {num_docs} unique texts saved to "{output_filepath}"', bold=True,
    )


def main(
    input_filepath: str,
    output_filepath: Optional[str] = None,
    overwrite: bool = False,
    max_length: Optional[int] = None,
    n_gram: int = 9,
    n_permutations: int = 100,
    hash_bits: int = 64,
    no_of_bands: int = 50,
    min_jaccard: float = 0.5,
) -> None:

    # Default to overwriting the input file (will throw an error if --overwrite is not supplied).
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath) if output_filepath is not None else Path(input_filepath)
    if (input_filepath == output_filepath or output_filepath.exists()) and not overwrite:
        typer.secho(
            (
                f"{WARNING} {output_filepath} already exists. Please provide the argument"
                " --overwrite if you want to overwrite it."
            ),
            fg=typer.colors.YELLOW,
            bold=True,
            err=True,
        )
        raise typer.Abort()

    # We only maintain a copy of the normalized text in memory. For the raw text, we simply
    # yield it as we need it (trading memory for compute).
    normalized_text = [_normalize(text) for text in _yield_text(input_filepath)]
    labels = list(range(len(normalized_text)))
    typer.secho(
        f'{IO} Loaded & normalized {len(normalized_text)} texts from "{input_filepath}"', bold=True,
    )

    lsh = _create_lsh(
        text=normalized_text,
        labels=labels,
        n_gram=n_gram,
        n_permutations=n_permutations,
        hash_bits=hash_bits,
        no_of_bands=no_of_bands,
    )

    duplicate_ids = _get_duplicate_ids(text=normalized_text, lsh=lsh, min_jaccard=min_jaccard)
    deduplicated_text = (
        text for i, text in enumerate(_yield_text(input_filepath)) if i not in duplicate_ids
    )

    _write_output_to_disk(text=deduplicated_text, output_filepath=output_filepath)


if __name__ == "__main__":
    typer.run(main)
