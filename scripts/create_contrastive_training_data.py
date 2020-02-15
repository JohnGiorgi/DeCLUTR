"""A simple script which creates data for a model training against a contrastive objective.

Call `python create_contrastive_training_data.py --help` for usage instructions.
"""

import csv
from pathlib import Path
from typing import List

import fire
import spacy
import spacy.lang
from tqdm import tqdm


def main(input_file: str, output_file: str, spacy_model: str = "en_core_web_sm") -> None:
    """For some given input documents, generates the data neccecary for a self-supervised contrastive loss.

    This script takes as input a text file containing one document per line (`input_file`), and returns a tsv file
    containing pairs of "anchors" and "positives". These examples can be used to train a model with a contrastive
    objective. Specifically, the anchors are the input documents, and the positives are a single sentence from each
    input document.

    Args:
        input_file (str): Path to a text file containing one document per line.
        output_file (str): Path to the tsv file where the generated (anchor, positive) pairs will be saved.
        spacy_model (str, optional): SpaCy model to load (see https://spacy.io/models). Must be installed with
            python -m spacy download <spacy_model>`. Defaults to `"en_core_web_sm"`.
    """

    print('Loading tokenizer...', end=' ', flush=True)
    nlp = spacy.load(spacy_model, disable=['ner'])
    print('Done.')

    anchors = Path(input_file).read_text().split('\n')
    positives = _generate_positives(anchors, nlp)

    print(f'Saving generated training data to {output_file}...', end=' ', flush=True)
    train_data = zip(anchors, positives)
    _save_train_data_to_disk(output_file, train_data)
    print('Done.')


def _generate_positives(anchors: List[str], nlp: spacy.lang):
    """For each anchor in `anchors`, extracts a single sentence which will serve as the positive example.

    Args:
        anchors (List[str]): A list of strings, containing the input documents which serves as anchors.
        nlp (spacy.lang): A loaded spacy language model.

    Returns:
        List: A list of positive examples, one per anchor in `anchors`.
    """
    dropped = 0
    positives = []

    docs = nlp.pipe(anchors, n_process=-1)
    for doc in tqdm(docs, total=len(anchors), desc='Generating positive examples', dynamic_ncols=True):
        # TODO (John): We don't want especially short sentences as they contain little information. As a temporary
        # hack, take the longest sentence from each anchor.
        sents = list(doc.sents)

        if len(sents) == 0:
            dropped += 1
            continue

        positives.append(
            sorted(sents, key=len)[-1].text
        )

    if dropped:
        print(f"Dropped {dropped}/{len(anchors)} ({dropped/len(anchors):.2%}) anchors with no sentences.")

    return positives


def _save_train_data_to_disk(output_file: str, train_data: zip) -> None:
    """Saves the zipped lists, `train_data` to a tsv file, `output_file`.

    Args:
        output_file (str): Path to the tsv file to save `train_data`.
        train_data (zip): A `zip` object containing two iterables, representing the positive, anchor pairs.
    """
    output_file = Path(output_file)
    output_file.parents[0].mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(train_data)


if __name__ == "__main__":
    fire.Fire(main)
