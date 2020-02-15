"""A simple script which uses pre-trained word embeddings to encode some documents, saving the
resulting emebddings to disk.

Call `python embed_with_word2vec.py --help` for usage instructions.
"""

import json
from pathlib import Path
from typing import List
from typing import Tuple

import fire
from tqdm import tqdm

import numpy as np
import spacy
from gensim.models import KeyedVectors

# This needs to have been downloaded with the `python -m spacy download <model_name>` command
SPACY_MODEL = "en_core_web_sm"


def main(input_file: str, output_file: str, word_vectors: str, binary: bool = False) -> None:

    print('Loading model and tokenizer...', end=' ', flush=True)
    nlp, model = _init_model_and_tokenizer(word_vectors, binary)
    print('Done.')

    texts = Path(input_file).read_text().split('\n')
    embeddings = _embed(texts, nlp, model)

    print(f'Saving document embeddings to {output_file}...', end=' ', flush=True)
    _save_embeddings_to_disk(output_file, embeddings)
    print('Done.')


def _init_model_and_tokenizer(word_vectors: str, binary: bool) -> Tuple:

    nlp = spacy.load(SPACY_MODEL, disable=['ner'])
    model = KeyedVectors.load_word2vec_format(word_vectors, binary=binary)

    return nlp, model


def _embed(texts: List[str], nlp: spacy.lang, model) -> List[float]:
    """Using `model` and its corresponding `tokenizer`, encodes each instance in `text` and returns
    the resulting list of embeddings.

    Args:
        texts (List[str]): A list containing the text instances to embed.
        tokenizer (spacy.lang): The spaCy model which will be used for tokenization.
        model ([type]): An initialized model from the Transformers library.

    Returns:
        List[float]: A list containing the embeddings for each instance in `text`.
    """

    # TODO (John): Empty lines lead to empty lists being appended. Figure out how to filter.
    doc_embeddings = []
    docs = nlp.pipe(texts, n_process=-1)
    for doc in tqdm(docs, total=len(texts), desc='Embedding documents', dynamic_ncols=True):
        word_embeddings = []
        for token in doc:
            # TODO (John): Smarter way to do this? If I don't find a word I lowercase it and try
            # again
            if token.text not in model.vocab:
                if token.text.lower() in model.vocab:
                    word_embeddings.append(model[token.text.lower()])
            else:
                word_embeddings.append(model[token.text])

        # HACK (John): An empty string will lead to a divide by 0 error. Instead, set the embedding
        # to a random vector
        if len(word_embeddings) > 0:
            word_embeddings = np.array(word_embeddings)
        else:
            word_embeddings = np.random.rand(1, model.vector_size)

        doc_embeddings.append(np.mean(word_embeddings, axis=0).tolist())

    return doc_embeddings


def _save_embeddings_to_disk(output_file: str, embeddings: List[float]) -> None:
    """Saves `embeddings` to a JSON lines formatted file `output_file`. Each line looks like:

        {"doc_embeddings": [-0.4989708960056305, ..., 0.19127938151359558]}

    Args:
        output_file (str): Path to save the embeddings.
        embeddings (List[float]): A list of lists, containing one embedding per document.
    """
    output_file = Path(output_file)
    output_file.parents[0].mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        # Format the embeddings in JSON lines format
        for embedding in embeddings:
            json.dump({'doc_embeddings': embedding}, f)
            f.write('\n')


if __name__ == '__main__':
    fire.Fire(main)
