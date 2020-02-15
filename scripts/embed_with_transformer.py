"""A simple script which uses a pre-trained transformer to encode some documents, saving the
resulting emebddings to disk. We wrap the Transformers library
(https://github.com/huggingface/transformers), so you can use any of the pre-trained models listed
here: https://huggingface.co/transformers/pretrained_models.html.

Call `python embed_with_transformer.py --help` for usage instructions.
"""

import json
from pathlib import Path
from typing import List
from typing import Tuple

import fire
import torch
from tqdm import trange
from transformers import AutoModel
from transformers import AutoTokenizer


def main(
    pretrained_model_name_or_path: str,
    input_file: str,
    output_file: str,
    batch_size: int = 16,
    disable_cuda: bool = False,
    opt_level: str = None,
) -> None:

    device = _get_device(disable_cuda)

    print('Loading model and tokenizer...', end=' ', flush=True)
    tokenizer, model = _init_model_and_tokenizer(pretrained_model_name_or_path, device, opt_level)
    print('Done.')

    text = Path(input_file).read_text().split('\n')
    embeddings = _embed(text, tokenizer, model, batch_size, device)

    print(f'Saving document embeddings to {output_file}...', end=' ', flush=True)
    _save_embeddings_to_disk(output_file, embeddings)
    print('Done.')


def _get_device(disable_cuda):
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def _init_model_and_tokenizer(pretrained_model_name_or_path: str, device: torch.device, opt_level: str) -> Tuple:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = AutoModel.from_pretrained(pretrained_model_name_or_path).to(device)

    if opt_level is not None:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model = amp.initialize(model, opt_level=opt_level)

    return tokenizer, model


def _embed(texts: List[str], tokenizer, model, batch_size: int, device: torch.device) -> List[float]:
    """Using `model` and its corresponding `tokenizer`, encodes each instance in `text` and returns
    the resulting list of embeddings.

    Args:
        text (List[str]): A list containing the text instances to embed.
        tokenizer ([type]): An initialized tokenizer from the Transformers library.
        model ([type]): An initialized model from the Transformers library.
        batch_size (int): Batch size to use when embedding instances in `text`.
        device (torch.device): A torch.device object specifying which device (cpu or gpu) to use.

    Returns:
        List[float]: A list containing the embeddings for each instance in `text`.
    """
    input_ids = torch.tensor([tokenizer.encode(text, max_length=512, pad_to_max_length=True) for text in texts])
    attention_masks = torch.where(
        input_ids == tokenizer.pad_token_id,
        torch.zeros_like(input_ids),
        torch.ones_like(input_ids)
    )

    doc_embeddings = []
    for i in trange(0, input_ids.size(0), batch_size, desc='Embedding documents', dynamic_ncols=True):
        batch = {
            "input_ids": input_ids[i:i+batch_size].to(device),
            "attention_mask": attention_masks[i:i+batch_size].to(device)
        }

        word_embeddings, _ = model(**batch)
        doc_embeddings.extend(
            (torch.sum(word_embeddings * batch['attention_mask'].unsqueeze(-1), dim=1) /
             torch.clamp(torch.sum(batch['attention_mask'], dim=1, keepdims=True), min=1e-9)).tolist()
        )

        del batch

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
