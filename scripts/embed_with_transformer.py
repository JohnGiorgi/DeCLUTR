#!/usr/bin/env python
"""A simple script which uses a pre-trained transformer to encode some documents, saving the
resulting emebddings to disk. We wrap the Transformers library
(https://github.com/huggingface/transformers), so you can use any of the pre-trained models listed
here: https://huggingface.co/transformers/pretrained_models.html.

Call `python embed_with_transformer.py --help` for usage instructions.
"""

import json
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import torch
import transformers
import typer
from transformers import AutoModel, AutoTokenizer

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py
WARNING = "\U000026A0"
SUCCESS = "\U00002705"
RUNNING = "\U000023F3"
SAVING = "\U0001F4BE"
FAST = "\U0001F3C3"
SCORE = "\U0001F4CB"


class Pooler(str, Enum):
    mean = "mean"
    cls = "cls"


def main(
    pretrained_model_name_or_path: str,
    input_filepath: str,
    output_filepath: str,
    batch_size: int = 16,
    pooler: Pooler = Pooler.mean,
    disable_cuda: bool = False,
    opt_level: str = None,
) -> None:
    """Uses a pre-trained transformer to encode some documents, saving the resulting emebddings to disk. We wrap
    the Transformers library (https://github.com/huggingface/transformers), so you can use any of the pre-trained
    models listed here: https://huggingface.co/transformers/pretrained_models.html.
    """

    device = _get_device(disable_cuda)

    tokenizer, model = _init_model_and_tokenizer(pretrained_model_name_or_path, device, opt_level)

    text = Path(input_filepath).read_text().split("\n")
    embeddings = _embed(text, tokenizer, model, pooler, batch_size, device)

    _save_embeddings_to_disk(output_filepath, embeddings)


def _get_device(disable_cuda):
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        typer.secho(
            f"{FAST} Using GPU device: {torch.cuda.current_device()}", fg=typer.colors.WHITE, bold=True,
        )
    else:
        device = torch.device("cpu")

    return device


def _init_model_and_tokenizer(
    pretrained_model_name_or_path: str, device: torch.device, opt_level: str
) -> Tuple[transformers.PreTrainedTokenizer, transformers.PreTrainedModel]:
    # Load the Transformers tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    typer.secho(
        f'{SUCCESS} Tokenizer "{pretrained_model_name_or_path}" from Transformers loaded successfully.',
        fg=typer.colors.GREEN,
        bold=True,
    )
    # Load the Transformers model
    model = AutoModel.from_pretrained(pretrained_model_name_or_path).to(device)
    model.eval()
    typer.secho(
        f'{SUCCESS} Model "{pretrained_model_name_or_path}" from Transformers loaded successfully.',
        fg=typer.colors.GREEN,
        bold=True,
    )

    if opt_level is not None:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model = amp.initialize(model, opt_level=opt_level)

    return tokenizer, model


def _embed(
    texts: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    pooler: Pooler,
    batch_size: int,
    device: torch.device,
) -> List[float]:
    """Using `model` and its corresponding `tokenizer`, encodes each instance in `text` and returns
    the resulting list of embeddings.

    Args:
        text (List[str]): A list containing the text instances to embed.
        tokenizer (PreTrainedTokenizer): An initialized tokenizer from the Transformers library.
        model (PreTrainedModel): An initialized model from the Transformers library.
        batch_size (int): Batch size to use when embedding instances in `text`.
        device (torch.device): A torch.device object specifying which device (cpu or gpu) to use.
        pooler (str): Controls the pooling strategy. A choice of (mean, cls).

    Returns:
        List[float]: A list containing the embeddings for each instance in `texts`.
    """
    # To embed with transformers, we (minimally) need the input ids and the attention masks.
    input_ids = torch.as_tensor(
        # TODO (John): Hardcoded this for now because of error with BioBERT
        [tokenizer.encode(text, max_length=512, pad_to_max_length=True) for text in texts],
    )
    attention_mask = torch.where(
        input_ids == tokenizer.pad_token_id, torch.zeros_like(input_ids), torch.ones_like(input_ids)
    )

    embeddings = []
    with typer.progressbar(range(0, input_ids.size(0), batch_size), label="Embedding") as progress:
        for batch_idx in progress:
            batch = {
                "input_ids": input_ids[batch_idx : batch_idx + batch_size].to(device),
                "attention_mask": attention_mask[batch_idx : batch_idx + batch_size, :].to(device),
            }
            with torch.no_grad():
                sequence_output, pooled_output = model(**batch)

            # When mean pooling, we take the average of the token-level embeddings, accounting for pads.
            # Otherwise, we take the pooled output for this specific model, which is typically the linear
            # projection of a special tokens embedding, like [CLS] or <s>, which is prepended to the input during
            # tokenization.
            if pooler.value == "mean":
                pooled_output = torch.sum(
                    sequence_output * batch["attention_mask"].unsqueeze(-1), dim=1
                ) / torch.clamp(torch.sum(batch["attention_mask"], dim=1, keepdims=True), min=1e-9)

            pooled_output = pooled_output.tolist()
            embeddings.extend(pooled_output)

    return embeddings


def _save_embeddings_to_disk(output_filepath: str, embeddings: List[float]) -> None:
    """Saves `embeddings` to a JSON lines formatted file `output_filepath`. Each line looks like:

        {"embeddings": [-0.4989708960056305, ..., 0.19127938151359558]}

    Args:
        output_filepath (str): Path to save the embeddings.
        embeddings (List[float]): A list of lists, containing one embedding per document.
    """
    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)

    with open(output_filepath, "w") as f:
        # Format the embeddings in JSON lines format
        for embedding in embeddings:
            json.dump({"embeddings": embedding}, f)
            f.write("\n")

    typer.secho(f"{SAVING} Results saved to: {output_filepath}", fg=typer.colors.WHITE, bold=True)


if __name__ == "__main__":
    typer.run(main)
