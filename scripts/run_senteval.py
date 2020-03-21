#!/usr/bin/env python
from __future__ import absolute_import, division, unicode_literals

import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
import typer

# TODO (John): I only need this for the sanitize method. Would be better if this was NOT a dependency for people
# who don't want to evaluate AllenNLP models.
from allennlp.common import util as common_util

try:
    from apex import amp
except ImportError:
    amp = None


app = typer.Typer()

# Set up logger
logger = logging.getLogger(__name__)

# TODO (John): This should be user configurable
TRANSFER_TASKS = [
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "MR",
    "CR",
    "MPQA",
    "SUBJ",
    "SST2",
    "SST5",
    "TREC",
    "MRPC",
    "SICKEntailment",
    "SICKRelatedness",
    "STSBenchmark",
    "Length",
    "WordContent",
    "Depth",
    "TopConstituents",
    "BigramShift",
    "Tense",
    "SubjNumber",
    "ObjNumber",
    "OddManOut",
    "CoordinationInversion",
]

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py
WARNING = "\U000026A0"
SUCCESS = "\U00002705"
RUNNING = "\U000023F3"
SAVING = "\U0001F4BE"
FAST = "\U0001F3C3"
SCORE = "\U0001F4CB"


def _get_device(cuda_device):
    """Return a `torch.cuda` device if `torch.cuda.is_available()` and `cuda_device` is non-negative. Otherwise
    returns a `torch.cpu` device.
    """
    if cuda_device != -1 and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def _compute_aggregate_score(results):
    """Computes a simple aggregate score score for the given SentEval `results`.
    """
    aggregate_score = 0
    for task, scores in results.items():
        if task == "STSBenchmark" or task == "SICKRelatedness":
            aggregate_score += scores["spearman"] * 100
        elif task.startswith("STS"):
            aggregate_score += scores["all"]["spearman"]["mean"] * 100
        elif "acc" in scores:
            aggregate_score += scores["acc"]
        else:
            raise ValueError(f'Found an unexpected field in results, "{task}".')
    return aggregate_score / len(results)


def _setup_mixed_precision_with_amp(model: torch.nn.Module, opt_level: str = None):
    """Wraps a model with NVIDIAs amp API for mixed-precision inference. This is a no-op if `opt_level is None`.
    """

    if opt_level is not None:
        if amp is None:
            raise ValueError(
                (
                    "Apex not installed but opt_level was provided. Please install NVIDIA's Apex to enable"
                    " automatic mixed precision (AMP) for inference. See: https://github.com/NVIDIA/apex."
                )
            )

        model = amp.initialize(model, opt_level=opt_level)
        typer.secho(
            f'{FAST} Using mixed-precision with "opt_level={opt_level}".', fg=typer.colors.WHITE, bold=True
        )

    return model


def _pad_sequences(sequences, pad_token):
    """Pads the elements of `sequences` with `pad_token` up to the longest sequence length."""
    max_len = len(max(sequences, key=len))
    padded_sequences = [seq + ([pad_token] * (max_len - len(seq))) for seq in sequences]
    return padded_sequences


def _setup_senteval(path_to_senteval: str, prototyping_config: bool = False, verbose: bool = False) -> None:
    if verbose:
        logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)

    # Set params for SentEval
    # See: https://github.com/facebookresearch/SentEval#senteval-parameters for explanation of the protoype config
    path_to_data = os.path.join(path_to_senteval, "data")
    if prototyping_config:
        typer.secho(
            (
                f"{WARNING} Using prototyping config. Pass --no-prototyping-config to get results comparable to"
                "  the literature."
            ),
            fg=typer.colors.YELLOW,
            bold=True,
        )
        params_senteval = {"task_path": path_to_data, "usepytorch": True, "kfold": 5}
        params_senteval["classifier"] = {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 128,
            "tenacity": 3,
            "epoch_size": 2,
        }
    else:
        params_senteval = {"task_path": path_to_data, "usepytorch": True, "kfold": 10}
        params_senteval["classifier"] = {
            "nhid": 0,
            "optim": "adam",
            "batch_size": 64,
            "tenacity": 5,
            "epoch_size": 4,
        }

    return params_senteval


def _run_senteval(
    params, path_to_senteval: str, batcher: Callable, prepare: Callable, output_filepath: str = None
) -> None:
    sys.path.insert(0, path_to_senteval)
    import senteval

    typer.secho(
        f"{SUCCESS}  SentEval repository and transfer task data loaded successfully.",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.secho(f"{RUNNING}  Running evaluation. This may take a while!", fg=typer.colors.WHITE, bold=True)

    se = senteval.engine.SE(params, batcher, prepare)
    results = se.eval(TRANSFER_TASKS)
    typer.secho(f"{SUCCESS}  Evaluation complete!", fg=typer.colors.GREEN, bold=True)

    score = _compute_aggregate_score(results)
    typer.secho(f"{SCORE}  Aggregate score: {score:.2f}%", fg=typer.colors.WHITE, bold=True)

    if output_filepath is not None:
        # Create the directory path if it doesn't exist
        output_filepath = Path(output_filepath)
        output_filepath.parents[0].mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "w") as fp:
            json.dump(common_util.sanitize(results), fp, indent=2)
        typer.secho(f"{SAVING}  Results saved to: {output_filepath}", fg=typer.colors.WHITE, bold=True)
    else:
        typer.secho(
            f"{WARNING} --output_filepath was not provided, printing results to console instead.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
        print(results)
    return


@app.command()
def aggregate_score(path_to_results: str):
    with open(path_to_results, "r") as f:
        results = json.load(f)
    score = _compute_aggregate_score(results)
    typer.secho(f"{SCORE}  Aggregate score: {score:.2f}%", fg=typer.colors.WHITE, bold=True)
    return score


@app.command()
def bow() -> None:
    """Evaluates pre-trained word vectors against the SentEval benchmark.
    """
    raise NotImplementedError


@app.command()
def transformers(
    path_to_senteval: str,
    pretrained_model_name_or_path: str,
    output_filepath: str = None,
    prototyping_config: bool = False,
    mean_pool: bool = False,
    cuda_device: int = -1,
    opt_level: str = None,
    verbose: bool = False,
) -> None:
    """Evaluates a pre-trained model from the Transformers library against the SentEval benchmark.
    """

    # This prevents import errors when a user doesn't have the dependencies for this command installed
    from transformers import AutoModel, AutoTokenizer

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        # Re-tokenize the input text using the pre-trained tokenizer
        batch = [tokenizer.encode(" ".join(tokens)) for tokens in batch]
        batch = _pad_sequences(batch, tokenizer.pad_token_id)

        # To embed with transformers, we (minimally) need the input ids and the attention masks.
        input_ids = torch.as_tensor(batch, device=params.device)
        attention_masks = torch.where(
            input_ids == params.tokenizer.pad_token_id, torch.zeros_like(input_ids), torch.ones_like(input_ids)
        )

        with torch.no_grad():
            sequence_output, pooled_output = params.model(input_ids=input_ids, attention_mask=attention_masks)

        # If mean_pool, we take the average of the token-level embeddings, accounting for pads.
        # Otherwise, we take the pooled output for this specific model, which is typically the linear projection
        # of a special tokens embedding, like [CLS] or <s>, which is prepended to the input during tokenization.
        if mean_pool:
            embeddings = torch.sum(sequence_output * attention_masks.unsqueeze(-1), dim=1) / torch.clamp(
                torch.sum(attention_masks, dim=1, keepdims=True), min=1e-9
            )
        else:
            embeddings = pooled_output

        embeddings = embeddings.cpu().numpy()
        return embeddings

    # Determine the torch device
    device = _get_device(cuda_device)

    # Load the Transformers tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    typer.secho(
        f'{SUCCESS}  Tokenizer "{pretrained_model_name_or_path}" from Transformers loaded successfully.',
        fg=typer.colors.GREEN,
        bold=True,
    )

    # Load the Transformers model
    model = AutoModel.from_pretrained(pretrained_model_name_or_path).to(device)
    model.eval()
    typer.secho(
        f'{SUCCESS}  Model "{pretrained_model_name_or_path}" from Transformers loaded successfully.',
        fg=typer.colors.GREEN,
        bold=True,
    )

    # Used mixed-precision to speed up inference
    model = _setup_mixed_precision_with_amp(model, opt_level)

    # Performs a few setup steps and returns the SentEval params
    params_senteval = _setup_senteval(path_to_senteval, prototyping_config, verbose)
    params_senteval["tokenizer"] = tokenizer
    params_senteval["model"] = model
    params_senteval["device"] = device
    _run_senteval(params_senteval, path_to_senteval, batcher, prepare, output_filepath)

    return


@app.command()
def sentence_transformers(
    path_to_senteval: str,
    pretrained_model_name_or_path: str,
    output_filepath: str = None,
    prototyping_config: bool = False,
    cuda_device: int = -1,
    opt_level: str = None,
    verbose: bool = False,
) -> None:
    """Evaluates a pre-trained model from the Sentence Transformers library against the SentEval benchmark.
    """

    # This prevents import errors when a user doesn't have the dependencies for this command installed
    from sentence_transformers import SentenceTransformer

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        # Sentence Transformers API expects un-tokenized sentences.
        batch = [" ".join(tokens) for tokens in batch]
        with torch.no_grad():
            embeddings = params.model.encode(batch)
        embeddings = np.vstack(embeddings)
        return embeddings

    # Determine the torch device
    device = _get_device(cuda_device)

    # Load the Sentence Transformers tokenizer
    model = SentenceTransformer(pretrained_model_name_or_path, device=device)
    model.eval()
    typer.secho(
        f'{SUCCESS}  Model "{pretrained_model_name_or_path}" from Sentence Transformers loaded successfully.',
        fg=typer.colors.GREEN,
        bold=True,
    )
    # Used mixed-precision to speed up inference
    model = _setup_mixed_precision_with_amp(model, opt_level)

    # Performs a few setup steps and returns the SentEval params
    params_senteval = _setup_senteval(path_to_senteval, prototyping_config, verbose)
    params_senteval["model"] = model
    _run_senteval(params_senteval, path_to_senteval, batcher, prepare, output_filepath)

    return


@app.command()
def allennlp(
    path_to_senteval: str,
    path_to_allennlp_archive: str,
    predictor_name: str,
    output_filepath: str = None,
    prototyping_config: bool = False,
    embeddings_field: str = "embeddings",
    cuda_device: int = -1,
    include_package: List[str] = None,
    verbose: bool = False,
) -> None:
    """Evaluates a trained AllenNLP model against the SentEval benchmark.
    """

    # This prevents import errors when a user doesn't have the dependencies for this command installed
    from allennlp.models.archival import load_archive
    from allennlp.predictors import Predictor

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        with torch.no_grad():
            # Re-tokenize the input text using the tokenizer of the dataset reader
            inputs = [{"text": " ".join(tokens)} for tokens in batch]
            outputs = params.predictor.predict_batch_json(inputs)
        # AllenNLP models return a dictionary, so we need to access the embeddings with the given key.
        embeddings = [output[embeddings_field] for output in outputs]

        embeddings = np.vstack(embeddings)
        return embeddings

    # This allows us to import custom dataset readers and models that may exist in the AllenNLP archive.
    # See: https://github.com/allenai/allennlp/blob/e19605aae05eff60b0f41dc521b9787867fa58dd/allennlp/commands/train.py#L404
    include_package = include_package or []
    for package_name in include_package:
        common_util.import_module_and_submodules(package_name)

    # Load the archived Model
    archive = load_archive(path_to_allennlp_archive)
    predictor = Predictor.from_archive(archive, predictor_name)
    typer.secho(f"{SUCCESS}  Model from AllenNLP archive loaded successfully.", fg=typer.colors.GREEN, bold=True)

    # Used mixed-precision to speed up inference
    # model = _setup_mixed_precision_with_amp(model, allennlp_params["trainer"]["opt_level"])

    # Performs a few setup steps and returns the SentEval params
    params_senteval = _setup_senteval(path_to_senteval, prototyping_config, verbose)
    params_senteval["predictor"] = predictor
    _run_senteval(params_senteval, path_to_senteval, batcher, prepare, output_filepath)

    return


if __name__ == "__main__":
    app()
