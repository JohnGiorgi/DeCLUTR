#!/usr/bin/env python
from __future__ import absolute_import, division, unicode_literals

import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, List
from statistics import mean

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

DOWNSTREAM_TASKS = [
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
]
PROBING_TASKS = [
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
TRANSFER_TASKS = DOWNSTREAM_TASKS + PROBING_TASKS

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


def _compute_aggregate_scores(results):
    """Computes aggregate scores for the dev and test sets in a given SentEval `results`.
    """
    aggregate_scores = {
        "downstream": {"dev": 0, "test": 0},
        "probing": {"dev": 0, "test": 0},
        "all": {},
    }
    for task, scores in results.items():
        # Tasks belong to two groups only, "downstream" or "probing"
        task_set = "downstream" if task in DOWNSTREAM_TASKS else "probing"
        # All of this conditional logic is required to deal with the various ways performance is
        # reported for each task.
        # These two tasks report pearsonr for dev and spearman for test. Not sure why?
        if task == "STSBenchmark" or task == "SICKRelatedness":
            aggregate_scores[task_set]["dev"] += scores["devpearson"] * 100
            aggregate_scores[task_set]["test"] += scores["spearman"] * 100
        # There are no partitions for these tasks as no model is trained on top of the embeddings,
        # therefore we count their scores in both the dev and test accumulators.
        elif task.startswith("STS"):
            sts_score = scores["all"]["spearman"]["mean"] * 100
            aggregate_scores[task_set]["dev"] += sts_score
            aggregate_scores[task_set]["test"] += sts_score
        # The rest of the tasks seem to all contain a dev and test accuracy
        elif "devacc" in scores and "acc" in scores:
            aggregate_scores[task_set]["dev"] += scores["devacc"]
            # f1 is in the score, average it with acc before accumulating
            if "f1" in scores:
                aggregate_scores[task_set]["test"] += mean([scores["acc"], scores["f1"]])
            else:
                aggregate_scores[task_set]["test"] += scores["acc"]
        else:
            raise ValueError(f'Found an unexpected field in results, "{task}".')

    # Aggregate scores for "downstream" tasks
    aggregate_scores["downstream"]["dev"] /= len(DOWNSTREAM_TASKS)
    aggregate_scores["downstream"]["test"] /= len(DOWNSTREAM_TASKS)
    # Aggregate scores for "probing" tasks
    aggregate_scores["probing"]["dev"] /= len(PROBING_TASKS)
    aggregate_scores["probing"]["test"] /= len(PROBING_TASKS)
    # Aggregate score across all of SentEval
    aggregate_scores["all"]["dev"] = mean(
        [aggregate_scores["downstream"]["dev"], aggregate_scores["probing"]["dev"]]
    )
    aggregate_scores["all"]["test"] = mean(
        [aggregate_scores["downstream"]["test"], aggregate_scores["probing"]["test"]]
    )

    return aggregate_scores


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
            f'{FAST} Using mixed-precision with "opt_level={opt_level}".',
            fg=typer.colors.WHITE,
            bold=True,
        )

    return model


def _pad_sequences(sequences, pad_token):
    """Pads the elements of `sequences` with `pad_token` up to the longest sequence length."""
    max_len = len(max(sequences, key=len))
    padded_sequences = [seq + ([pad_token] * (max_len - len(seq))) for seq in sequences]
    return padded_sequences


def _setup_senteval(
    path_to_senteval: str, prototyping_config: bool = False, verbose: bool = False
) -> None:
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
    typer.secho(
        f"{RUNNING}  Running evaluation. This may take a while!", fg=typer.colors.WHITE, bold=True
    )

    se = senteval.engine.SE(params, batcher, prepare)
    results = se.eval(TRANSFER_TASKS)
    typer.secho(f"{SUCCESS}  Evaluation complete!", fg=typer.colors.GREEN, bold=True)

    aggregate_scores = _compute_aggregate_scores(results)
    typer.secho(
        f'{SCORE} Aggregate dev score: {aggregate_scores["all"]["dev"]:.2f}%',
        fg=typer.colors.WHITE,
        bold=True,
    )

    if output_filepath is not None:
        # Create the directory path if it doesn't exist
        output_filepath = Path(output_filepath)
        output_filepath.parents[0].mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "w") as fp:
            # Convert anything that can't be serialized to JSON to a python type
            json_safe_results = common_util.sanitize(results)
            # Add aggregate scores to results dict
            json_safe_results["aggregate_scores"] = aggregate_scores
            json.dump(json_safe_results, fp, indent=2)
        typer.secho(
            f"{SAVING}  Results saved to: {output_filepath}", fg=typer.colors.WHITE, bold=True
        )
    else:
        typer.secho(
            f"{WARNING} --output_filepath was not provided, printing results to console instead.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
        print(results)
    return


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

    @torch.no_grad()
    def batcher(params, batch):
        # I am not sure why, but some SentEval tasks contain empty batches which triggers an error
        # with HuggingfFace's tokenizer.
        # I am using the solution found in the SentEvel repo here:
        # https://github.com/facebookresearch/SentEval/blob/6b13ac2060332842f59e84183197402f11451c94/examples/bow.py#L77
        batch = [sent if sent != [] else ['.'] for sent in batch]
        # Re-tokenize the input text using the pre-trained tokenizer
        batch = [tokenizer.encode(" ".join(tokens)) for tokens in batch]
        batch = _pad_sequences(batch, tokenizer.pad_token_id)

        # To embed with transformers, we (minimally) need the input ids and the attention masks.
        input_ids = torch.as_tensor(batch, device=params.device)
        attention_masks = torch.where(
            input_ids == params.tokenizer.pad_token_id,
            torch.zeros_like(input_ids),
            torch.ones_like(input_ids),
        )

        sequence_output, _ = params.model(input_ids=input_ids, attention_mask=attention_masks)

        # If mean_pool, we take the average of the token-level embeddings, accounting for pads.
        # Otherwise, we take the pooled output for this specific model, which is typically the linear projection
        # of a special tokens embedding, like [CLS] or <s>, which is prepended to the input during tokenization.
        if mean_pool:
            embeddings = torch.sum(
                sequence_output * attention_masks.unsqueeze(-1), dim=1
            ) / torch.clamp(torch.sum(attention_masks, dim=1, keepdims=True), min=1e-9)
        else:
            # TODO (John): Replace this with the built in pooler from the Transformers lib,
            # as it will check if the last token should be used.
            embeddings = sequence_output[:, 0, :]

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
        f'{SUCCESS} Model "{pretrained_model_name_or_path}" from Transformers loaded successfully.',
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

    @torch.no_grad()
    def batcher(params, batch):
        # I am not sure why, but some SentEval tasks contain empty batches which triggers an error
        # with HuggingfFace's tokenizer.
        # I am using the solution found in the SentEvel repo here:
        # https://github.com/facebookresearch/SentEval/blob/6b13ac2060332842f59e84183197402f11451c94/examples/bow.py#L77
        batch = [sent if sent != [] else ['.'] for sent in batch]
        # Sentence Transformers API expects un-tokenized sentences.
        batch = [" ".join(tokens) for tokens in batch]
        embeddings = params.model.encode(batch)
        embeddings = np.vstack(embeddings)
        return embeddings

    # Determine the torch device
    device = _get_device(cuda_device)

    # Load the Sentence Transformers tokenizer
    model = SentenceTransformer(pretrained_model_name_or_path, device=device)
    model.eval()
    typer.secho(
        f'{SUCCESS} Model "{pretrained_model_name_or_path}" from Sentence Transformers loaded successfully.',
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
    opt_level: str = "O0",
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

    @torch.no_grad()
    def batcher(params, batch):
        # I am not sure why, but some SentEval tasks contain empty batches which triggers an error
        # with HuggingfFace's tokenizer.
        # I am using the solution found in the SentEvel repo here:
        # https://github.com/facebookresearch/SentEval/blob/6b13ac2060332842f59e84183197402f11451c94/examples/bow.py#L77
        batch = [sent if sent != [] else ['.'] for sent in batch]
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
    archive = load_archive(path_to_allennlp_archive, cuda_device=cuda_device)
    predictor = Predictor.from_archive(archive, predictor_name)
    typer.secho(
        f"{SUCCESS}  Model from AllenNLP archive loaded successfully.",
        fg=typer.colors.GREEN,
        bold=True,
    )

    # Performs a few setup steps and returns the SentEval params
    params_senteval = _setup_senteval(path_to_senteval, prototyping_config, verbose)
    params_senteval["predictor"] = predictor
    _run_senteval(params_senteval, path_to_senteval, batcher, prepare, output_filepath)

    return


if __name__ == "__main__":
    app()
