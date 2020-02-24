from __future__ import absolute_import, division, unicode_literals

import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import typer

from allennlp.common import Params
from allennlp.common import util as common_util
from allennlp.data import DatasetReader
from allennlp.models.model import Model

app = typer.Typer()

# Set up logger
logger = logging.getLogger(__name__)

# TODO (John): This should be user configurable
TRANSFER_TASKS = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                  'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                  'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                  'Length', 'WordContent', 'Depth', 'TopConstituents',
                  'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                  'OddManOut', 'CoordinationInversion']

# Emoji's used in typer.secho calls
WARNING = '\U000026A0'
SUCCESS = '\U00002705'
RUNNING = '\U000023F3'
SAVING = '\U0001F4BE'


def _setup_senteval(
    path_to_senteval: str,
    prototyping_config: bool = False,
    verbose: bool = False
) -> None:
    if verbose:
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    # Set params for SentEval
    # See: https://github.com/facebookresearch/SentEval#senteval-parameters for explanation of the protoype config
    path_to_data = os.path.join(path_to_senteval, 'data')
    if prototyping_config:
        typer.secho(
            (f"{WARNING} Using prototyping config. Pass --no-prototyping-config to get results comparable to the"
             "literature."),
            fg=typer.colors.YELLOW,
            bold=True
        )
        params_senteval = {'task_path': path_to_data, 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
    else:
        params_senteval = {'task_path': path_to_data, 'usepytorch': True, 'kfold': 10}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}

    return params_senteval


def _run_senteval(params, path_to_senteval: str, batcher, prepare, output_filepath: str = None):
    # Import SentEval
    sys.path.insert(0, path_to_senteval)
    import senteval

    typer.secho(
        f"{SUCCESS} SentEval repository and transfer task data loaded successfully.",
        fg=typer.colors.GREEN,
        bold=True
    )
    typer.secho(f"{RUNNING} Running evaluation. This may take a while!", fg=typer.colors.WHITE, bold=True)

    se = senteval.engine.SE(params, batcher, prepare)
    results = se.eval(TRANSFER_TASKS)
    typer.secho(f"{SUCCESS} Evaluation complete!", fg=typer.colors.GREEN, bold=True)

    if output_filepath is not None:
        with open(output_filepath, "w") as fp:
            json.dump(common_util.sanitize(results), fp, indent=2)

        typer.secho(f"{SAVING} Results saved to: {output_filepath}", fg=typer.colors.WHITE, bold=True)
    else:
        typer.secho(
            f"{WARNING} --output_filepath was not provided, printing results to console instead.",
            fg=typer.colors.YELLOW,
            bold=True
        )
        print(results)


@app.command()
def allennlp(
    path_to_senteval: str,
    path_to_allennlp_archive: str,
    output_filepath: str = None,
    prototyping_config: bool = False,
    embeddings_field: str = "embeddings",
    cuda_device: int = -1,
    overrides: str = "",
    include_package: List[str] = None,
    verbose: bool = False
) -> None:
    """Evaluates a trained AllenNLP model against the SentEval benchmark.
    """

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        instances = [params.reader.text_to_instance(" ".join(tokenized_text)) for tokenized_text in batch]
        outputs = params.model.forward_on_instances(instances)
        embeddings = np.vstack([output[embeddings_field] for output in outputs])

        return embeddings

    # Performs a few setup steps and returns the SentEval params
    params_senteval = _setup_senteval(path_to_senteval, prototyping_config, verbose)

    # This allows us to import custom dataset readers and models that may exist in the AllenNLP archive.
    # See: https://github.com/allenai/allennlp/blob/e19605aae05eff60b0f41dc521b9787867fa58dd/allennlp/commands/train.py#L404
    include_package = include_package or []
    for package_name in include_package:
        common_util.import_module_and_submodules(package_name)

    serialization_dir = Path(path_to_allennlp_archive)
    config_filepath = serialization_dir / 'config.json'
    allennlp_params = Params.from_file(config_filepath, overrides)
    dataset_reader_params = allennlp_params["dataset_reader"]

    # Load the archived DatasetReader
    reader = DatasetReader.from_params(dataset_reader_params)
    typer.secho(
        f"{SUCCESS} DatasetReader from AllenNLP archive loaded successfully.",
        fg=typer.colors.GREEN,
        bold=True
    )

    # Load the archived AllenNLP model
    model = Model.load(
        allennlp_params, serialization_dir=serialization_dir, cuda_device=cuda_device
    ).eval()
    typer.secho(f"{SUCCESS} Model from AllenNLP archive loaded successfully", fg=typer.colors.GREEN, bold=True)

    params_senteval['reader'] = reader
    params_senteval['model'] = model

    _run_senteval(params_senteval, path_to_senteval, batcher, prepare, output_filepath)


@app.command()
def bow():
    raise NotImplementedError


@app.command()
def transformers():
    raise NotImplementedError


if __name__ == "__main__":
    app()
