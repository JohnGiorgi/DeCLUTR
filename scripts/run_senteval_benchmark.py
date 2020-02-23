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

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def main(
    path_to_senteval: str,
    path_to_allennlp_archive: str,
    prototyping_config: bool = False,
    cuda_device: int = -1,
    overrides: str = "",
    include_package: List[str] = None,
) -> None:
    """Evaluates a trained AllenNLP model against the SentEval benchmark.
    """
    # Set params for SentEval
    # See: https://github.com/facebookresearch/SentEval#senteval-parameters for explanation of the protoype config
    path_to_data = os.path.join(path_to_senteval, 'data')
    if prototyping_config:
        typer.secho(
            f"\U000026A0 Using prototyping config. Pass --no-prototyping-config to get results comparable to the literature.",
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

    # Load an AllenNLP Model
    load_model(params_senteval, path_to_allennlp_archive, cuda_device, overrides, include_package)

    # Run SentEval
    run_senteval(params_senteval, path_to_senteval)


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    instances = [params.reader.text_to_instance(" ".join(tokenized_text)) for tokenized_text in batch]
    outputs = params.model.forward_on_instances(instances)
    embeddings = np.vstack([output['embeddings'] for output in outputs])

    return embeddings


def load_model(
    params,
    path_to_allennlp_archive: str,
    cuda_device: int = -1,
    overrides: str = "",
    include_package: List[str] = None,
) -> None:
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
        f"\U00002705 DatasetReader from AllenNLP archive loaded successfully.",
        fg=typer.colors.GREEN,
        bold=True
    )

    # Load the archived AllenNLP model
    model = Model.load(
        allennlp_params, serialization_dir=serialization_dir, cuda_device=cuda_device
    ).eval()
    typer.secho(f"\U00002705 Model from AllenNLP archive loaded successfully", fg=typer.colors.GREEN, bold=True)

    params['reader'] = reader
    params['model'] = model

    return


def run_senteval(params, path_to_senteval: str):
    # Import SentEval
    sys.path.insert(0, path_to_senteval)
    import senteval

    typer.secho(
        f"\U00002705 SentEval repository and transfer task data loaded successfully.",
        fg=typer.colors.GREEN,
        bold=True
    )

    se = senteval.engine.SE(params, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    # TODO (John): It would be great if we could:
    #   1. Format this nicely as a table.
    #   2. Save it to disk as a CSV or something comparable.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    typer.run(main)
