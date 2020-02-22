# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import io
import logging
import sys
from pathlib import Path

import numpy as np
from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.models.model import Model
from allennlp.predictors import Predictor

# Set PATHs
PATH_TO_SENTEVAL = 'SentEval'
PATH_TO_DATA = 'SentEval/data'
PATH_TO_ALLENLP_ARCHIVE = ''
CUDA_DEVICE = 0

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
sys.path.insert(1, '../t2t')
import senteval
from t2t.models import ContrastiveTextEncoder
from t2t.data.dataset_readers import ContrastiveDatasetReader

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):s
    instances = [params.reader.text_to_instance(" ".join(tokenized_text)) for tokenized_text in batch]
    outputs = params.model.forward_on_instances(instances)
    embeddings = np.vstack([output['embeddings'] for output in outputs])

    return embeddings

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load an AllenNLP Model ######################################################################################
    serialization_dir = Path(PATH_TO_ALLENLP_ARCHIVE)
    config_filepath = serialization_dir / 'config.json'
    allennlp_params = Params.from_file(config_filepath)
    dataset_reader_params = allennlp_params["dataset_reader"]

    # TODO (John): Is there a way to use --include-package so that these do not need to be
    # fully specified paths
    allennlp_params['model']['type'] = 't2t.models.contrastive_text_encoder.ContrastiveTextEncoder'
    dataset_reader_params['type'] = 't2t.data.dataset_readers.contrastive.ContrastiveDatasetReader'

    dataset_reader_params['sample_spans'] = False

    reader = DatasetReader.from_params(dataset_reader_params)
    model = Model.load(
        allennlp_params, serialization_dir=serialization_dir, cuda_device=CUDA_DEVICE
    ).eval()

    params_senteval['reader'] = reader
    params_senteval['model'] = model
    ###############################################################################################################

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
