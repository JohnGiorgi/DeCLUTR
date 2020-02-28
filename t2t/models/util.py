import random
from typing import Tuple

import torch

from allennlp.data import TextFieldTensors


def sample_anchor_positive_pairs(tokens) -> Tuple[TextFieldTensors, TextFieldTensors]:
    """Returns a tuple of `TextFieldTensors` containing random batches of anchors and positives from tokens.

    # Parameters

    tokens : TextFieldTensors
        From a `TextField`
    """
    # The procedure for sampling anchor, positive pairs is as follows:
    #   1. Sample two random spans for every training instance
    #   2. Unpack the TextFieldTensors, extract the token ids, masks, and type ids for the sampled pairs
    #   3. Repackage the information into TextFieldTensors
    num_spans = tokens["tokens"]["token_ids"].size(1)
    index = torch.as_tensor(
        random.sample(range(0, num_spans), 2),
        device=tokens["tokens"]["token_ids"].device,
    )

    random_token_ids = torch.index_select(
        tokens["tokens"]["token_ids"], dim=1, index=index
    )
    random_masks = torch.index_select(tokens["tokens"]["mask"], dim=1, index=index)
    random_type_ids = torch.index_select(
        tokens["tokens"]["type_ids"], dim=1, index=index
    )

    anchor_token_ids, positive_token_ids = torch.chunk(random_token_ids, 2, dim=1)
    anchor_masks, positive_masks = torch.chunk(random_masks, 2, dim=1)
    anchor_type_ids, positive_type_ids = torch.chunk(random_type_ids, 2, dim=1)

    anchors: TextFieldTensors = {
        "tokens": {
            "token_ids": anchor_token_ids.squeeze(1),
            "mask": anchor_masks.squeeze(1),
            "type_ids": anchor_type_ids.squeeze(1),
        }
    }
    positives: TextFieldTensors = {
        "tokens": {
            "token_ids": positive_token_ids.squeeze(1),
            "mask": positive_masks.squeeze(1),
            "type_ids": positive_type_ids.squeeze(1),
        }
    }

    return anchors, positives
