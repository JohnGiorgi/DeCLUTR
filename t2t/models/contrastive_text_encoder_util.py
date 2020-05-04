from typing import Tuple, List

import torch
import torch.distributed as dist

from allennlp.common import util
from allennlp.data import TextFieldTensors


def unpack_pack_chunks(tokens: TextFieldTensors) -> List[TextFieldTensors]:
    """Returns a tuple of `TextFieldTensors` containing random batches of anchors and positives from tokens.

    # Parameters

    tokens : TextFieldTensors
        From a `TextField`

    anchors : TextFieldTensors
        `TextFieldTensors` containing the sampled anchors.
    positives : TextFieldTensors
        `TextFieldTensors` containing the sampled positives.
    """
    batch_size = tokens["tokens"]["token_ids"].size(0)

    token_ids = torch.chunk(tokens["tokens"]["token_ids"], batch_size, dim=0)
    masks = torch.chunk(tokens["tokens"]["mask"], batch_size, dim=0)
    type_ids = torch.chunk(tokens["tokens"]["type_ids"], batch_size, dim=0)

    chunks = []
    for x, y, z in zip(token_ids, masks, type_ids):
        chunk: TextFieldTensors = {
            "tokens": {"token_ids": x.squeeze(0), "mask": y.squeeze(0), "type_ids": z.squeeze(0)}
        }
        chunks.append(chunk)

    return chunks


def all_gather_anchor_positive_pairs(
    anchors: torch.Tensor, positives: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """If training on 2 or more GPUs, `all_gather`s the embeddings produced on each replica,
    ensuring that the gradients for the embeddings produced on each replica are not lost. The
    returned anchor, positive pairs can be fed to a contrastive loss. This method is necessary to
    ensure that we train against the expected number of negatives 2 * (batch size - 1) per batch,
    as a naive implementation would end up training against 2 * (batch size / n_gpus - 1) number of
    negatives. If we are not training on 2 or more GPUs, this method is a no-op and returns its
    inputs.

    # Parameters

    anchors : torch.Tensor
        Embedded text representing the anchors.
    positives : TextFieldTensors
        Embedded text representing the positives.

    # Returns

    Tuple[torch.Tensor, torch.Tensor]
        Embedded anchor, positive pairs that can be fed to a contrastive loss.
    """

    # If we are not using distributed training, this is a no-op.
    if not util.is_distributed():
        return anchors, positives

    # Gather the encoded anchors and positives on all replicas
    anchors_list = [torch.ones_like(anchors) for _ in range(dist.get_world_size())]
    positives_list = [torch.ones_like(positives) for _ in range(dist.get_world_size())]
    dist.all_gather(anchors_list, anchors)
    dist.all_gather(positives_list, positives)
    # The gathered copy of the current replicas positive pairs have no gradients, so we overwrite them
    # with the positive pairs generated on this replica, which DO have gradients back to the encoder.
    anchors_list[dist.get_rank()] = anchors
    positives_list[dist.get_rank()] = positives
    # Finally, we concatenate the positive pairs so they can be fed to the contrastive loss
    anchors = torch.cat(anchors_list)
    positives = torch.cat(positives_list)

    return anchors, positives
