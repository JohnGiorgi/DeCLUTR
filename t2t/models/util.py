from typing import Tuple

import torch


def format_embed_pt_metric_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A helper function which formats a pair of anchor, positive embeddings for use with a
    PyTorch Metric Learning loss function (https://github.com/KevinMusgrave/pytorch-metric-learning).
    These loss functions expect a single embedding tensor, and a corresponding set of labels. Given two tensors:
    `anchor_embeddings` and `positive_embeddings` each of shape `(batch_size, embedding_dim)`, concatenate them
    along the first dimension to produce a single tensor, `embeddings`, of shape `(batch_size * 2, embedding_dim)`.
    Then, generate the corresponding `labels` tensor of shape `(batch_size * 2)` by assigning a matching integer
    index to each pair of anchor, positive embeddings in `embeddings`.

    # Parameters

    anchor_embeddings : `torch.Tensor`
        Encoded representations of the anchors.
    positive_embeddings : `torch.Tensor`
        Encoded representations of the positives.

    # Returns

    A tuple of embeddings and labels that can be fed directly to any PyTorch-Metric-Learning loss function.
    """
    embeddings = torch.cat((anchor_embeddings, positive_embeddings))
    indices = torch.arange(0, anchor_embeddings.size(0), device=anchor_embeddings.device)
    labels = torch.cat((indices, indices))

    return embeddings, labels
