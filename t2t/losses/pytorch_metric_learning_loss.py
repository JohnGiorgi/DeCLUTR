from typing import List, Tuple

import torch
from pytorch_metric_learning.losses import NTXentLoss as NTXent

from allennlp.common import Registrable


class PyTorchMetricLearningLoss(Registrable):

    @classmethod
    def get_embeddings_and_labels(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Formats a pair of anchor, positive embeddings for use with a PyTorch Metric Learning loss function
        (https://github.com/KevinMusgrave/pytorch-metric-learning). These loss functions expect a single embedding
        tensor, and a corresponding set of labels. Given two tensors: `anchor_embeddings` and `positive_embeddings`
        each of shape `(batch_size, embedding_dim)`, concatenate them along the first dimension to produce a single
        tensor, `embeddings`, of shape `(batch_size * 2, embedding_dim)`. Then, generate the corresponding `labels`
        tensor of shape `(batch_size * 2)` by assigning a matching integer index to each pair of anchor, positive
        embeddings in `embeddings`.

        # Parameters

        anchor_embeddings : `torch.Tensor`
            Encoded representations of the anchors.
        positive_embeddings : `torch.Tensor`
            Encoded representations of the positives.

        # Returns

        A tuple of embeddings and labels that can be fed directly to any PyTorch-Metric-Learning loss function.
        """
        embeddings = torch.cat((anchors, positives))
        indices = torch.arange(0, anchors.size(0), device=anchors.device)
        labels = torch.cat((indices, indices))

        return embeddings, labels


@PyTorchMetricLearningLoss.register('nt-xent')
class NTXentLoss(PyTorchMetricLearningLoss, NTXent):
    def __init__(
        self,
        temperature: float,
        normalize_embeddings: bool = True,
        num_class_per_param: int = None,
        learnable_param_names: List[str] = None
    ) -> None:

        super().__init__(
            temperature=temperature,
            normalize_embeddings=normalize_embeddings,
            num_class_per_param=num_class_per_param,
            learnable_param_names=learnable_param_names
        )
