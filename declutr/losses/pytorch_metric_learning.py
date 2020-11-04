import sys
from typing import Tuple

import torch
from allennlp.common import Registrable

from declutr.miners import PyTorchMetricLearningMiner
from pytorch_metric_learning import losses


class PyTorchMetricLearningLoss(Registrable):
    """This class allows us to implement `Registrable` for PyTorch Metric Learning loss functions.
    Subclasses of this class should also subclass a loss function from PyTorch Metric Learning
    (see: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/), and accept as arguments
    to the constructor the same arguments that the loss function does. See `NTXentLoss` below for
    an example.
    """

    default_implementation = "nt_xent"

    @classmethod
    def get_embeddings_and_labels(
        self, anchors: torch.Tensor, positives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Formats a pair of anchor, positive embeddings for use with a PyTorch Metric Learning loss
        function (https://github.com/KevinMusgrave/pytorch-metric-learning). These loss functions
        expect a single embedding tensor, and a corresponding set of labels. Given two tensors:
        `anchor_embeddings` and `positive_embeddings` each of shape `(batch_size, embedding_dim)`,
        concatenate them along the first dimension to produce a single tensor, `embeddings`, of
        shape `(batch_size * 2, embedding_dim)`. Then, generate the corresponding `labels` tensor of
        shape `(batch_size * 2)` by assigning a matching integer index to each pair of anchor,
        positive embeddings in `embeddings`.

        # Parameters

        anchor_embeddings : `torch.Tensor`
            Encoded representations of the anchors.
        positive_embeddings : `torch.Tensor`
            Encoded representations of the positives.

        # Returns

        A tuple of embeddings and labels that can be fed directly to any PyTorch Metric Learning
        loss function.
        """
        embeddings = torch.cat((anchors, positives))
        # When using CrossBatchMemory, labels persist across batches so they need to be unique.
        # By choosing a random integer in (0, sys.maxsize) we can be reasonably sure of this.
        # Obviously, there are better (i.e. deterministic ways to do this), but I don't have
        # access to the current batch id or some other uniquely identifying value.
        indices = torch.randint(sys.maxsize, (anchors.size(0),), device=anchors.device)
        labels = torch.cat((indices, indices))

        return embeddings, labels


@PyTorchMetricLearningLoss.register("cross_batch_memory")
class CrossBatchMemory(PyTorchMetricLearningLoss, losses.CrossBatchMemory):
    """Wraps the `CrossBatchMemory` implementation from Pytorch Metric Learning:
    (https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#crossbatchmemory).

    Registered as a `PyTorchMetricLearningLoss` with name "cross_batch_memory".
    """

    def __init__(
        self,
        loss: PyTorchMetricLearningLoss,
        embedding_size: int,
        memory_size: int = 1024,
        miner: PyTorchMetricLearningMiner = None,
    ) -> None:

        super().__init__(
            loss=loss,
            embedding_size=embedding_size,
            memory_size=memory_size,
            miner=miner,
        )


@PyTorchMetricLearningLoss.register("nt_xent")
class NTXentLoss(PyTorchMetricLearningLoss, losses.NTXentLoss):
    """Wraps the `NTXentLoss` implementation from Pytorch Metric Learning:
    (https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss).

    Registered as a `PyTorchMetricLearningLoss` with name "nt_xent".
    """

    def __init__(self, temperature: float) -> None:

        super().__init__(temperature=temperature)
