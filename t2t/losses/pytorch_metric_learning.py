from typing import List, Tuple

import torch
from pytorch_metric_learning import losses

from allennlp.common import Registrable
from t2t.miners import PyTorchMetricLearningMiner


class PyTorchMetricLearningLoss(Registrable):
    """This class just allows us to implement `Registrable` for PyTorch Metric Learning loss functions.
    Subclasses of this class should also subclass a loss function from PyTorch Metric Learning
    (see: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/), and accept as arguments
    to the constructor the same arguments that the loss function does. See `NTXentLoss` below for an example.
    """

    default_implementation = "nt_xent"

    @classmethod
    def get_embeddings_and_labels(
        self, anchors: torch.Tensor, positives: torch.Tensor
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


@PyTorchMetricLearningLoss.register("circle_loss")
class CircleLoss(PyTorchMetricLearningLoss, losses.CircleLoss):
    """Wraps the `CircleLoss` implementation from Pytorch Metric Learning:
    (https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss).

    Registered as a `PyTorchMetricLearningLoss` with name "circle_loss".
    """

    def __init__(
        self,
        m: float = 0.4,
        gamma: int = 80,
        triplets_per_anchor: str = "all",
        num_class_per_param: int = None,
        learnable_param_names: List[str] = None,
    ) -> None:

        super().__init__(
            m=m,
            gamma=gamma,
            triplets_per_anchor=triplets_per_anchor,
            num_class_per_param=num_class_per_param,
            learnable_param_names=learnable_param_names,
        )


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
            loss=loss, embedding_size=embedding_size, memory_size=memory_size, miner=miner,
        )


@PyTorchMetricLearningLoss.register("multi_sim_loss")
class MultiSimilarityLoss(PyTorchMetricLearningLoss, losses.MultiSimilarityLoss):
    """Wraps the `MultiSimilarityLoss` implementation from Pytorch Metric Learning:
    (https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multisimilarityloss).

    Registered as a `PyTorchMetricLearningLoss` with name "multi_sim_loss".
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        base: float = 0.5,
        normalize_embeddings: bool = True,
        num_class_per_param: int = None,
        learnable_param_names: List[str] = None,
    ) -> None:

        super().__init__(
            alpha=alpha,
            beta=beta,
            base=base,
            normalize_embeddings=normalize_embeddings,
            num_class_per_param=num_class_per_param,
            learnable_param_names=learnable_param_names,
        )


@PyTorchMetricLearningLoss.register("nt_xent")
class NTXentLoss(PyTorchMetricLearningLoss, losses.NTXentLoss):
    """Wraps the `NTXentLoss` implementation from Pytorch Metric Learning:
    (https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss).

    Registered as a `PyTorchMetricLearningLoss` with name "nt_xent".
    """

    def __init__(
        self,
        temperature: float,
        normalize_embeddings: bool = True,
        num_class_per_param: int = None,
        learnable_param_names: List[str] = None,
    ) -> None:

        super().__init__(
            temperature=temperature,
            normalize_embeddings=normalize_embeddings,
            num_class_per_param=num_class_per_param,
            learnable_param_names=learnable_param_names,
        )
