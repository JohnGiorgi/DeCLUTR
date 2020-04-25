from pytorch_metric_learning import miners

from allennlp.common import Registrable


class PyTorchMetricLearningMiner(Registrable):
    """This class just allows us to implement `Registrable` for PyTorch Metric Learning miner functions.
    Subclasses of this class should also subclass a miner function from PyTorch Metric Learning
    (see: https://kevinmusgrave.github.io/pytorch-metric-learning/miners/), and accept as arguments
    to the constructor the same arguments that the miner function does. See `MaximumLossMiner` below
    for an example.
    """

    default_implementation = "batch_hard"


@PyTorchMetricLearningMiner.register("batch_hard")
class BatchHardMiner(PyTorchMetricLearningMiner, miners.BatchHardMiner):
    """Wraps the `BatchHardMinder` implementation from Pytorch Metric Learning:
    (https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#batchhardminer).

    Registered as a `PyTorchMetricLearningMiner` with name "batch_hard".
    """

    def __init__(
        self,
        use_similarity: bool = True,
        squared_distances: bool = False,
        normalize_embeddings: bool = True,
    ) -> None:

        super().__init__(
            use_similarity=use_similarity,
            squared_distances=squared_distances,
            normalize_embeddings=normalize_embeddings,
        )


@PyTorchMetricLearningMiner.register("hard_aware")
class HDCMiner(PyTorchMetricLearningMiner, miners.HDCMiner):
    """Wraps the `HDCMiner` implementation from Pytorch Metric Learning:
    (https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#hdcminer).

    Registered as a `PyTorchMetricLearningMiner` with name "hard_aware".
    """

    def __init__(
        self,
        filter_percentage: float,
        use_similarity: bool = True,
        squared_distances: bool = False,
        normalize_embeddings: bool = True,
    ) -> None:

        super().__init__(
            filter_percentage=filter_percentage,
            use_similarity=use_similarity,
            squared_distances=squared_distances,
            normalize_embeddings=normalize_embeddings,
        )


@PyTorchMetricLearningMiner.register("multi_sim_miner")
class MultiSimilarityMiner(PyTorchMetricLearningMiner, miners.MultiSimilarityMiner):
    """Wraps the `MultiSimilarityMiner` implementation from Pytorch Metric Learning:
    (https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#multisimilarityminer).

    Registered as a `PyTorchMetricLearningMiner` with name "multi_sim_miner".
    """

    def __init__(self, epsilon: float, normalize_embeddings: bool = True,) -> None:

        super().__init__(
            epsilon=epsilon, normalize_embeddings=normalize_embeddings,
        )
