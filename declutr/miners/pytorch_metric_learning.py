from pytorch_metric_learning import miners

from allennlp.common import Registrable


class PyTorchMetricLearningMiner(Registrable):
    """This class just allows us to implement `Registrable` for PyTorch Metric Learning miner functions.
    Subclasses of this class should also subclass a miner function from PyTorch Metric Learning
    (see: https://kevinmusgrave.github.io/pytorch-metric-learning/miners/), and accept as arguments
    to the constructor the same arguments that the miner function does. See `MaximumLossMiner` below
    for an example.
    """

    default_implementation = "pair_margin"


@PyTorchMetricLearningMiner.register("pair_margin")
class PairMarginMiner(PyTorchMetricLearningMiner, miners.PairMarginMiner):
    """Wraps the `PairMarginMiner` implementation from Pytorch Metric Learning:
    (https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#pairmarginminer).

    Registered as a `PyTorchMetricLearningMiner` with name "pair_margin".
    """

    def __init__(
        self,
        pos_margin: float,
        neg_margin: float,
        use_similarity: bool = True,
        squared_distances: bool = False,
    ) -> None:

        super().__init__(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            use_similarity=use_similarity,
            squared_distances=squared_distances,
        )
