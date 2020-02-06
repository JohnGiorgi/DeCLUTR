import torch
from torch import nn


class NPairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural
    Information Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective

    Implementation adapted from: https://github.com/leeesangwon/PyTorch-Image-Retrieval
    """

    def __init__(self, l2_reg=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchors, positives):
        negatives = self.get_negatives(positives)

        n_pair_loss = self.n_pair_loss(anchors, positives, negatives)
        l2_loss = self.l2_reg * self.l2_loss(anchors, positives)
        loss = n_pair_loss + l2_loss

        return loss

    @staticmethod
    def get_negatives(positives):
        """
        """
        # TODO (John): It would be much better if this was vectorized
        negatives = []

        batch_size, embedding_dim = positives.size()
        indices = torch.tensor(list(range(batch_size)))

        for idx in indices:
            n_negatives_mask = (idx != indices).unsqueeze(-1).expand_as(positives).bool().to(positives.device)
            n_negatives = torch.masked_select(positives, n_negatives_mask).reshape(batch_size - 1, embedding_dim)
            negatives.append(n_negatives)
        negatives = torch.stack(negatives)  # (n, n-1, embedding_size)

        return negatives

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]
