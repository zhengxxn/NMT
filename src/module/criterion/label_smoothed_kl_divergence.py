import torch.nn as nn
import torch


class LabelSmoothedKLDivergence(nn.Module):
    """
    Implement label smoothing.
    suppose the target is [1], size is 5, padding_idx = 0,
    so the smooth target is [0, 0.9, 0.0333, 0.0333, 0.0333]
    """

    def __init__(self, size, padding_idx, smoothing=0.0, reduction='sum'):
        super(LabelSmoothedKLDivergence, self).__init__()

        self.reduction=reduction
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, input, target):
        assert input.size(1) == self.size
        true_dist = input.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(input, true_dist.clone().detach())