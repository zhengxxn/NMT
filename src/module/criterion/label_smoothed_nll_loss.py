import torch.nn as nn


class LabelSmoothedNLLLoss(nn.Module):
    """
    select from fairseq implementation
    """

    def __init__(self, size, padding_idx, smoothing=0.0, reduction='sum'):
        super().__init__()
        self.reduction = reduction
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing

    def forward(self, lprobs, target):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        pad_mask = target.eq(self.padding_idx)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)

        # if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1. - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss
