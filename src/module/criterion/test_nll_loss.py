import torch.nn as nn
import torch


class TestNLLLoss(nn.Module):

    def __init__(self, size, padding_idx):
        super().__init__()
        self.size = size
        self.padding_idx = padding_idx
        self.criterion = nn.NLLLoss(reduction='none', ignore_index=padding_idx)

    def forward(self, lprobs, target):
        # lprobs: [b, s, n]
        # target: [b, s]
        batch_size, seq_len = lprobs.size(0), lprobs.size(1)
        length = (target != self.padding_idx)
        length = length.sum(dim=-1).float()

        lprobs = lprobs.view(batch_size * seq_len, -1)
        target = target.view(batch_size * seq_len)
        loss = self.criterion(lprobs, target)
        loss = loss.view(batch_size, seq_len)
        # avg_sample_loss = torch.sum(loss, dim=-1)
        # avg_sample_loss = avg_sample_loss / length
        return loss
        # return avg_sample_loss


if __name__ == "__main__":
    m = nn.LogSoftmax(dim=-1)
    import torch
    inp = torch.randn(3, 6, 5, requires_grad=True)
    target = torch.randint(0, 5, (3, 6))
    print(target)
    criterion = TestNLLLoss(size=5, padding_idx=0)
    out = m(inp)
    loss = criterion(out, target).tolist()
    loss = [[round(val, 2) for val in batch] for batch in loss]
    print(loss)
