import torch.nn as nn


class TestNLLLoss(nn.Module):

    def __init__(self, size, padding_idx):
        super().__init__()
        self.size = size
        self.padding_idx = padding_idx
        self.criterion = nn.NLLLoss(reduction='none', ignore_index=padding_idx)

    def forward(self, lprobs, target):
        loss = self.criterion(lprobs, target)
        length = (target != self.padding_idx)
        length = length.sum(dim=-1).float()
        avg_sample_loss = torch.sum(loss, dim=-1)
        avg_sample_loss = avg_sample_loss / length
        return avg_sample_loss


if __name__ == "__main__":
    m = nn.LogSoftmax(dim=-1)
    import torch
    inp = torch.randn(3, 5, 5, requires_grad=True)
    target = torch.randint(0, 4, (3, 5))
    criterion = TestNLLLoss(size=5, padding_idx=0)
    loss = criterion(m(inp), target)
    print(loss)

