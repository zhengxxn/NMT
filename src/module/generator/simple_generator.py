import torch
import torch.nn as nn


class SimpleGenerator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, feature_size, vocab_size, bias=False):

        super(SimpleGenerator, self).__init__()
        self.proj = nn.Linear(feature_size, vocab_size, bias=bias)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
