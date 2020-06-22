import torch.nn as nn
import torch


class FeedForwardAdapterLayer(nn.Module):
    """Implements FFN equation."""

    def __init__(self,
                 input_dim: int,
                 ff_dim: int,
                 dropout: float = 0.1) -> None:

        super().__init__()
        self.w_1 = nn.Linear(input_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
