import torch.nn as nn
import torch


class DenseFFNBias(nn.Module):
    """
    This bias depends on each token.
    and the Transformation is low rank
    """

    def __init__(self, input_dim: int, ff_dim: int, rank: int) -> None:
        super().__init__()

        self.w_1 = nn.Linear(input_dim, rank)
        self.w_2 = nn.Linear(rank, ff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: [batch size, seq len, input dim]
        :return:
        """

        s_1 = self.w_1(x)  # [batch size, seq len, rank]
        s_2 = self.w_2(s_1)  # [batch size, seq len, ff_dim]
        return s_2


class RandomFFNBias(nn.Module):
    """
    This bias is same for all sentence in the specific domain
    """

    def __init__(self, ff_dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.randn(1, ff_dim, requires_grad=True))

    def forward(self, x) -> torch.Tensor:
        return self.bias
