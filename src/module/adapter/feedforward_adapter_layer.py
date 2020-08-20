import torch.nn as nn
import torch


class FeedForwardAdapterLayer(nn.Module):
    """Implements FFN equation."""

    def __init__(self,
                 input_dim: int,
                 ff_dim: int,
                 dropout: float = 0.1,) -> None:

        super().__init__()
        self.w_1 = nn.Linear(input_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                external_bias: torch.Tensor = None,
                external_value_bias: torch.Tensor = None) -> torch.Tensor:
        """

        :param external_value_bias:
        :param x: [batch size, seq len, hid dim]
        :param external_bias: [batch size, seq len, ffn size] or [1, ffn size]
        :return:
        """

        s_1 = self.w_1(x)
        if external_bias is not None:
            s_1 = s_1 + external_bias
        s_1 = self.dropout(torch.relu(s_1))

        s_2 = self.w_2(s_1)
        if external_value_bias is not None:
            s_2 = s_2 + external_value_bias

        return s_2
