import torch.nn as nn
import torch


class FeedForwardAdapterLayer(nn.Module):
    """Implements FFN equation."""

    def __init__(self,
                 input_dim: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 activation_statistic: bool = False) -> None:

        super().__init__()

        self.w_1 = nn.Linear(input_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.memory_dim = ff_dim

        self.activation_statistic = activation_statistic
        if self.activation_statistic:
            self.activation = torch.zeros(ff_dim)
            self.num_tokens = 0

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

        if self.activation_statistic:
            # for simplification, batch size is set to 1
            # [batch size, seq len, ff_dim]
            activation = s_1.view(-1, s_1.size(-1)).sum(dim=0)
            self.activation = self.activation + activation.cpu()
            self.num_tokens = self.num_tokens + activation.size(0)

        s_2 = self.w_2(s_1)
        if external_value_bias is not None:
            s_2 = s_2 + external_value_bias

        return s_2
