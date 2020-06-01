import torch.nn as nn
import torch
import math


class PositionalEncoding(torch.nn.Module):
    """Implement the Positional Encoding function."""

    def __init__(self, input_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_len, input_dim, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * -(math.log(10000.0) / input_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        return self.positional_encoding[:, :x.size(1)]
        # return self.dropout(x)
