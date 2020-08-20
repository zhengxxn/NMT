import torch.nn as nn
import torch


class SingleHeadFeedForwardAdapterLayer(nn.Module):
    """
    Implements FFN equation as a Adapter Layer to Transformer
    """

    def __init__(self,
                 input_dim: int,
                 memory_count: int,
                 dropout: float = 0.1) -> None:
        super().__init__()

        self.w_1 = nn.Linear(input_dim, memory_count, bias=False)  # w_1.weight [memory count, input_dim]
        self.w_2 = nn.Linear(memory_count, input_dim)  # w_2.weight [input_dim, memory count]
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim

        self.memory_count = memory_count
        self.memory_score_bias = nn.Parameter(torch.randn((1, self.memory_count), requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        global_key_vector = self.w_1.weight
        global_value_vector = self.w_2.weight

        query = x  # [batch size, seq len, input dim]

        key = global_key_vector  # [mem count, input_dim]

        # [batch, seq len, mem count]
        score = torch.einsum('ij, abj->abi', (key, query))

        # a: batch size, b: head num, c: seq len, i: mem count, j: head dim
        score = score + self.memory_score_bias
        score = self.dropout(torch.relu(score))

        value = global_value_vector.transpose(0, 1)  # [mem count, input dim]

        multi_head_output = torch.einsum('ij,abi->abj', (value, score))   # [batch size, seq len, head_dim]
        # a: batch size, b: head num, c: seq len, i: mem count, j: head dim

        multi_head_output = multi_head_output + self.w_2.bias

        return multi_head_output
