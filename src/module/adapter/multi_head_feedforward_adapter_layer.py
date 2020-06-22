import torch.nn as nn
import torch


class MultiHeadFeedForwardAdapterLayer(nn.Module):
    """Implements FFN equation."""

    def __init__(self,
                 input_dim: int,
                 memory_count: int,
                 head_num: int,
                 dropout: float = 0.1) -> None:

        super().__init__()
        self.w_1 = nn.Linear(input_dim, memory_count, bias=False)  # w_1.weight [memory count, input_dim]
        self.w_2 = nn.Linear(memory_count, input_dim)  # w_2.weight [input_dim, memory count]
        self.dropout = nn.Dropout(dropout)
        self.head_num = head_num
        self.head_dim = input_dim // head_num
        self.memory_count = memory_count
        self.memory_score_bias = nn.Parameter(torch.randn((self.head_num, 1, self.memory_count), requires_grad=True))

        # self.query_layer = nn.Linear(input_dim, input_dim)
        # self.final_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch_size = x.size(0)
        seq_len = x.size(1)
        global_key_vector = self.w_1.weight
        global_value_vector = self.w_2.weight

        # x = self.query_layer(x)
        query = x.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        # [batch size, seq len, input dim] -> [batch size, head num, seq len, head dim]

        key = global_key_vector.view(self.memory_count, self.head_num, self.head_dim).transpose(0, 1)
        # [mem count, h_n * h_d] -> [head num, mem count, head dim]

        # score = torch.matmul(query, key.transpose(-1, -2))  # [batch, head num, seq len, mem count]
        score = torch.einsum('bij,abcj->abci', (key, query))

        # a: batch size, b: head num, c: seq len, i: mem count, j: head dim
        score = score + self.memory_score_bias

        score = self.dropout(torch.relu(score))

        value = global_value_vector.view(self.memory_count, self.head_num, self.head_dim).transpose(0, 1)
        # [mem count, h_n * h_d] -> [head num, mem count, head dim]

        # multi_head_output = torch.matmul(score, value)  # [batch size, head num, seq len, head_dim]
        multi_head_output = torch.einsum('bij,abci->abcj', (value, score))
        # a: batch size, b: head num, c: seq len, i: mem count, j: head dim

        multi_head_output = multi_head_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # [batch size, seq len, input dim]

        multi_head_output = multi_head_output + self.w_2.bias

        return multi_head_output
        # return self.final_linear(multi_head_output)
