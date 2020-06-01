import torch.nn as nn
import torch
import math


class Embeddings(nn.Module):
    def __init__(self,
                 emb_size,
                 vocab_size,
                 dropout,
                 max_len=5000,
                 linear_combination=False,
                 ):

        super(Embeddings, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

        self.linear_combination = linear_combination
        if self.linear_combination is False:
            self.position_emb_size = int(emb_size / 2)
        else:
            self.position_emb_size = emb_size
            self.combination_function = nn.Sequential(nn.Linear(emb_size * 2, emb_size), nn.Tanh())

        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_len, self.position_emb_size, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.position_emb_size, 2).float() * -(math.log(10000.0) / self.position_emb_size))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x, inner_position_index, outer_position_index):
        """

        :param x: [batch size, seq len]
        :param inner_position_index: [batch size, seq len]
        :param outer_position_index: [batch size, seq len]
        :return:
        """

        # word_embedding
        x = self.embedding_layer(x) * math.sqrt(self.emb_size)

        # position encoding
        batch_size = inner_position_index.size(0)
        seq_len = inner_position_index.size(1)
        inner_position_index = inner_position_index.view(batch_size * seq_len).unsqueeze(1)\
            .expand(batch_size * seq_len, self.position_emb_size).unsqueeze(0)  # [1, batch size * seq len, emb size]
        inner_position_encoding = torch.gather(self.positional_encoding,
                                               dim=1,
                                               index=inner_position_index)
        inner_position_encoding = inner_position_encoding.view(batch_size, seq_len, self.position_emb_size)

        outer_position_index = outer_position_index.view(batch_size * seq_len).unsqueeze(1)\
            .expand(batch_size * seq_len, self.position_emb_size).unsqueeze(0)  # [1, batch size * seq len, emb size]
        outer_position_encoding = torch.gather(self.positional_encoding,
                                               dim=1,
                                               index=outer_position_index)
        outer_position_encoding = outer_position_encoding.view(batch_size, seq_len, self.position_emb_size)

        position_encoding = torch.cat((inner_position_encoding, outer_position_encoding), dim=-1)
        if self.linear_combination:
            position_encoding = self.combination_function(position_encoding)

        # sum
        x = x + position_encoding
        return self.dropout(x)
