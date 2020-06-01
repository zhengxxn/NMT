import torch.nn as nn
import math
from module.position_encoding.absolute_positional_encoding import PositionalEncoding


class Embeddings(nn.Module):
    def __init__(self, emb_size, vocab_size, dropout=0.1, max_len=5000):
        super(Embeddings, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoding_layer = PositionalEncoding(input_dim=emb_size, max_len=max_len)
        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, x):
        x = self.embedding_layer(x) * math.sqrt(self.emb_size)
        x = x + self.positional_encoding_layer.forward(x)
        x = self.dropout(x)
        return x
