import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, emb_size, vocab_size, dropout):
        super(Embeddings, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.emb_size = emb_size

    def forward(self, x):
        return self.dropout(self.embedding_layer(x) * math.sqrt(self.emb_size))
