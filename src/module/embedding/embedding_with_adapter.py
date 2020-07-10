import torch.nn as nn
import math
from module.position_encoding.absolute_positional_encoding import PositionalEncoding
from module.adapter.feedforward_adapter_layer import FeedForwardAdapterLayer
import collections


class EmbeddingWithAdapter(nn.Module):
    def __init__(self,
                 emb_size,
                 vocab_size,
                 domain_adapter_dict,
                 dropout=0.1, max_len=5000):
        super(EmbeddingWithAdapter, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoding_layer = PositionalEncoding(input_dim=emb_size, max_len=max_len)
        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

        adapter_layers = nn.ModuleDict({})
        _adapter_layers = collections.OrderedDict()
        for domain in domain_adapter_dict.keys():
            _adapter_layers[domain] = FeedForwardAdapterLayer(input_dim=emb_size,
                                                              ff_dim=domain_adapter_dict[domain]['emb_adapt_count'],
                                                              dropout=0.1)

        adapter_layers.update(_adapter_layers)
        self.adapter_layers = adapter_layers

    def forward(self, x, target_domain):
        x = self.embedding_layer(x)
        x = x + self.adapter_layers[target_domain](x)

        x = x * math.sqrt(self.emb_size)
        x = x + self.positional_encoding_layer.forward(x)
        x = self.dropout(x)

        return x
