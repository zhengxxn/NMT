import torch.nn as nn
import math
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import SublayerConnection, \
    PositionwiseFeedForward
import collections


class Embeddings(nn.Module):
    def __init__(self,
                 emb_size,
                 vocab_size,
                 adapter_dict,
                 adapter_bottleneck_size,
                 dropout_rate,
                 ):

        super(Embeddings, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

        adapters = nn.ModuleDict({})
        sublayer_connection_for_adapter = nn.ModuleDict({})
        _adapters = collections.OrderedDict()
        _sublayer_connection_for_adapter = collections.OrderedDict()
        for domain in adapter_dict:
            _adapters[domain] = PositionwiseFeedForward(input_dim=emb_size,
                                                        ff_dim=adapter_bottleneck_size,
                                                        dropout=dropout_rate)
            _sublayer_connection_for_adapter[domain] = SublayerConnection(emb_size, dropout_rate)
        adapters.update(_adapters)
        sublayer_connection_for_adapter.update(_sublayer_connection_for_adapter)
        self.adapters = adapters
        self.sublayer_connection_for_adapter = sublayer_connection_for_adapter

        self.current_domain = None

    def forward(self, x):

        x = self.embedding_layer(x)

        if self.current_domain is not None:
            x = self.sublayer_connection_for_adapter[self.current_domain](x, self.adapters[self.current_domain])
        return x * math.sqrt(self.emb_size)
