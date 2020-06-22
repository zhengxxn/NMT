import torch.nn as nn
from module.adapter.multi_head_feedforward_adapter_layer import MultiHeadFeedForwardAdapterLayer
from module.sublayer_connection.sublayer_connection import SublayerConnection
import collections


class StackedMultiHeadAdapter(nn.Module):
    def __init__(self, domain_adapter_dict, feature_size, dropout_rate):
        super(StackedMultiHeadAdapter, self).__init__()

        adapter_layers = nn.ModuleDict({})
        _adapter_layers = collections.OrderedDict()
        sublayer_connection_for_adapter = nn.ModuleDict({})
        _sublayer_connection_for_adapter = collections.OrderedDict()

        for domain in domain_adapter_dict.keys():
            _adapter_layers[domain] = MultiHeadFeedForwardAdapterLayer(input_dim=feature_size,
                                                                       memory_count=domain_adapter_dict[domain]['memory_count'],
                                                                       head_num=domain_adapter_dict[domain]['head_num'],
                                                                       dropout=dropout_rate)
            _sublayer_connection_for_adapter[domain] = SublayerConnection(size=feature_size,
                                                                          dropout=dropout_rate)

        adapter_layers.update(_adapter_layers)
        sublayer_connection_for_adapter.update(_sublayer_connection_for_adapter)

        self.adapter_layers = adapter_layers
        self.sublayer_connection_for_adapter = sublayer_connection_for_adapter

    def forward(self, x, target_domain):

        return self.sublayer_connection_for_adapter[target_domain](x, self.adapter_layers[target_domain])
