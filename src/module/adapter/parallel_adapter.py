import torch.nn as nn
import torch
from module.adapter.feedforward_adapter_layer import FeedForwardAdapterLayer
import collections


class ParallelAdapter(nn.Module):
    def __init__(self, domain_adapter_dict, feature_size, dropout_rate):
        super(ParallelAdapter, self).__init__()

        adapter_layers = nn.ModuleDict({})
        _adapter_layers = collections.OrderedDict()

        for domain in domain_adapter_dict.keys():
            _adapter_layers[domain] = FeedForwardAdapterLayer(input_dim=feature_size,
                                                              ff_dim=domain_adapter_dict[domain]['memory_count'],
                                                              dropout=dropout_rate)
        adapter_layers.update(_adapter_layers)
        self.adapter_layers = adapter_layers

        self.adapter_combine = nn.Linear(feature_size, len(domain_adapter_dict.keys())+1, bias=False)

    def forward(self, x, original_ffn, target_domain):
        # [batch size, seq len, hid dim]
        original_ffn_value = original_ffn(x)

        # [batch size, seq len, hid dim]
        target_domain_adapter_value = self.adapter_layers[target_domain](x)

        # [batch size, seq len, hid dim, 2]
        concat_value = torch.cat([original_ffn_value.unsqueeze(-1), target_domain_adapter_value.unsqueeze(-1)], -1)

        # [batch size, seq len, 2, 1]
        domain_score = torch.softmax(self.adapter_combine(x), dim=-1).unsqueeze(-1)

        # [batch size, seq len, hid dim]
        global_memory = torch.matmul(concat_value, domain_score).squeeze(-1)

        return global_memory
