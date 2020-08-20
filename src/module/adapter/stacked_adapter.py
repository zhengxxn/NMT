import torch.nn as nn
import torch
from module.adapter.feedforward_adapter_layer import FeedForwardAdapterLayer
from module.sublayer_connection.sublayer_connection import SublayerConnection


class StackedAdapter(nn.Module):

    def __init__(self,
                 domain_adapter_dict,
                 feature_size,
                 dropout_rate,):
        super(StackedAdapter, self).__init__()

        adapter_layers = nn.ModuleDict({})
        sublayer_connection_for_adapter = nn.ModuleDict({})

        for domain in domain_adapter_dict.keys():
            adapter_layers[domain] = FeedForwardAdapterLayer(input_dim=feature_size,
                                                             ff_dim=domain_adapter_dict[domain]['memory_count'],
                                                             dropout=dropout_rate)
            sublayer_connection_for_adapter[domain] = SublayerConnection(size=feature_size,
                                                                         dropout=dropout_rate)

        self.adapter_layers = adapter_layers
        self.sublayer_connection_for_adapter = sublayer_connection_for_adapter

    def forward(self, x, target_domain):

        if isinstance(target_domain, str):
            # use for training
            if target_domain == 'news':
                return x
            else:
                return self.sublayer_connection_for_adapter[target_domain](x, self.adapter_layers[target_domain])

        else:
            # the target_domain is a dict, the key is domain and value is idx of batch
            # {
            # 'news' : [0, 1, 2, 3]
            # 'iwslt' : [4, 5]
            # 'laws' : [6]
            # }
            assert isinstance(target_domain, dict)

            output = torch.zeros_like(x, requires_grad=True)

            for domain in target_domain.keys():

                batch_domain_idx = target_domain[domain]
                domain_inp = torch.index_select(x, dim=0, index=batch_domain_idx)

                if domain == 'news':
                    domain_out = domain_inp
                else:
                    domain_out = self.sublayer_connection_for_adapter[domain](domain_inp, self.adapter_layers[domain])

                output = output.index_copy(dim=0, index=batch_domain_idx, source=domain_out)

            return output
