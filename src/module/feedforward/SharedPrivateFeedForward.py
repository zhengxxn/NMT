import torch.nn as nn
import torch
from .positional_wise_feed_forward import PositionWiseFeedForward


class SharedPrivateFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self,
                 input_dim: int,
                 ffn_dict: dict,
                 dropout: float = 0.1) -> None:
        super().__init__()

        ffn_layers = nn.ModuleDict({})
        sublayer_connection_for_adapter = nn.ModuleDict({})

        for domain in ffn_dict.keys():

            if domain_adapter_dict[domain].get('adapter_type', None) == 'memory':

                adapter_layers[domain] = MemoryAdapterLayer(input_dim=feature_size,
                                                            ff_dim=domain_adapter_dict[domain]['memory_count'],
                                                            dropout=dropout_rate)

            elif domain_adapter_dict[domain].get('adapter_type', None) == 'domain_mix':

                domain_mix_layers[domain] = AdapterMixLayer(used_adapters=domain_adapter_dict[domain]['used_adapters'],
                                                            feature_size=feature_size,
                                                            dropout_rate=dropout_rate,
                                                            classifier_dict=domain_adapter_dict[domain]['classifier_dict'])

            elif domain_adapter_dict[domain].get('is_generate', False):
                adapter_layers[domain] = ParameterGeneratorForAdapter(adapter_dict=domain_adapter_dict,
                                                                      used_adapters=domain_adapter_dict[domain]['used_adapters'],
                                                                      generate_dim=domain_adapter_dict[domain]['generate_dim'],
                                                                      feature_size=feature_size,
                                                                      bottleneck_dim=domain_adapter_dict[domain]['bottle_neck_dim'],
                                                                      dropout_rate=dropout_rate,
                                                                      linear_transform=domain_adapter_dict[domain].get('linear_transform', False))
                self.adapter_types[domain] = 'generate'
                sublayer_connection_for_adapter[domain] = SublayerConnection(size=feature_size,
                                                                             dropout=dropout_rate)

            else:
                adapter_layers[domain] = FeedForwardAdapterLayer(input_dim=feature_size,
                                                                 ff_dim=domain_adapter_dict[domain]['memory_count'],
                                                                 dropout=dropout_rate,
                                                                 activation_statistic=domain_adapter_dict[domain]['activation_statistic'] if 'activation_statistic' in domain_adapter_dict[domain]
                                                                 else False)
                self.adapter_types[domain] = 'simple'

                sublayer_connection_for_adapter[domain] = SublayerConnection(size=feature_size,
                                                                             dropout=dropout_rate)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
