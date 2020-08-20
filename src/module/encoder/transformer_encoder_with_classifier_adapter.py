import torch.nn as nn
import torch
import copy
from module.sublayer_connection.sublayer_connection import SublayerConnection
from module.adapter.stacked_adapter import StackedAdapter


def clones(sub_module, num_layers):
    """
    Produce N identical layers
    :param sub_module:
    :param num_layers:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(sub_module) for _ in range(num_layers)])


class TransformerEncoderLayerWithClassifierAdapter(nn.Module):
    def __init__(self,
                 feature_size,
                 self_attention_layer,
                 feed_forward_layer,
                 domain_adapter_dict,
                 dropout_rate):
        """

        :param feature_size: the input size of each transformer encoder layer, same as output size
        :param self_attention_layer:
        :param feed_forward_layer:
        :param domain_adapter_dict:
        :param dropout_rate:
        """
        super().__init__()

        self.self_attention_layer = self_attention_layer  # sub layer 1
        self.feed_forward_layer = feed_forward_layer  # sub layer 2

        self.domain_adapter_dict = domain_adapter_dict

        # make adapter
        self.adapters = StackedAdapter(domain_adapter_dict=domain_adapter_dict,
                                       feature_size=feature_size,
                                       dropout_rate=dropout_rate)

        self.sub_layer_connections = clones(SublayerConnection(feature_size, dropout_rate), 2)
        self.feature_size = feature_size

    def forward(self, x, src_mask, target_domain=None):
        # x + dropout(self_attention(Layer_norm(x)))
        x = self.sub_layer_connections[0](x, lambda x: self.self_attention_layer(x, x, x, src_mask))
        # x + dropout(feed forward(layer_norm(x)))
        x = self.sub_layer_connections[1](x, self.feed_forward_layer)

        if target_domain is not None:
            x = self.adapters(x, target_domain)

        return x


class TransformerEncoderWithClassifierAdapter(nn.Module):
    def __init__(self,
                 feature_size: int,
                 layer: TransformerEncoderLayerWithClassifierAdapter,
                 num_layers: int,
                 domain_label_dict: {}):

        super().__init__()

        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.domain_label_dict = domain_label_dict

    def forward(self, x, src_mask, target_domain=None):
        if not isinstance(target_domain, str):
            # target_domain is domain label, [0, 1, 2, 3, 4]
            target_domain_idx_dict = {}

            for domain in self.domain_label_dict.keys():
                domain_label = self.domain_label_dict[domain]
                idx_equal_label: torch.Tensor = (target_domain == domain_label)
                if idx_equal_label.sum().item() == 0:
                    continue
                else:
                    idx_equal_label: torch.Tensor = idx_equal_label.nonzero().flatten()
                target_domain_idx_dict[domain] = idx_equal_label

            target_domain = target_domain_idx_dict
            print(target_domain)

        layers_adapter_output = []
        for layer in self.layers:
            x = layer(x, src_mask, target_domain)
            layers_adapter_output.append(x)
            # layers_ref_adapter_list.append(ref_adapter_list)

        return {'memory': self.layer_norm(x),
                'adapter_output': layers_adapter_output}
