import torch.nn as nn
import copy
from module.sublayer_connection.sublayer_connection import SublayerConnection
from module.adapter.stacked_adapter import StackedAdapter


# layer norm:
# sublayer connection:  A residual connection followed by a layer norm.
#   x + self.dropout(sublayer(self.norm(x)))


def clones(sub_module, num_layers):
    """
    Produce N identical layers
    :param sub_module:
    :param num_layers:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(sub_module) for _ in range(num_layers)])


class TransformerEncoderLayerWithSynthesizerAdapter(nn.Module):
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
        x = self.sub_layer_connections[0](x, lambda x: self.self_attention_layer(x, x, x, src_mask, target_domain))
        # x + dropout(feed forward(layer_norm(x)))
        x = self.sub_layer_connections[1](x, self.feed_forward_layer)

        if target_domain is not None:
            x = self.adapters(x, target_domain)

        return x


class TransformerEncoderWithSynthesizerAdapter(nn.Module):
    def __init__(self, feature_size, layer: TransformerEncoderLayerWithSynthesizerAdapter, num_layers: int):
        super().__init__()

        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size)

    def forward(self, x, src_mask, target_domain=None):
        layers_adapter_output = []
        for layer in self.layers:
            x = layer(x, src_mask, target_domain)
            layers_adapter_output.append(x)
            # layers_ref_adapter_list.append(ref_adapter_list)

        return {'memory': self.layer_norm(x),
                'adapter_output': layers_adapter_output}
