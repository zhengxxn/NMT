import torch.nn as nn
import copy
from module.sublayer_connection.sublayer_connection import SublayerConnection
from module.feedforward.positional_wise_feed_forward import PositionWiseFeedForward
import collections


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


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 feature_size,
                 self_attention_layer,
                 feed_forward_layer,
                 adapter_dict,
                 adapter_bottleneck_size,
                 dropout_rate):
        """

        :param feature_size: the input size of each transformer encoder layer, same as output size
        :param self_attention_layer:
        :param feed_forward_layer:
        :param adapter_dict:
        :param dropout_rate:
        """
        super().__init__()

        self.self_attention_layer = self_attention_layer  # sub layer 1
        self.feed_forward_layer = feed_forward_layer  # sub layer 2

        self.adapter_dict = adapter_dict

        self.domain_specific_sublayer_connection = None
        self.domain_specific_adapter_for_ffn = None
        self.domain_specific_adapter_for_self_attn = None
        self.domain_specific_sublayer_connection_for_adapter = None
        self.init_adapter(feature_size, dropout_rate, adapter_dict, adapter_bottleneck_size)

        self.sub_layer_connections = clones(SublayerConnection(feature_size, dropout_rate), 2)
        self.feature_size = feature_size

    def init_adapter(self, feature_size, dropout_rate, adapter_dict, adapter_bottleneck_size, ):

        # define two new sublayer_connection for each domain with domain-specific layer norm
        domain_specific_sublayer_connection = nn.ModuleDict({})
        _domain_specific_sublayer_connection = collections.OrderedDict()
        for domain in adapter_dict:
            _domain_specific_sublayer_connection[domain] = clones(SublayerConnection(feature_size, dropout_rate), 2)
        domain_specific_sublayer_connection.update(_domain_specific_sublayer_connection)
        self.domain_specific_sublayer_connection = domain_specific_sublayer_connection

        # define two adapter layer for each sub_layer
        domain_specific_adapter_for_self_attn = nn.ModuleDict({})
        _domain_specific_adapter_for_self_attn = collections.OrderedDict()
        domain_specific_adapter_for_ffn = nn.ModuleDict({})
        _domain_specific_adapter_for_ffn = collections.OrderedDict()
        domain_specific_sublayer_connection_for_adapter = nn.ModuleDict({})
        _domain_specific_sublayer_connection_for_adapter = collections.OrderedDict()
        for domain, domain_sz in zip(adapter_dict, adapter_bottleneck_size):
            _domain_specific_adapter_for_self_attn[domain] = PositionWiseFeedForward(feature_size,
                                                                                     domain_sz,
                                                                                     dropout_rate)
            _domain_specific_adapter_for_ffn[domain] = PositionWiseFeedForward(feature_size,
                                                                               domain_sz,
                                                                               dropout_rate)
            _domain_specific_sublayer_connection_for_adapter[domain] = clones(SublayerConnection(feature_size, dropout_rate), 2)

        domain_specific_adapter_for_self_attn.update(_domain_specific_adapter_for_self_attn)
        domain_specific_adapter_for_ffn.update(_domain_specific_adapter_for_ffn)
        domain_specific_sublayer_connection_for_adapter.update(_domain_specific_sublayer_connection_for_adapter)
        self.domain_specific_adapter_for_self_attn = domain_specific_adapter_for_self_attn
        self.domain_specific_adapter_for_ffn = domain_specific_adapter_for_ffn
        self.domain_specific_sublayer_connection_for_adapter = domain_specific_sublayer_connection_for_adapter

    def init_adapter_parameter(self):

        for domain in self.domain_specific_sublayer_connection.keys():
            self.domain_specific_sublayer_connection[domain][0].norm.weight.data.copy_(
                self.sub_layer_connections[0].norm.weight.data
            )
            self.domain_specific_sublayer_connection[domain][0].norm.bias.data.copy_(
                self.sub_layer_connections[0].norm.bias.data
            )
            self.domain_specific_sublayer_connection[domain][1].norm.weight.data.copy_(
                self.sub_layer_connections[1].norm.weight.data
            )
            self.domain_specific_sublayer_connection[domain][1].norm.bias.data.copy_(
                self.sub_layer_connections[1].norm.bias.data
            )

    def forward(self, x, src_mask, target_domain=None):
        # x + dropout(self_attention(Layer_norm(x)))
        # print('encoder ', target_domain)
        if target_domain is None:
            x = self.sub_layer_connections[0](x, lambda x: self.self_attention_layer(x, x, x, src_mask))
            # x + dropout(feed forward(layer_norm(x)))
            x = self.sub_layer_connections[1](x, self.feed_forward_layer)
        else:
            x = self.domain_specific_sublayer_connection[target_domain][0] \
                (x, lambda x: self.self_attention_layer(x, x, x, src_mask))

            # x = x + self.domain_specific_adapter_for_self_attn[target_domain](x)
            x = self.domain_specific_sublayer_connection_for_adapter[target_domain][0]\
                (x, self.domain_specific_adapter_for_self_attn[target_domain])

            x = self.domain_specific_sublayer_connection[target_domain][1](x, self.feed_forward_layer)

            x = self.domain_specific_sublayer_connection_for_adapter[target_domain][1]\
                (x, self.domain_specific_adapter_for_ffn[target_domain])
            # x = x + self.domain_specific_adapter_for_ffn[target_domain](x)

        # if target_domain is not None:
        #     x = self.sublayer_connection_for_adapter[target_domain](x, self.adapters[target_domain])
        # ref_adapter_list = []
        # if ref_domains is not None:
        #     for ref_domain in ref_domains:
        #         t = self.self.sublayer_connection_for_adapter[ref_domain](x, self.adapters[ref_domain])
        #         ref_adapter_list.append(t.unsqueeze(0))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, feature_size, layer: TransformerEncoderLayer, num_layers: int,
                 adapter_dict, ):
        super().__init__()

        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size)

        self.domain_specific_layer_norm = None
        self.init_adapter(feature_size, adapter_dict)

    def init_adapter(self, feature_size, adapter_dict):

        domain_specific_layer_norm = nn.ModuleDict({})
        _domain_specific_layer_norm = collections.OrderedDict()
        for domain in adapter_dict:
            _domain_specific_layer_norm[domain] = nn.LayerNorm(feature_size)
        domain_specific_layer_norm.update(_domain_specific_layer_norm)
        self.domain_specific_layer_norm = domain_specific_layer_norm

    def init_adapter_parameter(self):

        for domain in self.domain_specific_layer_norm.keys():
            self.domain_specific_layer_norm[domain].weight.data.copy_(self.layer_norm.weight.data)
            self.domain_specific_layer_norm[domain].bias.data.copy_(self.layer_norm.bias.data)

        for layer in self.layers:
            layer.init_adapter_parameter()

    def forward(self, x, src_mask, target_domain=None):
        # print('encoder ', target_domain)
        layers_adapter_output = []
        for layer in self.layers:
            x = layer(x, src_mask, target_domain)
            layers_adapter_output.append(x)
            # layers_ref_adapter_list.append(ref_adapter_list)

        if target_domain is None:
            memory = self.layer_norm(x)
        else:
            memory = self.domain_specific_layer_norm[target_domain](x)

        return {'memory': memory,
                'adapter_output': layers_adapter_output}
