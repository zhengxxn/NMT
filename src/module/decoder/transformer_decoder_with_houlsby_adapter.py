import torch.nn as nn
import copy
from module.sublayer_connection.sublayer_connection import SublayerConnection
from module.sublayer_connection.sublayer_connection_with_cache import SublayerConnectionWithCache
from module.feedforward.positional_wise_feed_forward import PositionWiseFeedForward
import collections


def clones(sub_module, num_layers):
    """
    Produce N identical layers
    :param sub_module:
    :param num_layers:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(sub_module) for _ in range(num_layers)])


class TransformerDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self,
                 feature_size,
                 self_attention_layer,
                 cross_attention_layer,
                 feed_forward_layer,
                 adapter_dict,
                 adapter_bottleneck_size,
                 dropout_rate):
        super(TransformerDecoderLayer, self).__init__()
        self.feature_size = feature_size

        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_layer = feed_forward_layer

        # adapters = nn.ModuleDict({})
        # sublayer_connection_for_adapter = nn.ModuleDict({})
        # _adapters = collections.OrderedDict()
        # _sublayer_connection_for_adapter = collections.OrderedDict()
        # for domain, size in zip(adapter_dict, adapter_bottleneck_size):
        #     _adapters[domain] = PositionWiseFeedForward(input_dim=feature_size,
        #                                                 ff_dim=size,
        #                                                 dropout=dropout_rate)
        #     _sublayer_connection_for_adapter[domain] = SublayerConnection(feature_size, dropout_rate)
        # adapters.update(_adapters)
        # sublayer_connection_for_adapter.update(_sublayer_connection_for_adapter)
        #
        # self.adapters = adapters
        # self.sublayer_connection_for_adapter = sublayer_connection_for_adapter

        self.sublayer_with_cache = clones(SublayerConnectionWithCache(feature_size, dropout_rate), 2)
        self.sublayer = SublayerConnection(feature_size, dropout_rate)

        self.domain_specific_sublayer_with_cache = None
        self.domain_specific_sublayer = None
        self.init_adapter(feature_size, dropout_rate, adapter_dict)

    def init_adapter(self, feature_size, dropout_rate, adapter_dict):

        domain_specific_sublayer_with_cache = nn.ModuleDict({})
        _domain_specific_sublayer_with_cache = collections.OrderedDict()
        for domain in adapter_dict:
            _domain_specific_sublayer_with_cache[domain] = clones(SublayerConnectionWithCache(feature_size, dropout_rate), 2)
        domain_specific_sublayer_with_cache.update(_domain_specific_sublayer_with_cache)
        self.domain_specific_sublayer_with_cache = domain_specific_sublayer_with_cache

        domain_specific_sublayer = nn.ModuleDict({})
        _domain_specific_sublayer = collections.OrderedDict()
        for domain in adapter_dict:
            _domain_specific_sublayer[domain] = SublayerConnection(feature_size, dropout_rate)
        domain_specific_sublayer.update(_domain_specific_sublayer)
        self.domain_specific_sublayer = domain_specific_sublayer

    def init_adapter_parameter(self):

        for domain in self.domain_specific_sublayer_with_cache.keys():
            self.domain_specific_sublayer_with_cache[domain][0].norm.weight.data.copy_(
                self.sublayer_with_cache[0].norm.weight.data
            )
            self.domain_specific_sublayer_with_cache[domain][0].norm.bias.data.copy_(
                self.sublayer_with_cache[0].norm.bias.data
            )
            self.domain_specific_sublayer_with_cache[domain][1].norm.weight.data.copy_(
                self.sublayer_with_cache[1].norm.weight.data
            )
            self.domain_specific_sublayer_with_cache[domain][1].norm.bias.data.copy_(
                self.sublayer_with_cache[1].norm.bias.data
            )
            self.domain_specific_sublayer[domain].norm.weight.data.copy_(
                self.sublayer.norm.weight.data
            )
            self.domain_specific_sublayer[domain].norm.bias.data.copy_(
                self.sublayer.norm.bias.data
            )

    def forward(self,
                x,
                memory,
                src_mask,
                trg_mask,
                enc_attn_cache=None,
                self_attn_cache=None,
                target_domain=None, ):

        m = memory

        if target_domain is not None:

            x, new_self_attn_cache = self.sublayer_with_cache[0] \
                (x, lambda x: self.self_attention_layer(query=x,
                                                        key=x,
                                                        value=x,
                                                        mask=trg_mask,
                                                        enc_attn_cache=None,
                                                        self_attn_cache=self_attn_cache,
                                                        is_self_attn=True))
            x, new_enc_attn_cache = self.sublayer_with_cache[1] \
                (x, lambda x: self.cross_attention_layer(query=x,
                                                         key=m,
                                                         value=m,
                                                         mask=src_mask,
                                                         enc_attn_cache=enc_attn_cache,
                                                         self_attn_cache=None,
                                                         is_self_attn=False))

            x = self.sublayer(x, self.feed_forward_layer)

        else:
            x, new_self_attn_cache = self.domain_specific_sublayer_with_cache[target_domain][0] \
                (x, lambda x: self.self_attention_layer(query=x,
                                                        key=x,
                                                        value=x,
                                                        mask=trg_mask,
                                                        enc_attn_cache=None,
                                                        self_attn_cache=self_attn_cache,
                                                        is_self_attn=True))
            x, new_enc_attn_cache = self.domain_specific_sublayer_with_cache[target_domain][1] \
                (x, lambda x: self.cross_attention_layer(query=x,
                                                         key=m,
                                                         value=m,
                                                         mask=src_mask,
                                                         enc_attn_cache=enc_attn_cache,
                                                         self_attn_cache=None,
                                                         is_self_attn=False))

            x = self.domain_specific_sublayer[target_domain](x, self.feed_forward_layer)

        # if target_domain is not None:
        #     x = self.sublayer_connection_for_adapter[target_domain](x, self.adapters[target_domain])
        # ref_adapter_list = []
        # if ref_domain_list is not None:
        #     for ref_domain in ref_domain_list:
        #         ref_adapter_result = self.sublayer_connection_for_adapter[ref_domain](x, self.adapters[ref_domain])
        #         ref_adapter_list.append(ref_adapter_result)

        return x, new_enc_attn_cache, new_self_attn_cache


class TransformerDecoder(nn.Module):

    def __init__(self,
                 feature_size,
                 layer,
                 num_layers,
                 adapter_dict):

        super(TransformerDecoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.num_layers = num_layers

        self.layer_norm_adapters = None
        self.init_adapter(feature_size, adapter_dict)

    def init_adapter(self, feature_size, adapter_dict):

        layer_norm_adapters = nn.ModuleDict({})
        _adapters = collections.OrderedDict()
        for domain in adapter_dict:
            _adapters[domain] = nn.LayerNorm(feature_size)
        layer_norm_adapters.update(_adapters)
        self.layer_norm_adapters = layer_norm_adapters

    def init_adapter_parameter(self):

        for domain in self.layer_norm_adapters.keys():
            self.layer_norm_adapters[domain].weight.data.copy_(self.layer_norm.weight.data)
            self.layer_norm_adapters[domain].bias.data.copy_(self.layer_norm.bias.data)

        for layer in self.layers:
            layer.init_adapter_parameter()

    def forward(self, x,
                memory,
                src_mask,
                trg_mask,
                enc_attn_cache_list=None,
                self_attn_cache_list=None,
                target_domain=None, ):

        if self_attn_cache_list is None:
            self_attn_cache_list = [None] * self.num_layers

        if enc_attn_cache_list is None:
            enc_attn_cache_list = [None] * self.num_layers

        new_enc_attn_cache_list = []
        new_self_attn_cache_list = []

        # layers_ref_adapter_list = []
        layers_adapter_output = []

        for i, layer in enumerate(self.layers):
            x, new_enc_attn_cache, new_self_attn_cache = layer(x,
                                                               memory,
                                                               src_mask,
                                                               trg_mask,
                                                               enc_attn_cache_list[i],
                                                               self_attn_cache_list[i],
                                                               target_domain,
                                                               )
            layers_adapter_output.append(x)
            # layers_ref_adapter_list.append(ref_adapter_list)
            new_self_attn_cache_list = new_self_attn_cache_list + [new_self_attn_cache]
            new_enc_attn_cache_list = new_enc_attn_cache_list + [new_enc_attn_cache]

        if target_domain is None:
            norm_x = self.layer_norm(x)
        else:
            norm_x = self.layer_norm_adapters[target_domain](x)

        return norm_x, layers_adapter_output, new_enc_attn_cache_list, new_self_attn_cache_list
