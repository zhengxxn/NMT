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

        self.sublayer_with_cache = clones(SublayerConnectionWithCache(feature_size, dropout_rate), 2)
        self.sublayer = SublayerConnection(feature_size, dropout_rate)

        self.domain_specific_sublayer_with_cache = None
        self.domain_specific_sublayer = None
        self.domain_specific_adapter_for_ffn = None
        self.domain_specific_adapter_for_self_attn = None
        self.domain_specific_adapter_for_cross_attn = None
        self.init_adapter(feature_size, dropout_rate, adapter_dict, adapter_bottleneck_size)

    def init_adapter(self, feature_size, dropout_rate, adapter_dict, adapter_bottleneck_size):

        domain_specific_sublayer_with_cache = nn.ModuleDict({})
        _domain_specific_sublayer_with_cache = collections.OrderedDict()
        for domain in adapter_dict:
            _domain_specific_sublayer_with_cache[domain] = clones(
                SublayerConnectionWithCache(feature_size, dropout_rate), 2)
        domain_specific_sublayer_with_cache.update(_domain_specific_sublayer_with_cache)
        self.domain_specific_sublayer_with_cache = domain_specific_sublayer_with_cache

        domain_specific_sublayer = nn.ModuleDict({})
        _domain_specific_sublayer = collections.OrderedDict()
        for domain in adapter_dict:
            _domain_specific_sublayer[domain] = SublayerConnection(feature_size, dropout_rate)
        domain_specific_sublayer.update(_domain_specific_sublayer)
        self.domain_specific_sublayer = domain_specific_sublayer

        # define three adapter for each sub layer

        domain_specific_adapter_for_self_attn = nn.ModuleDict({})
        _domain_specific_adapter_for_self_attn = collections.OrderedDict()
        domain_specific_adapter_for_cross_attn = nn.ModuleDict({})
        _domain_specific_adapter_for_cross_attn = collections.OrderedDict()
        domain_specific_adapter_for_ffn = nn.ModuleDict({})
        _domain_specific_adapter_for_ffn = collections.OrderedDict()
        for domain, domain_sz in zip(adapter_dict, adapter_bottleneck_size):
            _domain_specific_adapter_for_self_attn[domain] = PositionWiseFeedForward(feature_size,
                                                                                     domain_sz,
                                                                                     dropout_rate)
            _domain_specific_adapter_for_cross_attn[domain] = PositionWiseFeedForward(feature_size,
                                                                                      domain_sz,
                                                                                      dropout_rate)
            _domain_specific_adapter_for_ffn[domain] = PositionWiseFeedForward(feature_size,
                                                                               domain_sz,
                                                                               dropout_rate)

        domain_specific_adapter_for_self_attn.update(_domain_specific_adapter_for_self_attn)
        domain_specific_adapter_for_cross_attn.update(_domain_specific_adapter_for_cross_attn)
        domain_specific_adapter_for_ffn.update(_domain_specific_adapter_for_ffn)
        self.domain_specific_adapter_for_self_attn = domain_specific_adapter_for_self_attn
        self.domain_specific_adapter_for_ffn = domain_specific_adapter_for_ffn
        self.domain_specific_adapter_for_cross_attn = domain_specific_adapter_for_cross_attn

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

            # x = x + self.domain_specific_adapter_for_self_attn[target_domain](x)

            x, new_enc_attn_cache = self.domain_specific_sublayer_with_cache[target_domain][1] \
                (x, lambda x: self.cross_attention_layer(query=x,
                                                         key=m,
                                                         value=m,
                                                         mask=src_mask,
                                                         enc_attn_cache=enc_attn_cache,
                                                         self_attn_cache=None,
                                                         is_self_attn=False))

            # x = x + self.domain_specific_adapter_for_cross_attn[target_domain](x)

            x = self.domain_specific_sublayer[target_domain](x, self.feed_forward_layer)

            # x = x + self.domain_specific_adapter_for_ffn[target_domain](x)

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

    def forward(self, x,
                memory,
                src_mask,
                trg_mask,
                enc_attn_cache_list=None,
                self_attn_cache_list=None,
                target_domain=None, ):
        # print('decoder ', target_domain)
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
            norm_x = self.domain_specific_layer_norm[target_domain](x)

        return norm_x, layers_adapter_output, new_enc_attn_cache_list, new_self_attn_cache_list
