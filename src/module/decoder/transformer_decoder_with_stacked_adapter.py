import torch.nn as nn
import copy
from module.sublayer_connection.sublayer_connection import SublayerConnection
from module.sublayer_connection.sublayer_connection_with_cache import SublayerConnectionWithCache
from module.adapter.stacked_adapter import StackedAdapter


def clones(sub_module, num_layers):
    """
    Produce N identical layers
    :param sub_module:
    :param num_layers:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(sub_module) for _ in range(num_layers)])


class TransformerDecoderLayerWithStackedAdapter(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self,
                 feature_size,
                 self_attention_layer,
                 cross_attention_layer,
                 feed_forward_layer,
                 domain_adapter_dict,
                 dropout_rate):
        super(TransformerDecoderLayerWithStackedAdapter, self).__init__()
        self.feature_size = feature_size

        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_layer = feed_forward_layer

        self.adapter = StackedAdapter(domain_adapter_dict, feature_size, dropout_rate)

        self.sublayer_with_cache = clones(SublayerConnectionWithCache(feature_size, dropout_rate), 2)
        self.sublayer = SublayerConnection(feature_size, dropout_rate)

    def forward(self,
                x,
                memory,
                src_mask,
                trg_mask,
                enc_attn_cache=None,
                self_attn_cache=None,
                target_domain=None, ):

        m = memory

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

        if target_domain is not None:
            x = self.adapter(x, target_domain)

        return x, new_enc_attn_cache, new_self_attn_cache


class TransformerDecoderWithStackedAdapter(nn.Module):

    def __init__(self,
                 feature_size,
                 layer,
                 num_layers):

        super(TransformerDecoderWithStackedAdapter, self).__init__()
        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.num_layers = num_layers

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

        return self.layer_norm(x), layers_adapter_output, new_enc_attn_cache_list, new_self_attn_cache_list
