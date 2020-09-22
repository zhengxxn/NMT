import torch.nn as nn
import torch

import copy
from module.sublayer_connection.sublayer_connection import SublayerConnection
from module.sublayer_connection.sublayer_connection_with_cache import SublayerConnectionWithCache


def clones(sub_module, num_layers):
    """
    Produce N identical layers
    :param sub_module:
    :param num_layers:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(sub_module) for _ in range(num_layers)])


class TransformerDecoderLayerWithMixAdapter(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self,
                 feature_size,
                 self_attention_layer,
                 cross_attention_layer,
                 feed_forward_layer,
                 adapters,
                 dropout_rate,
                 ):
        super(TransformerDecoderLayerWithMixAdapter, self).__init__()

        self.feature_size = feature_size
        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.sublayer_with_cache = clones(SublayerConnectionWithCache(feature_size, dropout_rate), 2)
        self.sublayer = SublayerConnection(feature_size, dropout_rate)

        # make adapter
        self.adapters = adapters

    def forward(self,
                x,
                memory,
                src_mask,
                trg_mask,
                enc_attn_cache=None,
                self_attn_cache=None,
                target_domain=None,
                mix_output: bool = False,
                used_domain_list: list = None,
                ):
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

        result = self.adapters(x, target_domain, mix_output, used_domain_list)
        result['enc_attn_cache'] = new_enc_attn_cache
        result['self_attn_cache'] = new_self_attn_cache
        return result


class TransformerDecoderWithMixAdapter(nn.Module):

    def __init__(self,
                 feature_size,
                 layer,
                 num_layers):

        super(TransformerDecoderWithMixAdapter, self).__init__()
        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.num_layers = num_layers

    def forward(self, x,
                memory,
                src_mask,
                trg_mask,
                enc_attn_cache_list=None,
                self_attn_cache_list=None,
                adapter_output_list=None,
                mix_gate_list=None,
                target_domain=None,
                mix_output: bool = False,
                used_domain_list: list = None,
                ):

        if self_attn_cache_list is None:
            self_attn_cache_list = [None] * self.num_layers

        if enc_attn_cache_list is None:
            enc_attn_cache_list = [None] * self.num_layers

        if mix_gate_list is None:
            mix_gate_list = [None] * self.num_layers

        if adapter_output_list is None:
            adapter_output_list = [None] * self.num_layers

        new_enc_attn_cache_list = []
        new_self_attn_cache_list = []

        for i, layer in enumerate(self.layers):

            # to Test if each domain in the same layer do the similar things,
            # by use different domain adapter in different layers

            if isinstance(target_domain, list):
                cur_layer_target_domain = target_domain[i]
            else:
                cur_layer_target_domain = target_domain

            result = layer(x,
                           memory,
                           src_mask,
                           trg_mask,
                           enc_attn_cache_list[i],
                           self_attn_cache_list[i],
                           cur_layer_target_domain,
                           mix_output,
                           used_domain_list,
                           )

            x = result.get('output', None)

            if adapter_output_list[i] is None:
                adapter_output_list[i] = x
            else:
                adapter_output_list[i] = torch.cat([adapter_output_list[i], x], dim=1)

            if mix_gate_list[i] is None:
                mix_gate_list[i] = result.get('mix_gate', None)
            else:
                mix_gate_list[i] = torch.cat([mix_gate_list[i], result.get('mix_gate', None)], dim=1)

            new_self_attn_cache_list = new_self_attn_cache_list + [result['self_attn_cache']]
            new_enc_attn_cache_list = new_enc_attn_cache_list + [result['enc_attn_cache']]

        return {
            'logits': self.layer_norm(x),
            'adapter_output': adapter_output_list if adapter_output_list[0] is not None else None,
            'mix_gate': mix_gate_list if mix_gate_list[0] is not None else None,
            'self_attn_cache_list': new_self_attn_cache_list,
            'enc_attn_cache_list': new_enc_attn_cache_list,
        }
