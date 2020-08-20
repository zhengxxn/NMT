import torch.nn as nn
import torch

import copy
from module.sublayer_connection.sublayer_connection import SublayerConnection
from module.sublayer_connection.sublayer_connection_with_cache import SublayerConnectionWithCache

from module.adapter.stacked_adapter import StackedAdapter
from module.adapter.mixture_of_adapter import MixtureOfAdapter
from module.adapter.parallel_adapter import ParallelAdapter


def clones(sub_module, num_layers):
    """
    Produce N identical layers
    :param sub_module:
    :param num_layers:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(sub_module) for _ in range(num_layers)])


class TransformerDecoderLayerWithAdapter(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self,
                 feature_size,
                 self_attention_layer,
                 cross_attention_layer,
                 feed_forward_layer,
                 dropout_rate,
                 adapter_setting,
                 domain_adapter_dict: dict = None,
                 domain_list: list = None,
                 max_domain_num: int = 0,
                 ):

        super(TransformerDecoderLayerWithAdapter, self).__init__()

        self.feature_size = feature_size
        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.sublayer_with_cache = clones(SublayerConnectionWithCache(feature_size, dropout_rate), 2)
        self.sublayer = SublayerConnection(feature_size, dropout_rate)

        # make adapter
        self.domain_adapter_dict = domain_adapter_dict
        self.adapter_type = adapter_setting['type']
        self.adapter_fusion = adapter_setting['fusion']
        self.domain_list = domain_list
        self.max_domain_num = max_domain_num

        if self.adapter_fusion == 'mix':
            self.adapters = MixtureOfAdapter(adapter_type=self.adapter_type,
                                             domain_adapter_dict=domain_adapter_dict,
                                             feature_size=feature_size,
                                             dropout_rate=dropout_rate,
                                             domain_list=domain_list,
                                             has_inner_gate=False,
                                             max_domain_num=max_domain_num)

        else:
            if self.adapter_type == 'stack':

                self.adapters = StackedAdapter(domain_adapter_dict=domain_adapter_dict,
                                               feature_size=feature_size,
                                               dropout_rate=dropout_rate)

            elif self.adapter_type == 'parallel':
                self.adapters = ParallelAdapter(domain_adapter_dict=domain_adapter_dict,
                                                feature_size=feature_size,
                                                dropout_rate=dropout_rate,
                                                max_domain_num=adapter_setting['max_domain_num'],
                                                domain_idx_dict=adapter_setting['domain_idx_dict'],
                                                )

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
                mix_weight: torch.Tensor = None,
                domain_mask: torch.Tensor = None
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

        if target_domain is not None:
            if self.adapter_fusion == 'mix' and self.adapter_type == 'stack':
                x = self.adapters(x,
                                  target_domain=target_domain,
                                  mix_output=mix_output,
                                  used_domain_list=used_domain_list,
                                  mix_weight=mix_weight,
                                  domain_mask=domain_mask,
                                  )
            else:
                # todo: add others
                x = self.adapters(x, target_domain)

        return x, new_enc_attn_cache, new_self_attn_cache


class TransformerDecoderWithAdapter(nn.Module):

    def __init__(self,
                 feature_size,
                 layer,
                 num_layers):

        super(TransformerDecoderWithAdapter, self).__init__()
        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.num_layers = num_layers

        self.domain_adapter_dict = layer.domain_adapter_dict
        self.adapter_type = layer.adapter_type
        self.adapter_fusion = layer.adapter_fusion

    def forward(self, x,
                memory,
                src_mask,
                trg_mask,
                enc_attn_cache_list=None,
                self_attn_cache_list=None,
                target_domain=None,
                mix_output: bool = False,
                used_domain_list: list = None,
                mix_weight: torch.Tensor = None,
                domain_mask: torch.Tensor = None
                ):

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
                                                               mix_output,
                                                               used_domain_list,
                                                               mix_weight,
                                                               domain_mask
                                                               )
            layers_adapter_output.append(x)
            # layers_ref_adapter_list.append(ref_adapter_list)
            new_self_attn_cache_list = new_self_attn_cache_list + [new_self_attn_cache]
            new_enc_attn_cache_list = new_enc_attn_cache_list + [new_enc_attn_cache]

        return self.layer_norm(x), layers_adapter_output, new_enc_attn_cache_list, new_self_attn_cache_list
