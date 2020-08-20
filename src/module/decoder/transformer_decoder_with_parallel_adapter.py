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


class TransformerDecoderLayerWithParallelAdapter(nn.Module):

    def __init__(self,
                 feature_size,
                 self_attention_layer,
                 cross_attention_layer,
                 feed_forward_layer,
                 parallel_adapter_layer,
                 dropout_rate,
                 layer_norm_rescale: bool = True):
        super(TransformerDecoderLayerWithParallelAdapter, self).__init__()
        self.feature_size = feature_size

        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.parallel_adapter = parallel_adapter_layer

        self.sublayer_with_cache = clones(SublayerConnectionWithCache(feature_size,
                                                                      dropout_rate,
                                                                      layer_norm_rescale=layer_norm_rescale), 2)
        self.sublayer = SublayerConnection(feature_size, dropout_rate, layer_norm_rescale=layer_norm_rescale)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor,
                trg_mask: torch.Tensor,
                enc_attn_cache: torch.Tensor = None,
                self_attn_cache: torch.Tensor = None,
                used_domain_list: list = None,):

        m = memory

        x, new_self_attn_cache = self.sublayer_with_cache[0](x, lambda x: self.self_attention_layer(query=x,
                                                                                                    key=x,
                                                                                                    value=x,
                                                                                                    mask=trg_mask,
                                                                                                    enc_attn_cache=None,
                                                                                                    self_attn_cache=self_attn_cache,
                                                                                                    is_self_attn=True))
        x, new_enc_attn_cache = self.sublayer_with_cache[1](x, lambda x: self.cross_attention_layer(query=x,
                                                                                                    key=m,
                                                                                                    value=m,
                                                                                                    mask=src_mask,
                                                                                                    enc_attn_cache=enc_attn_cache,
                                                                                                    self_attn_cache=None,
                                                                                                    is_self_attn=False))

        if used_domain_list is not None:
            x = self.sublayer(x, lambda x: self.parallel_adapter(x,
                                                                 self.feed_forward_layer,
                                                                 used_domain_list,))
        else:
            x = self.sublayer(x, self.feed_forward_layer)

        return x, new_enc_attn_cache, new_self_attn_cache


class TransformerDecoderWithParallelAdapter(nn.Module):

    def __init__(self,
                 feature_size,
                 layer,
                 num_layers,
                 layer_norm_rescale: bool = True):

        super(TransformerDecoderWithParallelAdapter, self).__init__()
        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size, elementwise_affine=layer_norm_rescale)
        self.num_layers = num_layers

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor,
                trg_mask: torch.Tensor,
                enc_attn_cache_list: torch.Tensor = None,
                self_attn_cache_list: torch.Tensor = None,
                used_domain_list: list = None,):

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
                                                               used_domain_list,
                                                               )
            layers_adapter_output.append(x)
            # layers_ref_adapter_list.append(ref_adapter_list)
            new_self_attn_cache_list = new_self_attn_cache_list + [new_self_attn_cache]
            new_enc_attn_cache_list = new_enc_attn_cache_list + [new_enc_attn_cache]

        return self.layer_norm(x), layers_adapter_output, new_enc_attn_cache_list, new_self_attn_cache_list
