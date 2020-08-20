import torch.nn as nn
import torch
import copy

from module.adapter.stacked_adapter import StackedAdapter
from module.adapter.parallel_adapter import ParallelAdapter
from module.adapter.mixture_of_adapter import MixtureOfAdapter


def clones(sub_module, num_layers):
    """
    Produce N identical layers
    :param sub_module:
    :param num_layers:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(sub_module) for _ in range(num_layers)])


class TransformerEncoderLayerWithAdapter(nn.Module):
    def __init__(self,
                 feature_size,
                 self_attention_layer,
                 feed_forward_layer,
                 dropout_rate,
                 adapter_setting,
                 domain_adapter_dict: dict = None,
                 domain_list: list = None,
                 max_domain_num: int = 0,
                 ):
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
        self.feature_size = feature_size

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

    def forward(self, x,
                src_mask,
                target_domain=None,
                mix_output: bool = False,
                used_domain_list: list = None,
                mix_weight: torch.Tensor = None,
                domain_mask: torch.Tensor = None
                ):

        # x + dropout(self_attention(Layer_norm(x)))
        x = self.sub_layer_connections[0](x, lambda x: self.self_attention_layer(x, x, x, src_mask))
        # x + dropout(feed forward(layer_norm(x)))
        x = self.sub_layer_connections[1](x, self.feed_forward_layer)

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

        return x


class TransformerEncoderWithAdapter(nn.Module):
    def __init__(self, feature_size, layer: TransformerEncoderLayerWithAdapter, num_layers: int):
        super().__init__()

        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size)

        self.domain_adapter_dict = layer.domain_adapter_dict
        self.adapter_type = layer.adapter_type
        self.adapter_fusion = layer.adapter_fusion

    def forward(self, x,
                src_mask,
                target_domain=None,
                mix_output: bool = False,
                used_domain_list: list = None,
                mix_weight: torch.Tensor = None,
                domain_mask: torch.Tensor = None):

        layers_adapter_output = []
        for layer in self.layers:
            if self.adapter_fusion == 'mix':
                x = layer(x, src_mask, target_domain, mix_output, used_domain_list, mix_weight, domain_mask)
            else:
                # todo
                x = layer(x, src_mask, target_domain)

            layers_adapter_output.append(x)
            # layers_ref_adapter_list.append(ref_adapter_list)

        return {'memory': self.layer_norm(x),
                'adapter_output': layers_adapter_output}
