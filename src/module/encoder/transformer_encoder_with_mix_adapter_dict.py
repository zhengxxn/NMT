import torch.nn as nn
import torch
import copy

from module.adapter.mixture_of_adapter_with_classifier import MixtureOfAdapterWithClassifier
from module.sublayer_connection.sublayer_connection import SublayerConnection


def clones(sub_module, num_layers):
    """
    Produce N identical layers
    :param sub_module:
    :param num_layers:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(sub_module) for _ in range(num_layers)])


class TransformerEncoderLayerWithMixAdapter(nn.Module):
    def __init__(self,
                 feature_size,
                 self_attention_layer,
                 feed_forward_layer,
                 dropout_rate,
                 adapter_setting,
                 domain_adapter_dict: dict = None,
                 domain_list: list = None,
                 max_domain_num: int = 0,
                 domain_inner_gate_list: list = None,
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
        self.sub_layer_connections = clones(SublayerConnection(feature_size, dropout_rate), 2)
        self.feature_size = feature_size

        # make adapter
        self.domain_adapter_dict = domain_adapter_dict
        self.adapter_type = adapter_setting['type']
        self.adapter_fusion = adapter_setting['fusion']
        self.domain_list = domain_list
        self.max_domain_num = max_domain_num

        self.adapters = MixtureOfAdapterWithClassifier(adapter_type=self.adapter_type,
                                                       domain_adapter_dict=domain_adapter_dict,
                                                       feature_size=feature_size,
                                                       dropout_rate=dropout_rate,
                                                       domain_list=domain_list,
                                                       domain_inner_gate_list=domain_inner_gate_list,
                                                       max_domain_num=max_domain_num)

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

        result = self.adapters(x,
                               target_domain=target_domain,
                               mix_output=mix_output,
                               used_domain_list=used_domain_list,
                               mix_weight=mix_weight,
                               domain_mask=domain_mask,
                               )

        # {'output', 'classify_logits'}
        return result


class TransformerEncoderWithAdapter(nn.Module):
    def __init__(self, feature_size, layer: TransformerEncoderLayerWithMixAdapter, num_layers: int):
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
        layers_classify_logits = []

        for i, layer in enumerate(self.layers):

            # to Test if each domain in the same layer do the similar things,
            # by use different domain adapter in different layers
            if isinstance(target_domain, list):
                cur_layer_target_domain = target_domain[i]
            else:
                cur_layer_target_domain = target_domain

            if mix_output is True:
                x, calculate_mix_weight = layer(x, src_mask, cur_layer_target_domain, mix_output, used_domain_list,
                                                mix_weight,
                                                domain_mask)
                calculate_mix_weights.append(calculate_mix_weight)

            else:
                # todo
                x = layer(x, src_mask, cur_layer_target_domain)

            layers_adapter_output.append(x)
            # layers_ref_adapter_list.append(ref_adapter_list)

        return {'memory': self.layer_norm(x),
                'adapter_output': layers_adapter_output,
                'encoder_calculate_mix_weights': calculate_mix_weights}
