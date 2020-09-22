import torch.nn as nn
import torch
import copy

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
                 adapters,
                 dropout_rate,
                 ):
        """

        :param feature_size: the input size of each transformer encoder layer, same as output size
        :param self_attention_layer:
        :param feed_forward_layer:
        :param dropout_rate:
        """
        super().__init__()

        self.self_attention_layer = self_attention_layer  # sub layer 1
        self.feed_forward_layer = feed_forward_layer  # sub layer 2
        self.sub_layer_connections = clones(SublayerConnection(feature_size, dropout_rate), 2)
        self.feature_size = feature_size

        # make adapter
        self.adapters = adapters

    def forward(self, x,
                src_mask,
                target_domain=None,
                mix_output: bool = False,
                used_domain_list: list = None,
                ):

        # x + dropout(self_attention(Layer_norm(x)))
        x = self.sub_layer_connections[0](x, lambda x: self.self_attention_layer(x, x, x, src_mask))
        # x + dropout(feed forward(layer_norm(x)))
        x = self.sub_layer_connections[1](x, self.feed_forward_layer)

        result = self.adapters(x, target_domain, mix_output, used_domain_list)

        return result


class TransformerEncoderWithMixAdapter(nn.Module):
    def __init__(self, feature_size, layer: TransformerEncoderLayerWithMixAdapter, num_layers: int):
        super().__init__()

        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size)

    def forward(self, x,
                src_mask,
                target_domain=None,
                mix_output: bool = False,
                used_domain_list: list = None,):

        layers_adapter_output = []
        mix_gate = []

        for i, layer in enumerate(self.layers):

            # to Test if each domain in the same layer do the similar things,
            # by use different domain adapter in different layers

            if isinstance(target_domain, list):
                cur_layer_target_domain = target_domain[i]
            else:
                cur_layer_target_domain = target_domain

            result = layer(x, src_mask, cur_layer_target_domain, mix_output, used_domain_list)

            x = result['output']
            layers_adapter_output.append(x)

            if 'mix_gate' in result.keys():
                mix_gate.append(result['mix_gate'])

        return {'memory': self.layer_norm(x),
                'adapter_output': layers_adapter_output,
                'mix_gate': mix_gate}
