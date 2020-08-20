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


class TransformerEncoderLayerWithParallelAdapter(nn.Module):
    def __init__(self,
                 feature_size,
                 self_attention_layer,
                 feed_forward_layer,
                 parallel_adapter_layer,
                 dropout_rate,
                 layer_norm_rescale: bool = True):
        """

        :param feature_size:
        :param self_attention_layer:
        :param feed_forward_layer:
        :param parallel_adapter_layer
        :param dropout_rate:
        """
        super().__init__()

        self.feature_size = feature_size
        self.self_attention_layer = self_attention_layer  # sub layer 1
        self.feed_forward_layer = feed_forward_layer  # sub layer 2
        self.parallel_adapter = parallel_adapter_layer
        self.sub_layer_connections = clones(SublayerConnection(feature_size,
                                                               dropout_rate,
                                                               layer_norm_rescale=layer_norm_rescale), 2)

    def forward(self,
                x: torch.Tensor,
                src_mask: torch.Tensor,
                used_domain_list: list = None):

        x = self.sub_layer_connections[0](x, lambda x: self.self_attention_layer(x, x, x, src_mask))

        if used_domain_list is not None:
            x = self.sub_layer_connections[1](x, lambda x: self.parallel_adapter(x,
                                                                                 self.feed_forward_layer,
                                                                                 used_domain_list,))

        else:
            x = self.sub_layer_connections[1](x, self.feed_forward_layer)

        return x


class TransformerEncoderWithParallelAdapter(nn.Module):
    def __init__(self,
                 feature_size,
                 layer: TransformerEncoderLayerWithParallelAdapter,
                 num_layers: int,
                 layer_norm_rescale: bool = True):

        super().__init__()

        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size, elementwise_affine=layer_norm_rescale)

    def forward(self,
                x: torch.Tensor,
                src_mask: torch.Tensor,
                used_domain_list: list = None):

        layers_adapter_output = []

        for layer in self.layers:
            x = layer(x, src_mask, used_domain_list)
            layers_adapter_output.append(x)

        return {'memory': self.layer_norm(x),
                'adapter_output': layers_adapter_output}

