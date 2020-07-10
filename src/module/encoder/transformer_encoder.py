import torch.nn as nn
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


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 feature_size: int,
                 self_attention_layer,
                 feed_forward_layer,
                 dropout_rate: float,
                 layer_norm_rescale: bool = True):
        """

        :param feature_size: the input size of each transformer encoder layer, same as output size
        :param self_attention_layer:
        :param feed_forward_layer:
        :param dropout_rate:
        """
        super().__init__()

        self.feature_size = feature_size
        self.self_attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer

        self.sub_layer_connections = clones(SublayerConnection(feature_size,
                                                               dropout_rate,
                                                               layer_norm_rescale=layer_norm_rescale), 2)

    def forward(self, x, src_mask):
        # x + dropout(self_attention(Layer_norm(x)))
        x = self.sub_layer_connections[0](x, lambda x: self.self_attention_layer(x, x, x, src_mask))
        # x + dropout(feed forward(layer_norm(x)))
        return self.sub_layer_connections[1](x, self.feed_forward_layer)


class TransformerEncoder(nn.Module):
    def __init__(self,
                 feature_size: int,
                 layer: TransformerEncoderLayer,
                 num_layers: int,
                 layer_norm_rescale: bool = True):
        super().__init__()

        self.layers = clones(layer, num_layers)
        self.layer_norm = nn.LayerNorm(feature_size, elementwise_affine=layer_norm_rescale)

    def forward(self, x, src_mask, src_lengths=None):
        for layer in self.layers:
            x = layer(x, src_mask)

        return {'memory': self.layer_norm(x)}
