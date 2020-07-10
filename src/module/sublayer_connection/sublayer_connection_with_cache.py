import torch.nn as nn


class SublayerConnectionWithCache(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self,
                 size: int,
                 dropout: float,
                 layer_norm_rescale: bool = True):

        super(SublayerConnectionWithCache, self).__init__()

        self.norm = nn.LayerNorm(size, elementwise_affine=layer_norm_rescale)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."

        t, cache = sublayer(self.norm(x))
        return x + self.dropout(t), cache
