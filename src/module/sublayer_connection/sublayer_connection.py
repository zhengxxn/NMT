import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self,
                 size,
                 dropout,
                 layer_norm_rescale=True):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, elementwise_affine=layer_norm_rescale)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

    def wo_residual_forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))
