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

    def wo_dropout_residual_forward(self, x, sublayer):
        return sublayer(self.norm(x))

    def parameter_generate_forward(self, x, sublayer, adapter_layers):
        return x + self.dropout(sublayer(self.norm(x), adapter_layers))

    def forward_used_for_adapter_distillation(self, x, sublayer):
        output = sublayer(self.norm(x))
        return x + self.dropout(output), output
