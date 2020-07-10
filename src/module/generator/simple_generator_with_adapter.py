import torch
import torch.nn as nn
from module.adapter.feedforward_adapter_layer import FeedForwardAdapterLayer
import collections


class SimpleGeneratorWithAdapter(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self,
                 feature_size,
                 vocab_size,
                 domain_adapter_dict,
                 bias=False):
        super(SimpleGeneratorWithAdapter, self).__init__()

        self.proj = nn.Linear(feature_size, vocab_size, bias=bias)

        adapter_layers = nn.ModuleDict({})
        _adapter_layers = collections.OrderedDict()
        for domain in domain_adapter_dict.keys():
            _adapter_layers[domain] = FeedForwardAdapterLayer(input_dim=feature_size,
                                                              ff_dim=domain_adapter_dict[domain]['emb_adapt_count'],
                                                              dropout=0.1)

        adapter_layers.update(_adapter_layers)
        self.adapter_layers = adapter_layers

    def forward(self, x, target_domain):
        w = self.proj.weight

        w = w + self.adapter_layers[target_domain](w)
        w = w.t()
        # print(w.shape)
        # print(x.shape)
        # x [batch, seq, feature], w [feature, vocab]
        if x.dim() == 3:
            x = torch.einsum('ij,abi->abj', (w, x))
        elif x.dim() == 2:
            x = torch.einsum('ij,ai->aj', (w, x))
        elif x.dim() == 4:
            x = torch.einsum('ij,abci->abcj', (w, x))

        return torch.log_softmax(x, dim=-1)
