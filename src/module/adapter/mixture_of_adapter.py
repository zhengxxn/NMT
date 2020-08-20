import torch.nn as nn
import torch
from module.adapter.feedforward_adapter_layer import FeedForwardAdapterLayer
from module.sublayer_connection.sublayer_connection import SublayerConnection


class MixtureOfAdapter(nn.Module):

    def __init__(self,
                 adapter_type,
                 domain_adapter_dict,
                 feature_size,
                 dropout_rate,
                 domain_list: list = None,
                 has_inner_gate: bool = False,
                 max_domain_num: int = None):
        """
        This module implements the Mixture-Of-Adapter Layer.
        Suppose we have some trained different domain adapter now,
        if the module contains a inner gate, then the gate will provide a mix weight,


        :param domain_adapter_dict:
        :param feature_size:
        :param dropout_rate:
        :param domain_list:
        :param has_inner_gate:
        :param max_domain_num:
        """
        super(MixtureOfAdapter, self).__init__()

        # todo: add parallel
        if adapter_type != 'stack':
            return

        # for all adapter based module
        adapter_layers = nn.ModuleDict({})
        sublayer_connection_for_adapter = nn.ModuleDict({})
        for domain in domain_adapter_dict.keys():
            adapter_layers[domain] = FeedForwardAdapterLayer(input_dim=feature_size,
                                                             ff_dim=domain_adapter_dict[domain]['memory_count'],
                                                             dropout=dropout_rate)
            sublayer_connection_for_adapter[domain] = SublayerConnection(size=feature_size,
                                                                         dropout=dropout_rate)
        self.adapter_layers = adapter_layers
        self.sublayer_connection_for_adapter = sublayer_connection_for_adapter

        # for mixture of experts
        self.domain_list = domain_list
        self.domain_dict = {}
        for i, domain in enumerate(self.domain_list):
            self.domain_dict[domain] = i

        # domain mix gate
        self.max_domain_num = max_domain_num
        self.has_inner_gate = has_inner_gate
        if has_inner_gate:
            self.inner_gate = nn.Linear(in_features=feature_size, out_features=max_domain_num)

    def forward(self,
                x,
                target_domain,
                mix_output: bool = False,
                used_domain_list: list = None,
                mix_weight: torch.Tensor = None,
                domain_mask: torch.Tensor = None):

        """

        confirm the order in domain_weight and used_domain_list is same

        :param x: [B, L, H]
        :param target_domain: a string, like 'news', 'book', 'bible'
        :param mix_output:
        :param used_domain_list: used domain list, like ['book', 'iwslt']
        :param mix_weight: [B, L, D]
        :param domain_mask: [D]
        :return:
        """

        # if we only use one adapter
        if mix_output is False:
            if target_domain == 'news':
                return x
            else:
                return self.sublayer_connection_for_adapter[target_domain](x, self.adapter_layers[target_domain])

        # else we should mix the current adapters output
        else:

            assert self.check_domain_list_order(used_domain_list) is True

            # first, produce the weight based on the input, or provide the weight
            if self.has_inner_gate and mix_weight is None:

                if domain_mask is None:
                    domain_mask = [0] * self.max_domain_num
                    for domain in used_domain_list:
                        domain_mask[self.domain_dict[domain]] = 1
                    domain_mask = torch.Tensor(domain_mask)

                mix_weight = self.inner_gate(x)  # [B, S, D_Max]
                mix_weight = torch.masked_fill(mix_weight, domain_mask == 0, -1e9)
                mix_weight = torch.softmax(mix_weight, dim=-1)

                # print(mix_weight)
                used_domain_idx = domain_mask.nonzero().flatten()
                mix_weight = mix_weight.index_select(dim=-1, index=used_domain_idx)  # [B, S, D_Max] -> [B, S, D_Used]
                # print(mix_weight)

            # calculate the adapter outputs separately
            adapter_outputs = []
            for domain in used_domain_list:
                if domain == 'news':  # this maybe not used, because we use residual connect
                    adapter_outputs.append(x)
                else:
                    # here not plus x
                    domain_adapter_output = self.sublayer_connection_for_adapter[domain]. \
                        wo_residual_forward(x, self.adapter_layers[target_domain])
                    adapter_outputs.append(domain_adapter_output)

            # mix the outputs
            adapter_outputs = torch.stack(adapter_outputs, dim=-1)  # [B, L, H, D]
            mix_weight = mix_weight.unsqueeze(-1)  # [B, L, D, 1]
            mix_adapter_outputs = torch.matmul(adapter_outputs, mix_weight).squeeze(-1)

            # residual
            return x + mix_adapter_outputs

    def check_domain_list_order(self, used_domain_list):
        order = [self.domain_dict[domain] for domain in used_domain_list]
        return all(order[i] < order[i + 1] for i in range(len(order) - 1))


if __name__ == "__main__":
    test_domain_adapter_dict = {
        'books': {
            'memory_count': 32,
        },
        'laws': {
            'memory_count': 32,
        },
        'medical': {
            'memory_count': 32,
        }
    }

    m = MixtureOfAdapter(adapter_type='stack',
                         domain_adapter_dict=test_domain_adapter_dict,
                         feature_size=16,
                         dropout_rate=0.2,
                         domain_list=['books', 'laws', 'medical'],
                         has_inner_gate=True,
                         max_domain_num=5)

    batch_size = 3
    seq_len = 4
    feature_sz = 16
    test_x = torch.randn(batch_size, seq_len, feature_sz)

    m.forward(test_x,
              target_domain='medical',
              mix_output=True,
              used_domain_list=['books', 'medical'],
              mix_weight=None, )
