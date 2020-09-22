import torch.nn as nn
import torch
from module.adapter.feedforward_adapter_layer import FeedForwardAdapterLayer
from module.sublayer_connection.sublayer_connection import SublayerConnection


class MixtureOfAdapter(nn.Module):

    def __init__(self,
                 domain_adapter_dict,
                 feature_size,
                 dropout_rate,
                 domain_list: list = None,
                 domain_inner_gate_list: list = None,
                 gate_activate_func='sigmoid'):
        """
        This module implements the Mixture-Of-Adapter Layer.

        :param domain_adapter_dict:
        :param feature_size:
        :param dropout_rate:
        :param domain_list:
        :param domain_inner_gate_list:
        """
        super(MixtureOfAdapter, self).__init__()

        # initialization of all adapter based module
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
        # we regard the adapter output as a modification (direction) of the original ffn module output
        # the input of each gate is (ffn_output; adapter output), and return a gate (tanh or sigmoid) for the adapter output
        #
        domain_gate = nn.ModuleDict({})
        for domain in domain_inner_gate_list:
            domain_gate[domain] = nn.Linear(2 * feature_size, 1)
        self.domain_gate = domain_gate
        self.gate_activate_func = gate_activate_func

    def forward(self,
                x,
                target_domain,
                mix_output: bool = False,
                used_domain_list: list = None,):

        """

        :param x: [B, L, H]
        :param target_domain: a string, like 'news', 'book', 'bible'
        :param mix_output:
        :param used_domain_list: used domain list, like ['book', 'iwslt']
        :return:
        """

        # if we only use one adapter
        if mix_output is False:

            if target_domain == 'news':
                return {'output': x}
            else:
                return {'output': self.sublayer_connection_for_adapter[target_domain](x, self.adapter_layers[target_domain])}

        # else we should mix the current adapters output
        else:
            # confirm the order in domain_weight and used_domain_list is same
            assert self.check_domain_list_order(used_domain_list) is True

            # calculate the adapter outputs separately
            adapter_outputs = []
            domain_gates = []

            for domain in used_domain_list:
                if domain == 'news':  # this maybe never used, because we use residual connect
                    adapter_outputs.append(x)
                else:
                    # here not plus x, looks like a modification
                    domain_adapter_output = self.sublayer_connection_for_adapter[domain]. \
                        wo_residual_forward(x, self.adapter_layers[domain])

                    # use x and domain adapter output to decide the length
                    domain_gate = self.domain_gate[domain](torch.cat([x, domain_adapter_output], dim=-1))  # [B, S, 1]
                    if self.gate_activate_func == 'sigmoid':
                        domain_gate = torch.sigmoid(domain_gate)
                    else:
                        domain_gate = torch.tanh(domain_gate)

                    domain_gates.append(domain_gate)
                    adapter_outputs.append(domain_gate * domain_adapter_output)

            # mix the outputs
            adapter_outputs = torch.stack(adapter_outputs, dim=-1)  # [B, L, H, D]
            # residual
            mix_adapter_outputs = torch.sum(adapter_outputs, dim=-1)

            # if target domain is not in used domain_list and not None, it means:
            #   1. we do not know the target domain, just to mix
            #   2. we are training for the target domain adapter
            if target_domain is not None and target_domain not in used_domain_list:
                mix_adapter_outputs += self.sublayer_connection_for_adapter[target_domain]. \
                        wo_residual_forward(x, self.adapter_layers[target_domain])

            # if mix adapter output, the return gate for check
            return {
                'output': x + mix_adapter_outputs,
                'mix_gate': domain_gates
            }

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

    m = MixtureOfAdapter(
                         domain_adapter_dict=test_domain_adapter_dict,
                         feature_size=16,
                         dropout_rate=0.2,
                         domain_list=['books', 'laws', 'medical'],
                         domain_inner_gate_list=['books', 'medical'],)

    batch_size = 3
    seq_len = 4
    feature_sz = 16
    test_x = torch.randn(batch_size, seq_len, feature_sz)

    m.forward(test_x,
              target_domain='medical',
              mix_output=True,
              used_domain_list=['books', 'medical'],
              )
