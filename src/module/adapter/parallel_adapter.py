import torch.nn as nn
import torch
from module.adapter.feedforward_adapter_layer import FeedForwardAdapterLayer
from module.adapter.bias import DenseFFNBias, RandomFFNBias


class ParallelAdapter(nn.Module):
    """
    Implementation of Parallel Adapter

    The Parallel Adapter consists of one or many domain adapters, and the original FFN layer output is regard as
    a (news) adapter

    The output is the summation of different adapters' output.

    """

    def __init__(self,
                 domain_adapter_dict: dict,
                 feature_size: int,
                 dropout_rate: float,
                 max_domain_num: int,
                 domain_idx_dict: dict,
                 weighted_sum: bool = False,
                 ):

        super(ParallelAdapter, self).__init__()

        self.bias_dict = {}
        adapters = nn.ModuleDict({})
        domain_bias = nn.ModuleDict({})
        for domain in domain_adapter_dict.keys():
            adapters[domain] = FeedForwardAdapterLayer(input_dim=feature_size,
                                                       ff_dim=domain_adapter_dict[domain]['memory_count'],
                                                       dropout=dropout_rate)
            if domain_adapter_dict[domain]['bias_for_other_adapter']:
                self.bias_dict[domain] = True

                if domain_adapter_dict[domain]['bias_type'] == 'random':
                    domain_bias['news_for_' + domain] = RandomFFNBias(ff_dim=2048)
                else:
                    domain_bias['news_for_' + domain] = DenseFFNBias(input_dim=feature_size,
                                                                     ff_dim=2048,
                                                                     rank=domain_adapter_dict[domain]['rank'])

                for other_domain in domain_adapter_dict.keys():
                    if other_domain != domain:
                        if domain_adapter_dict[domain]['bias_type'] == 'random':
                            domain_bias[other_domain + '_for_' + domain] = \
                                RandomFFNBias(ff_dim=domain_adapter_dict[other_domain]['memory_count'])

                        else:
                            domain_bias[other_domain + '_for_' + domain] = \
                                DenseFFNBias(input_dim=feature_size,
                                             ff_dim=domain_adapter_dict[other_domain]['memory_count'],
                                             rank=domain_adapter_dict[domain]['rank'])
            # add bias for other domain
            else:
                self.bias_dict[domain] = False

        self.adapters = adapters
        self.domain_bias = domain_bias
        self.weighted_sum = weighted_sum
        self.max_domain_num = max_domain_num
        # self.weighted_sum_layer = nn.Linear(feature_size, max_domain_num)
        self.domain_idx_dict = domain_idx_dict

    def forward(self,
                x,
                original_ffn,
                used_domain_list: list,  # suppose the target domain is in index 0
                ):
        target_domain = used_domain_list[0]
        if self.weighted_sum:

            weight = self.weighted_sum_layer(x)  # [batch size, seq len, max_domain_num]
            mask = torch.zeros(self.max_domain_num)
            for used_domain in used_domain_list:
                mask[self.domain_idx_dict[used_domain]] = 1
            weight = weight.masked_fill(mask == 0, -1e9)
            weight = torch.softmax(weight, dim=-1)  # get used domain percent

            adapter_outputs = [weight[:, :, 0:1] * original_ffn(x)]
            for used_domain in used_domain_list:
                domain_idx = self.domain_idx_dict[used_domain]
                adapter_outputs.append(weight[:, :, domain_idx:domain_idx+1] * self.adapters[used_domain](x))
            memory_sum = torch.sum(torch.stack(adapter_outputs, dim=0), dim=0)

        else:
            adapter_outputs = []

            if 'news' in used_domain_list:

                if self.bias_dict[target_domain]:
                    original_ffn_value = original_ffn(x, self.domain_bias['news_for_' + target_domain](x))
                else:
                    original_ffn_value = original_ffn(x)  # [batch size, seq len, hid dim]
                adapter_outputs.append(original_ffn_value)

            for used_domain in used_domain_list:
                if used_domain == target_domain or not self.bias_dict[target_domain]:
                    adapter_outputs.append(self.adapters[used_domain](x))

                else:
                    adapter_outputs.append(self.adapters[used_domain]
                                           (x, self.domain_bias[used_domain + '_for_' + target_domain](x)))

            memory_sum = torch.sum(torch.stack(adapter_outputs, dim=0), dim=0)

        # [batch size, seq len, hid dim, 2]
        # concat_value = torch.cat([original_ffn_value.unsqueeze(-1), target_domain_adapter_value.unsqueeze(-1)], -1)

        # [batch size, seq len, 2, 1]
        # domain_score = torch.softmax(self.adapter_combine(x), dim=-1).unsqueeze(-1)

        # [batch size, seq len, hid dim]
        # global_memory = torch.matmul(concat_value, domain_score).squeeze(-1)

        return memory_sum
