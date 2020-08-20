import torch
import math
import torch.nn as nn
import copy
from module.synthesizer.random_synthesizer import RandomSynthesizer
from module.synthesizer.dense_synthesizer import DenseSynthesizer


def clones(module, n):
    """
    Produce n identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def attention_plus_synthesizer(query: torch.Tensor,
                               key: torch.Tensor,
                               synthesizer_scores: torch.Tensor,
                               alpha: torch.Tensor,
                               mask: torch.Tensor = None,
                               dropout_layer: nn.Dropout = None,
                               is_dense: bool = False):
    """
    Compute 'Scaled Dot Product Attention' with random synthesizer

    :param is_dense: if random synthesizer, need rescale length, if dense synthesizer, do not need rescale length
    :param query: [batch size, head num, seq len, head dim]
    :param key: [batch size, head num, seq len, head dim]
    :param alpha: range: [0, 1]
    :param synthesizer_scores: [head num, max seq len, max seq len]
    :param mask: [batch size, seq len, seq len]
    :param dropout_layer:
    :return: the attention weight
    """
    d_k = query.size(-1)
    batch_size = query.size(0)
    head_num = query.size(1)
    max_sent_len = synthesizer_scores.size(-1)
    seq_len = query.size(-2)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [batch size, head num, seq len, seq len]

    if not is_dense:
        if seq_len < max_sent_len:  # cut off
            _synthesizer_scores = synthesizer_scores[:, :seq_len, :seq_len]  # [head num, seq len, seq len]
        else:  # expand
            _synthesizer_scores = torch.zeros(head_num, seq_len, seq_len, device=scores.device)
            _synthesizer_scores[:, :max_sent_len, :max_sent_len] = synthesizer_scores
    else:
        if seq_len < max_sent_len:  # cut off
            _synthesizer_scores = synthesizer_scores[:, :, :, :seq_len]  # [batch size, head num, seq len, seq len]
        else:  # expand
            _synthesizer_scores = torch.zeros(batch_size, head_num, seq_len, seq_len, device=scores.device)
            _synthesizer_scores[:, :, :, :max_sent_len] = synthesizer_scores

    merge_scores = scores * alpha + (1 - alpha) * _synthesizer_scores

    if mask is not None:
        merge_scores = merge_scores.masked_fill(mask == 0, -1e9)

    attention_weight = torch.softmax(merge_scores, dim=-1)

    if dropout_layer is not None:
        attention_weight = dropout_layer(attention_weight)

    return attention_weight


class MultiHeadedAttentionWithAdapterCache(nn.Module):
    """
    MultiHead(Q, K, V) = Concat(head_1, ..., head_n) W^O, W^O is project
    where head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^K)
    """

    def __init__(self, head_num, feature_size, domain_adapter_dict, dropout=0.1):

        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithAdapterCache, self).__init__()
        assert feature_size % head_num == 0
        # We assume d_v always equals d_k

        self.dimension_each_head = feature_size // head_num
        self.head_num = head_num
        self.linear_layers = clones(nn.Linear(feature_size, feature_size), 4)

        self.attention_weight = None
        self.dropout_layer = nn.Dropout(p=dropout)

        self.synthesizer_adapter = nn.ModuleDict()
        self.synthesizer_type = {}
        self.alpha_for_synthesizer_adapter = nn.ParameterDict()

        for domain in domain_adapter_dict.keys():
            if domain_adapter_dict[domain]['synthesizer_type'] == 'random':
                self.synthesizer_adapter[domain] = RandomSynthesizer(head_num=head_num,
                                                                     max_sent_len=domain_adapter_dict[domain][
                                                                         'max_sent_len'],
                                                                     factorized=domain_adapter_dict[domain][
                                                                         'factorized'],
                                                                     rank=domain_adapter_dict[domain]['rank'])
                self.synthesizer_type[domain] = 'random'
            else:
                self.synthesizer_adapter[domain] = DenseSynthesizer(head_num=head_num,
                                                                    feature_size=feature_size,
                                                                    max_sent_len=domain_adapter_dict[domain][
                                                                        'max_sent_len'],
                                                                    factorized=domain_adapter_dict[domain][
                                                                        'factorized'],
                                                                    len_a=domain_adapter_dict[domain]['len_a'],
                                                                    len_b=domain_adapter_dict[domain]['len_b'])
                self.synthesizer_type[domain] = 'dense'

            self.alpha_for_synthesizer_adapter[domain] = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, query, key, value,
                mask=None, enc_attn_cache=None, self_attn_cache=None, is_self_attn=True,
                target_domain=None):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        if is_self_attn:
            # print('trg_mask', mask.shape)
            key_up = self.linear_layers[1](key). \
                view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)

            value_up = self.linear_layers[2](value). \
                view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)

            if self_attn_cache is not None:
                key_up_prev, value_up_prev = self_attn_cache

                key_up = torch.cat([key_up_prev, key_up], dim=2)
                value_up = torch.cat([value_up_prev, value_up], dim=2)

        else:
            # print('src mask', mask.shape)
            if enc_attn_cache is not None:
                key_up, value_up = enc_attn_cache

            else:
                key_up = self.linear_layers[1](key). \
                    view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)
                value_up = self.linear_layers[2](value). \
                    view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)

        query_up = self.linear_layers[0](query). \
            view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)

        if self.synthesizer_type[target_domain] == 'random':
            synthesizer_scores = self.synthesizer_adapter[target_domain]()
        else:
            synthesizer_scores = self.synthesizer_adapter[target_domain](query)

        alpha = torch.sigmoid(self.alpha_for_synthesizer_adapter[target_domain])

        attention_weight = attention_plus_synthesizer(query=query_up,
                                                      key=key_up,
                                                      synthesizer_scores=synthesizer_scores,
                                                      alpha=alpha,
                                                      mask=mask,
                                                      dropout_layer=self.dropout_layer,
                                                      is_dense=False if self.synthesizer_type[target_domain] == 'random' else True)

        v = torch.matmul(attention_weight, value_up)

        # 3) "Concat" using a view and apply a final linear.
        v = v.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.head_num * self.dimension_each_head)

        return self.linear_layers[-1](v), (key_up, value_up)
