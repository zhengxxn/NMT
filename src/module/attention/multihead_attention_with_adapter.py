import torch
import math
import torch.nn as nn
import copy
from module.synthesizer.random_synthesizer import RandomSynthesizer


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
                               dropout_layer: nn.Dropout = None, ):
    """
    Compute 'Scaled Dot Product Attention' with random synthesizer

    :param query: [batch size, head num, seq len, head dim]
    :param key: [batch size, head num, seq len, head dim]
    :param alpha: range: [0, 1]
    :param synthesizer_scores: [head num, max seq len, max seq len]
    :param mask: [batch size, seq len, seq len]
    :param dropout_layer:
    :return: the attention weight
    """
    d_k = query.size(-1)
    head_num = synthesizer_scores.size(0)
    max_sent_len = synthesizer_scores.size(-1)
    seq_len = query.size(-2)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [batch size, head num, seq len, seq len]

    if seq_len < max_sent_len:  # cut off
        _synthesizer_scores = synthesizer_scores[:, :seq_len, :seq_len]  # [head num, seq len, seq len]
    else:  # expand
        _synthesizer_scores = torch.zeros(head_num, seq_len, seq_len, device=scores.device)
        _synthesizer_scores[:, :max_sent_len, :max_sent_len] = synthesizer_scores

    merge_scores = scores * alpha + (1 - alpha) * _synthesizer_scores

    if mask is not None:
        merge_scores = merge_scores.masked_fill(mask == 0, -1e9)

    attention_weight = torch.softmax(merge_scores, dim=-1)

    if dropout_layer is not None:
        attention_weight = dropout_layer(attention_weight)

    return attention_weight


class MultiHeadedAttentionWithAdapter(nn.Module):
    """
    The implementation of MultiHead Attention with Adapter,

    The Adapter is in synthesizer form as a extra attention weight plus with the original self attention weight.

    formulation:
        S_1(X) is Scaled Dot Product self Attention Weight calculated by query and key
        S_2(X) is Random Synthesizer Attention Weight
        Y = SoftMax ( alpha * S_1(X) + (1-alpha) * S_2(X) ) * V(x)

    """

    def __init__(self,
                 head_num,
                 feature_size,
                 domain_adapter_dict,
                 dropout=0.1,
                 ):

        super(MultiHeadedAttentionWithAdapter, self).__init__()
        assert feature_size % head_num == 0

        self.dimension_each_head = feature_size // head_num
        self.head_num = head_num
        self.linear_layers = clones(nn.Linear(feature_size, feature_size), 4)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.synthesizer_adapter = nn.ModuleDict()
        self.alpha_for_synthesizer_adapter = nn.ParameterDict()
        for domain in domain_adapter_dict.keys():
            self.synthesizer_adapter[domain] = RandomSynthesizer(head_num=head_num,
                                                                 max_sent_len=domain_adapter_dict[domain][
                                                                     'max_sent_len'],
                                                                 factorized=domain_adapter_dict[domain]['factorized'],
                                                                 rank=domain_adapter_dict[domain]['rank'])
            self.alpha_for_synthesizer_adapter[domain] = nn.Parameter(torch.randn(1), requires_grad=True)
            # self.weight_for_synthesizer_adapter[domain][0] = 0.5

        self.attention_weight = None

    def forward(self, query, key, value, mask=None, target_domain=None):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query_up = self.linear_layers[0](query).view(batch_size, -1, self.head_num, self.dimension_each_head) \
            .transpose(1, 2)
        key_up = self.linear_layers[1](key).view(batch_size, -1, self.head_num, self.dimension_each_head) \
            .transpose(1, 2)
        value_up = self.linear_layers[2](value).view(batch_size, -1, self.head_num, self.dimension_each_head) \
            .transpose(1, 2)

        synthesizer_scores = self.synthesizer_adapter[target_domain]()
        alpha = torch.sigmoid(self.alpha_for_synthesizer_adapter[target_domain]())

        attention_weight = attention_plus_synthesizer(query=query_up,
                                                      key=key_up,
                                                      synthesizer_scores=synthesizer_scores,
                                                      alpha=alpha,
                                                      mask=mask,
                                                      dropout_layer=self.dropout_layer, )

        v = torch.matmul(attention_weight, value_up)

        # 3) "Concat" using a view and apply a final linear.
        v = v.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.head_num * self.dimension_each_head)

        return self.linear_layers[-1](v)


if __name__ == "__main__":
    domain_dict = {
        'laws':
            {'max_sent_len': 10,
             'rank': 8,
             },
    }
    m = MultiHeadedAttentionWithAdapter(head_num=8, feature_size=128, domain_adapter_dict=domain_dict)

    mask = torch.ones(8, 30, 30, dtype=torch.int32)
    mask[:, 20:, 20:] = 0
    x = torch.randn(8, 30, 128)
    print(m(x, x, x, mask, 'laws'))
