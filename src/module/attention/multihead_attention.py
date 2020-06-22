import torch
import math
import torch.nn as nn
import copy


def clones(module, n):
    """
    Produce n identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout_layer=None):
    """
    Compute 'Scaled Dot Product Attention'
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout_layer:
    :return:
    """
    d_k = query.size(-1)

    # query, key, value [batch size, head num, seq len, head dim]
    # scores [batch size, head num, seq len, seq len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weight = torch.softmax(scores, dim=-1)

    if dropout_layer is not None:
        attention_weight = dropout_layer(attention_weight)

    return torch.matmul(attention_weight, value), attention_weight


class MultiHeadedAttention(nn.Module):
    """
    MultiHead(Q, K, V) = Concat(head_1, ..., head_n) W^O, W^O is project
    where head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^K)
    """
    def __init__(self, head_num, feature_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert feature_size % head_num == 0
        # We assume d_v always equals d_k

        self.dimension_each_head = feature_size // head_num
        self.head_num = head_num
        self.linear_layers = clones(nn.Linear(feature_size, feature_size), 4)

        self.attention_weight = None
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        key_up = self.linear_layers[1](key).view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)
        value_up = self.linear_layers[2](key).view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)

        # 1) Do all the linear projections in batch from [batch size, seq len, feature size] => Wq, Wk, Wv multiply =>
        # view as [batch size, head num, seq len, head dim]
        # query, key, value = \
        #     [linear(x).view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)
        #      for linear, x in zip(self.linear_layers, (query, key, value))]

        query_up = self.linear_layers[0](query).view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)
        # key = self.linear_layers[1](key).view(batch_size, -1, self.head_num, self.dimension_each_head)
        # value = self.linear_layers[2](key).view(batch_size, -1, self.head_num, self.dimension_each_head)

        # 2) Apply attention on all the projected vectors in batch.
        # query, key, value, x [batch size, head num, seq len, head dim]

        x, self.attention_weight = attention(query_up,
                                             key_up,
                                             value_up,
                                             mask=mask,
                                             dropout_layer=self.dropout_layer)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.head_num * self.dimension_each_head)

        return self.linear_layers[-1](x)
