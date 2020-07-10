import torch
import math
import torch.nn as nn
import copy


def clones(module, n):
    """
    Produce n identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def attention(query,
              key,
              value,
              mask=None,
              dropout_layer=None,
              synthesizer=None,
              synthesizer_weight=None):
    """
    Compute 'Scaled Dot Product Attention' with synthesizer
    :param synthesizer_weight:
    :param synthesizer:
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

    if synthesizer is not None:
        # [head num, max len, max_len]
        head_num = synthesizer.size(0)
        seq_len = scores.size(-1)
        max_sent_len = synthesizer.size(-1)

        if seq_len < max_sent_len:
            synthesizer_scores = synthesizer[:, :seq_len, :seq_len]  # [head num, seq len, seq len]
        else:
            synthesizer_scores = torch.zeros(head_num, seq_len, seq_len, device=scores.device)
            synthesizer_scores[:, :max_sent_len, :max_sent_len] = synthesizer

        merge_scores = scores * synthesizer_weight + (1 - synthesizer_weight) * synthesizer_scores

    if mask is not None:
        merge_scores = merge_scores.masked_fill(mask == 0, -1e9)

    attention_weight = torch.softmax(merge_scores, dim=-1)

    if dropout_layer is not None:
        attention_weight = dropout_layer(attention_weight)

    return torch.matmul(attention_weight, value), attention_weight


class MultiHeadedAttentionWithAdapterCache(nn.Module):
    """
    MultiHead(Q, K, V) = Concat(head_1, ..., head_n) W^O, W^O is project
    where head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^K)
    """
    def __init__(self, head_num, feature_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithAdapterCache, self).__init__()
        assert feature_size % head_num == 0
        # We assume d_v always equals d_k

        self.dimension_each_head = feature_size // head_num
        self.head_num = head_num
        self.linear_layers = clones(nn.Linear(feature_size, feature_size), 4)

        self.attention_weight = None
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, enc_attn_cache=None, self_attn_cache=None, is_self_attn=True):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        if is_self_attn:
            # print('trg_mask', mask.shape)
            key_up = self.linear_layers[1](key).\
                view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)

            value_up = self.linear_layers[2](value).\
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
                key_up = self.linear_layers[1](key).\
                    view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)
                value_up = self.linear_layers[2](value).\
                    view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)

        query_up = self.linear_layers[0](query).\
            view(batch_size, -1, self.head_num, self.dimension_each_head).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        # query, key, value, x [batch size, head num, query len, head dim]

        x, self.attention_weight = attention(query_up,
                                             key_up,
                                             value_up,
                                             mask=mask,
                                             dropout_layer=self.dropout_layer)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.head_num * self.dimension_each_head)

        return self.linear_layers[-1](x), (key_up, value_up)
