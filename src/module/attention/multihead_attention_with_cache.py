import torch
import torch.nn as nn
from module.attention.multihead_attention import attention, clones


class MultiHeadedAttentionWithCache(nn.Module):

    def __init__(self, head_num, feature_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithCache, self).__init__()
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

        attention_weight = attention(query_up,
                                     key_up,
                                     mask=mask,
                                     dropout_layer=self.dropout_layer)
        x = torch.matmul(attention_weight, value_up)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.head_num * self.dimension_each_head)

        return self.linear_layers[-1](x), (key_up, value_up)
