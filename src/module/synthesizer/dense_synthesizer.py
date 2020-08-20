import torch
import torch.nn as nn


class DenseSynthesizer(nn.Module):
    """
    The implementation of Dense Synthesizer in "SYNTHESIZER: Rethinking Self-Attention in Transformer Models"
        (https://arxiv.org/abs/2005.00743)

    This module generate a fix (H x L x L) attention weight for a sentence x

    The attention weight is only depend on each x_i.

    formulation:
        x shape is [Batch size, seq len, hid dim],

        F is a function that maps R^d to R^l, includes two linear layer
        W_1 shape is [hid dim, max seq len],
        W_2 shape is [max seq len, max seq len],
        F(x) = W_2 ( Relu ( W_1 * x +b_1 ) ) + b_2, shape is [Batch, seq len, max len]

        if factorized is True, the F is decomposed two function F_A maps R^d to R^a, F_B maps R^d to R^b
            and max_sent_len = a x b

        F_A(x) is (Batch size, seq len, a)
        F_B(x) is (Batch size, seq len, b)
            then repeat F_A(x) b times, repeat F_B(x) a times, sum

    """

    def __init__(self,
                 head_num: int,
                 feature_size: int,
                 max_sent_len: int,
                 factorized: bool = True,
                 len_a: int = 0,
                 len_b: int = 0,
                 ):
        """
        if factorized is True, the trainable model parameter is:
            Linear1 (hid dim -> hid dim)
            Linear2 (hid dim -> max sent len)

        else the trainable parameter is:
            Linear_1 (hid dim -> hid dim)
            Linear_2_a (hid dim -> len_a)
            Linear_2_b (hid dim -> len_b)

        to reduce the parameter count, remove the linear_1
=        """

        super(DenseSynthesizer, self).__init__()
        self.factorized = factorized

        if self.factorized:
            assert len_a == max_sent_len // len_b
            # self.linear_1 = nn.Linear(feature_size, feature_size)
            self.linear_2_a = nn.Linear(feature_size, len_a * head_num)
            self.linear_2_b = nn.Linear(feature_size, len_b * head_num)
            self.head_num = head_num
            self.len_a = len_a
            self.len_b = len_b
            self.max_sent_len = max_sent_len
        else:
            # self.linear_1 = nn.Linear(feature_size, feature_size)
            self.linear_2 = nn.Linear(feature_size, max_sent_len)

    def forward(self, x):
        """

        :param x: [batch size, seq len, feature size]
        :return:
        """
        if self.factorized:
            # dense_synthesizer_weight_a = self.linear_a_2(torch.relu(self.linear_a_1(x)))  # [batch size, max_sent_len, len_a]
            # dense_synthesizer_weight_b = self.linear_b_2(torch.relu(self.linear_b_1(x)))  # [batch size, max_sent_len, len_b]
            dense_synthesizer_weight_a = self.linear_2_a(x)  # [batch size, seq_len, len_a * head_num]
            dense_synthesizer_weight_b = self.linear_2_b(x)  # [batch size, seq_len, len_b * head num]
            batch_size, seq_len = dense_synthesizer_weight_a.size(0), dense_synthesizer_weight_a.size(1)
            dense_synthesizer_weight_a = dense_synthesizer_weight_a.view(batch_size, seq_len, self.head_num,
                                                                         self.len_a, 1)
            dense_synthesizer_weight_b = dense_synthesizer_weight_b.view(batch_size, seq_len, self.head_num,
                                                                         self.len_b, 1)
            dense_synthesizer_weight_a = dense_synthesizer_weight_a.expand(-1, -1, -1, -1,
                                                                           self.len_b).contiguous().view(batch_size,
                                                                                                         seq_len,
                                                                                                         self.head_num,
                                                                                                         -1)
            dense_synthesizer_weight_b = dense_synthesizer_weight_b.expand(-1, -1, -1, -1,
                                                                           self.len_a).contiguous().view(batch_size,
                                                                                                         seq_len,
                                                                                                         self.head_num,
                                                                                                         -1)
            dense_synthesizer_weight_a = dense_synthesizer_weight_a.transpose(-2, -3)  # [b, h, s, l]
            dense_synthesizer_weight_b = dense_synthesizer_weight_b.transpose(-2, -3)  # [b, h, s, l]
            # dense_synthesizer_weight = torch.matmul(dense_synthesizer_weight_a,
            #                                         dense_synthesizer_weight_b.transpose(-1, -2))
            dense_synthesizer_weight = dense_synthesizer_weight_a * dense_synthesizer_weight_b
        else:

            dense_synthesizer_weight = self.linear_2(torch.relu(self.linear_1(x)))

        return dense_synthesizer_weight


if __name__ == "__main__":
    m = DenseSynthesizer(head_num=4, feature_size=64, max_sent_len=20, factorized=True, len_a=4, len_b=5)
    print(m(torch.randn(5, 12, 64)).shape)
