import torch
import torch.nn as nn


class RandomSynthesizer(nn.Module):
    """
    The implementation of Random Synthesizer in "SYNTHESIZER: Rethinking Self-Attention in Transformer Models"
        (https://arxiv.org/abs/2005.00743)

    This module produce a fix (H x L x L) attention weight for all sentences.
        H: Head Num
        L: Max Sent Len
    """

    def __init__(self,
                 head_num: int,
                 max_sent_len: int,
                 factorized: bool = True,
                 rank: int = 0,
                 ):
        """
        if factorized is True, the trainable model parameter is:
            r_1 [head num, max sent len, rank]
            r_2 [head num, rank, max sent len]

        else the trainable parameter is:
            r [Head num, max sent len, max sent len]

        :param head_num: head num of attention
        :param max_sent_len:
        :param rank:
        :param factorized: if factorized is True, R is convert to the product of two law rank matrices
        """

        super(RandomSynthesizer, self).__init__()
        self.factorized = factorized

        if self.factorized:
            self.r_1 = nn.Parameter(torch.randn(head_num, max_sent_len, rank, requires_grad=True))
            self.r_2 = nn.Parameter(torch.randn(head_num, rank, max_sent_len, requires_grad=True))
        else:
            self.r = nn.Parameter(torch.randn(head_num, max_sent_len, max_sent_len, requires_grad=True))

    def forward(self):

        if self.factorized:
            random_synthesizer_weight = torch.bmm(self.r_1, self.r_2)  # [head, max_sent_len, max_sent_len]
        else:
            random_synthesizer_weight = self.r

        return random_synthesizer_weight


if __name__ == "__main__":
    m = RandomSynthesizer(head_num=4, max_sent_len=20, factorized=True, rank=2)
    print(m().shape)
