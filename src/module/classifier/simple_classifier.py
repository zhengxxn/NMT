import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):

    def __init__(self,
                 input_dim,
                 feature_size,
                 class_num,):
        super(SimpleClassifier, self).__init__()

        self.w_1 = nn.Linear(input_dim, feature_size)
        self.w_2 = nn.Linear(feature_size, class_num)

    def forward(self, x, mask, domain_mask):
        """

        :param x: [batch size, seq len, input dim]
        :param mask: [batch size, 1, seq len]
        :param domain_mask: [class num]
        :return:
        """

        mask = mask.transpose(-1, -2)
        x = x.masked_fill(mask == 0, 0)

        sent_length = torch.sum(mask, dim=-2)  # [batch size, 1]
        sent_representation = torch.sum(x, dim=-2)  # [batch size, input dim]
        sent_representation = sent_representation / sent_length

        logits = self.w_2(torch.relu(self.w_1(sent_representation)))  # [batch size, class num]
        logits = torch.masked_fill(logits, domain_mask == 0, -1e9)
        # [batch size, seq len]
        return logits





