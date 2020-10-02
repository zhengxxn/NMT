import torch
import torch.nn as nn


class SingleLayerClassifier(nn.Module):

    def __init__(self,
                 input_dim,
                 class_num,):
        super(SingleLayerClassifier, self).__init__()

        self.w_1 = nn.Linear(input_dim, class_num)

    def forward(self, x):
        """

        :param x: [batch size, seq len, input dim]
        :return:
        """

        logits = self.w_1(torch.relu(x))
        # [batch size, seq len, class num]
        return logits





