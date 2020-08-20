import torch
import torch.nn as nn
import torch.nn.functional as f


class CNNClassifier(nn.Module):

    def __init__(self,
                 num_class,
                 input_dim,
                 kernel_nums: list,
                 kernel_sizes: list,
                 dropout_rate=0.5):

        super().__init__()

        self.convs = nn.ModuleList(
                        [nn.Conv2d(in_channels=1,
                                   out_channels=num,
                                   kernel_size=(width, input_dim)) for (num, width) in zip(kernel_nums, kernel_sizes)],
                                   )

        self.dropout_layer = nn.Dropout(dropout_rate)

        self.feedforward_layer = nn.Linear(sum(kernel_nums), num_class)

        self.max_kernel_size = kernel_sizes[-1]

    def forward(self, x, mask, domain_mask):
        # x : [batch size, seq len, input dim]

        mask = mask.transpose(-1, -2)
        x = x.masked_fill(mask == 0, 0)

        if x.size(1) < self.max_kernel_size:
            pd = [0, 0, 0, self.max_kernel_size - x.size(1)]

        # [batch size, max seq len, input dim]
            x = f.pad(x, pd, 'constant', 0)

        # x : [batch size, kernel num, max seq len, input dim]
        x = x.unsqueeze(1)

        # x : [batch size, kernel num, max seq_len - width]
        x = [torch.relu(conv(x).squeeze(-1)) for conv in self.convs]

        x = [torch.max_pool1d(x_, x_.size(-1)).squeeze(-1) for x_ in x]

        # [batch size, sum(kernel_num)]
        x = torch.cat(x, dim=-1)

        logits = self.feedforward_layer(self.dropout_layer(x))
        logits = torch.masked_fill(logits, domain_mask == 0, -1e9)

        return logits
