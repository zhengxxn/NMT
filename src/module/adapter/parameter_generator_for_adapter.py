import torch
import torch.nn as nn


class ParameterGeneratorForAdapter(nn.Module):
    """
    This Module Produce the Parameter for a new Adapter Layer, given a set of Adapter Layer
    """

    def __init__(self,
                 adapter_dict: dict,
                 used_adapters: list,
                 generate_dim,
                 feature_size,
                 bottleneck_dim,
                 dropout_rate,
                 linear_transform=False):

        super().__init__()

        sum_memory_dim = 0

        self.used_adapters = used_adapters
        for domain in used_adapters:
            sum_memory_dim = sum_memory_dim + adapter_dict[domain]['memory_count']

        self.size_transform_w1 = nn.Linear(sum_memory_dim, generate_dim)  # for w_1
        self.size_transform_w2 = nn.Linear(sum_memory_dim, generate_dim)  # for w_2

        # bottleneck dim << feature size and sum_memory dim
        # low rank transform
        self.size_transform_for_bias1_1 = nn.Linear(sum_memory_dim, bottleneck_dim)
        self.size_transform_for_bias1_2 = nn.Linear(bottleneck_dim, generate_dim)
        self.size_transform_for_bias2_1 = nn.Linear(feature_size * len(used_adapters), bottleneck_dim)
        self.size_transform_for_bias2_2 = nn.Linear(bottleneck_dim, feature_size)

        # as described in the paper "weight distillation", we introduce two learnable matrices
        #
        self.linear_transform = linear_transform
        if not linear_transform:
            self.W_for_w1_weight = nn.Parameter(torch.ones(feature_size, generate_dim))
            self.B_for_w1_weight = nn.Parameter(torch.zeros(feature_size, generate_dim))
            self.W_for_w1_bias = nn.Parameter(torch.ones(generate_dim))
            self.B_for_w1_bias = nn.Parameter(torch.zeros(generate_dim))
            self.W_for_w2_weight = nn.Parameter(torch.ones(generate_dim, feature_size))
            self.B_for_w2_weight = nn.Parameter(torch.zeros(generate_dim, feature_size))
            self.W_for_w2_bias = nn.Parameter(torch.ones(feature_size))
            self.B_for_w2_bias = nn.Parameter(torch.zeros(feature_size))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adapter_layers):
        """

        """

        # first generate w_1
        w1_weight = []
        w1_bias = []
        for domain in self.used_adapters:
            w1_weight.append(adapter_layers[domain].w_1.weight.data)  # [memory_dim, feature_size]
            w1_bias.append(adapter_layers[domain].w_1.bias.data)  # [memory_dim]
        w1_weight = torch.cat(w1_weight, dim=0).transpose(0, 1)  # [feature size, sum_memory_dim]
        w1_bias = torch.cat(w1_bias, dim=0)  # [sum_memory_dim]
        w1_weight = self.size_transform_w1(w1_weight)  # [feature size, generate_dim]
        w1_bias = self.size_transform_for_bias1_2(self.size_transform_for_bias1_1(w1_bias))  # [generate_dim]
        if not self.linear_transform:
            w1_weight = torch.tanh(w1_weight) * self.W_for_w1_weight + self.B_for_w1_weight
            w1_bias = torch.tanh(w1_bias) * self.W_for_w1_bias + self.B_for_w1_bias

        # then generate w_2
        w2_weight = []
        w2_bias = []
        for domain in self.used_adapters:
            # [feature_size, memory_dim]
            w2_weight.append(adapter_layers[domain].w_2.weight.data)
            w2_bias.append(adapter_layers[domain].w_2.bias.data)
        w2_weight = torch.cat(w2_weight, dim=0)  # [feature size, sum_memory_dim]
        w2_bias = torch.cat(w2_bias, dim=0)  # [n * 512]
        w2_weight = self.size_transform_w2(w2_weight).transpose(0, 1)  # [generate dim, feature size]
        w2_bias = self.size_transform_for_bias2_2(self.size_transform_for_bias2_1(w2_bias))  # [feature size]
        if not self.linear_transform:
            w2_weight = torch.tanh(w2_weight) * self.W_for_w2_weight + self.B_for_w2_weight
            w2_bias = torch.tanh(w2_bias) * self.W_for_w2_bias + self.B_for_w2_bias

        # now have w_1, w_2, and look as a new adapter
        # x [batch size, seq len, feature size]
        s_1 = torch.matmul(x, w1_weight)  # [batch size, seq len, generate dim]
        s_1 = s_1 + w1_bias
        s_1 = self.dropout(torch.relu(s_1))

        s_2 = torch.matmul(s_1, w2_weight)  # [batch size, seq len, feature size]
        s_2 = s_2 + w2_bias

        return s_2

    def generate_param(self, adapter_layers):
        w1_weight = []
        w1_bias = []
        for domain in self.used_adapters:
            w1_weight.append(adapter_layers[domain].w_1.weight.data)  # [memory_dim, feature_size]
            w1_bias.append(adapter_layers[domain].w_1.bias.data)  # [memory_dim]
        w1_weight = torch.cat(w1_weight, dim=0).transpose(0, 1)  # [feature size, sum_memory_dim]
        w1_bias = torch.cat(w1_bias, dim=0)  # [sum_memory_dim]
        w1_weight = self.size_transform_w1(w1_weight)  # [feature size, generate_dim]
        w1_bias = self.size_transform_for_bias1_2(self.size_transform_for_bias1_1(w1_bias))  # [generate_dim]
        w1_weight = torch.tanh(w1_weight) * self.W_for_w1_weight + self.B_for_w1_weight
        w1_bias = torch.tanh(w1_bias) * self.W_for_w1_bias + self.B_for_w1_bias

        # then generate w_2
        w2_weight = []
        w2_bias = []
        for domain in self.used_adapters:
            # [feature_size, memory_dim]
            w2_weight.append(adapter_layers[domain].w_2.weight.data)
            w2_bias.append(adapter_layers[domain].w_2.bias.data)
        w2_weight = torch.cat(w2_weight, dim=0)  # [feature size, sum_memory_dim]
        w2_bias = torch.cat(w2_bias, dim=0)  # [n * 512]
        w2_weight = self.size_transform_w2(w2_weight).transpose(0, 1)  # [generate dim, feature size]
        w2_bias = self.size_transform_for_bias2_2(self.size_transform_for_bias2_1(w2_bias))  # [feature size]
        w2_weight = torch.tanh(w2_weight) * self.W_for_w2_weight + self.B_for_w2_weight
        w2_bias = torch.tanh(w2_bias) * self.W_for_w2_bias + self.B_for_w2_bias

        return w1_weight, w1_bias, w2_weight, w2_bias


if __name__ == "__main__":
    # used for test weight concat
    weight = []

    a = nn.Linear(3, 4)
    b = nn.Linear(3, 5)
    c = nn.Linear(3, 6)

    print(a.weight.data.size())
    print(a.bias.data.size())

    weight = torch.cat([a.weight.data, b.weight.data, c.weight.data], dim=0)
    print(weight.size())
