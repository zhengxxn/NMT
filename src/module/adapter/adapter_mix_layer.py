import torch
import torch.nn as nn


class AdapterMixLayer(nn.Module):
    """
    This Module will Mix the different domain adapters
    """

    def __init__(self,
                 used_adapters: list,
                 feature_size,
                 dropout_rate,
                 classifier_dict: dict = None):

        super().__init__()

        self.used_adapters = used_adapters
        self.dropout = nn.Dropout(dropout_rate)

        if classifier_dict is not None:

            self.classifier = nn.Sequential(nn.Linear(feature_size, classifier_dict['bottleneck_dim']),
                                            nn.Tanh(),
                                            nn.Linear(classifier_dict['bottleneck_dim'], len(used_adapters)))

            self.classifier_type = classifier_dict['type']  # sent or token

    def forward(self, x, adapter_layers, sublayer_connection_for_adapter, x_mask=None):

        adapter_outputs = []

        for domain in self.used_adapters:
            # domain_adapter_output = sublayer_connection_for_adapter[domain].wo_dropout_residual_forward(x, adapter_layers[domain])
            domain_adapter_output = sublayer_connection_for_adapter[domain](x, adapter_layers[domain])
            adapter_outputs.append(domain_adapter_output)

        adapter_outputs = torch.stack(adapter_outputs, dim=-1)  # [B, L, H, D_N]

        # mixture weights
        if self.classifier_type == 'token':
            logits = self.classifier(self.dropout(x))  # [B, L, D_N]
            weights = logits.softmax(-1)  # [B, L, D_N]
            weights = weights.unsqueeze(-1)  # [B, L, D_N, 1]
            adapter_outputs = torch.matmul(adapter_outputs, weights).squeeze(-1)  # [B, L, H]

        else:
            # type is sent
            # x_mask [B, L, 1]
            # sent mix is only can be used in encoder
            # print(x_mask.size())
            # print(x.size())
            x_mask = x_mask.transpose(-1, -2)
            mask_x = x.masked_fill(x_mask == 0, 0)
            sent_length = torch.sum(x_mask, dim=-2)  # [batch size, 1]
            sent_representation = torch.sum(mask_x, dim=-2)  # [batch size, input dim]
            sent_representation = sent_representation / sent_length.type_as(sent_representation)
            logits = self.classifier(self.dropout(sent_representation))  # [B, D_N]
            weights = logits.softmax(-1)  # [B, D_N]
            weights = weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, D_N, 1]
            adapter_outputs = torch.matmul(adapter_outputs, weights).squeeze(-1)  # [B, L, H]

        return adapter_outputs, logits
        # return x + self.dropout(adapter_outputs), logits


if __name__ == "__main__":
    # used for test weight concat

    random_mask = torch.randn(0, 2, (3, 4))

