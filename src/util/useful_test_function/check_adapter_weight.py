from util.model_build.make_model.make_transformer_with_adapter import make_transformer_with_adapter
import torch
import numpy as np
import matplotlib.pyplot as plt

model_config = {
    'head_num': 8,
    'feature_size': 512,
    'dropout_rate': 0.1,
    'feedforward_dim': 2048,
    'num_layers': 6,
    'generator_bias': False,
    'share_decoder_embedding': True,
    'share_enc_dec_embedding': True,

    'adapter_dict': ['laws'],
    'adapter_bottleneck_size': [1024],
}

vocab = {
    'src': [1] * 36549,
    'trg': [1] * 36549,
}

original_model = make_transformer_with_adapter(model_config, vocab)

model_path = '/Users/zx/Documents/model/wmt-2-laws/adapter-1024/bleu_best/model'
device = torch.device('cpu')

model_dict = original_model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
model_dict.update(pretrained_dict)
original_model.load_state_dict(model_dict)

for name, param in original_model.named_parameters():
    print(name)

adapter_w1_weight = original_model.state_dict()['encoder.layers.0.adapters.laws.w_1.weight']
adapter_w1_bias = original_model.state_dict()['encoder.layers.0.adapters.laws.w_1.bias']
print(adapter_w1_weight.shape)

for i in range(0, adapter_w1_bias.size(0)):
    if adapter_w1_bias[i].item() > 0:
        print(adapter_w1_weight[i])
# t = 0
# for i in range(0, adapter_w1_weight.size(0)):
#     if torch.equal(adapter_w1_weight[i], torch.zeros_like(adapter_w1_weight[i])):
#         t += 1
#     else:
#         print(adapter_w1_bias[i])
# print(t)

# print(original_model.state_dict()['encoder.layers.0.adapters.laws.w_1.weight'])
# print(original_model.state_dict()['encoder.layers.5.adapters.laws.w_1.weight'])