from util.model_build.make_model.make_transformer_with_stacked_adapter import make_transformer_with_stacked_adapter
import torch
import torch.nn as nn

import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
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

    'domain_adapter_dict': {
        'iwslt': {
            'memory_count': 1024,
        }
    }
}

vocab = {
    'src': [1] * 36549,
    'trg': [1] * 36549,
}

model = make_transformer_with_stacked_adapter(model_config, vocab)

model_path = '/Users/zx/Documents/model/wmt-2-iwslt/adapter-1024/bleu_best/model'
device = torch.device('cpu')

model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

num_neg = 0
for i in range(0, 6):
    globals()['encoder_' + str(i) + '_w1_bias'] = \
        model.state_dict()['encoder.layers.' + str(i) + '.adapters.adapter_layers.iwslt.memory_score_bias'].numpy()
    globals()['encoder_' + str(i) + '_w1_weight'] = \
        model.state_dict()['encoder.layers.' + str(i) + '.adapters.adapter_layers.iwslt.w_1.weight'].numpy()

    globals()['decoder_' + str(i) + '_w1_bias'] = \
        model.state_dict()['decoder.layers.' + str(i) + '.adapter.adapter_layers.iwslt.memory_score_bias'].numpy()
    globals()['decoder_' + str(i) + '_w1_weight'] = \
        model.state_dict()['decoder.layers.' + str(i) + '.adapter.adapter_layers.iwslt.w_1.weight'].numpy()

    globals()['encoder_' + str(i) + '_ffn_layer_norm_weight'] = \
        model.state_dict()['encoder.layers.' + str(i) + '.adapters.sublayer_connection_for_adapter.iwslt.norm.weight'].numpy()
    globals()['encoder_' + str(i) + '_ffn_layer_norm_bias'] = \
        model.state_dict()['encoder.layers.' + str(i) + '.adapters.sublayer_connection_for_adapter.iwslt.norm.bias'].numpy()

    globals()['decoder_' + str(i) + '_ffn_layer_norm_weight'] = \
        model.state_dict()['decoder.layers.' + str(i) + '.adapter.sublayer_connection_for_adapter.iwslt.norm.weight'].numpy()
    globals()['decoder_' + str(i) + '_ffn_layer_norm_bias'] = \
        model.state_dict()['decoder.layers.' + str(i) + '.adapter.sublayer_connection_for_adapter.iwslt.norm.bias'].numpy()
#
num_pos_s = [0] * 12
position_pos_s = [[0]*1024 for i in range(0, 12)]
for n in range(0, 100):
    y = torch.randn(512)
    for j in range(0, 6):
        layer_norm = nn.LayerNorm(512)
        layer_norm.weight.data = torch.from_numpy(globals()['encoder_'+str(j)+'_ffn_layer_norm_weight'])
        layer_norm.bias.data = torch.from_numpy(globals()['encoder_'+str(j)+'_ffn_layer_norm_bias'])
        # x = torch.randn(512)
        x = layer_norm(y)
        # print(LA.norm(x.detach().numpy()))
        w = globals()['encoder_' + str(j) + '_w1_weight']
        b = globals()['encoder_' + str(j) + '_w1_bias']
        # print(w.shape)
        # print(b.shape)
        score = torch.matmul(x.unsqueeze(0), torch.from_numpy(w).t())
        # print(score.shape)
        # print(score.shape)
        score = score.squeeze(0) - torch.from_numpy(b).squeeze(0)
        num_pos = 0
        for i in range(score.size(0)):
            if score[i].item() > 0:
                num_pos += 1
                position_pos_s[j][i] += 1
                # print(score[i].item())
        num_pos_s[j] += num_pos
        # print(num_pos)

    for j in range(0, 6):
        layer_norm = nn.LayerNorm(512)
        layer_norm.weight.data = torch.from_numpy(globals()['decoder_'+str(j)+'_ffn_layer_norm_weight'])
        layer_norm.bias.data = torch.from_numpy(globals()['decoder_'+str(j)+'_ffn_layer_norm_bias'])
        # x = torch.randn(512)
        x = layer_norm(y)
        # print(LA.norm(x.detach().numpy()))
        w = globals()['decoder_' + str(j) + '_w1_weight']
        b = globals()['decoder_' + str(j) + '_w1_bias']
        score = torch.matmul(x.unsqueeze(0), torch.from_numpy(w).t())
        # print(score.shape)
        score = score.squeeze(0) - torch.from_numpy(b).squeeze(0)
        num_pos = 0
        for i in range(score.size(0)):
            if score[i].item() > 0:
                num_pos += 1
                position_pos_s[6+j][i] += 1
                # print(score[i].item())
        num_pos_s[6+j] += num_pos
        # print(num_pos)

num_pos_s = [v / 100 for v in num_pos_s]
print(np.average(num_pos_s))
print(num_pos_s)
# with open('/Users/zx/Documents/model/wmt2014-ende/6000-4/ffn_used.txt', 'w') as f:
#     position_pos_s = [[str(count) for count in pos] for pos in position_pos_s]
#     position_pos_s = [' '.join(pos) for pos in position_pos_s]
#     position_pos_s = '\n'.join(position_pos_s)
#     f.write(position_pos_s)
