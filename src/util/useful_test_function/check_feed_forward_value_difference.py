from util.model_build.make_model.make_transformer import make_transformer
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
}

vocab = {
    'src': [1] * 36549,
    'trg': [1] * 36549,
}

model = make_transformer(model_config, vocab)

# model_path = '/Users/zx/Documents/model/wmt-2-bible-clean/finetune-feedforward/bleu_best/model'
model_path = '/Users/zx/Documents/model/wmt2014-ende/6000-4/loss_best_model'
# original_model_path = '/Users/zx/Documents/model/wmt2014-ende/6000-4/loss_best_model'
# finetuned_model_path = '/Users/zx/Documents/model/wmt-2-bible-clean/finetune-feedforward/bleu_best/model'
device = torch.device('cpu')

model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

num_neg = 0
# encoder_0_w2_bias = model.state_dict()['decoder.layers.0.feed_forward_layer.w_2.bias'].numpy()
# encoder_1_w2_bias = model.state_dict()['decoder.layers.1.feed_forward_layer.w_2.bias'].numpy()
# encoder_2_w2_bias = model.state_dict()['decoder.layers.2.feed_forward_layer.w_2.bias'].numpy()
# encoder_3_w2_bias = model.state_dict()['decoder.layers.3.feed_forward_layer.w_2.bias'].numpy()
# encoder_4_w2_bias = model.state_dict()['decoder.layers.4.feed_forward_layer.w_2.bias'].numpy()
# encoder_5_w2_bias = model.state_dict()['decoder.layers.5.feed_forward_layer.w_2.bias'].numpy()
for i in range(0, 6):
    globals()['encoder_' + str(i) + '_w1_bias'] = \
        model.state_dict()['encoder.layers.' + str(i) + '.feed_forward_layer.w_1.bias'].numpy()
    globals()['decoder_' + str(i) + '_w2_bias'] = \
        model.state_dict()['decoder.layers.' + str(i) + '.feed_forward_layer.w_2.bias'].numpy()
    globals()['encoder_' + str(i) + '_w1_weight'] = \
        model.state_dict()['encoder.layers.' + str(i) + '.feed_forward_layer.w_1.weight'].numpy()

    globals()['decoder_' + str(i) + '_w1_weight'] = \
        model.state_dict()['decoder.layers.' + str(i) + '.feed_forward_layer.w_1.weight'].numpy()
    globals()['decoder_' + str(i) + '_w1_bias'] = \
        model.state_dict()['decoder.layers.' + str(i) + '.feed_forward_layer.w_1.bias'].numpy()

    globals()['encoder_' + str(i) + '_ffn_layer_norm_weight'] = \
        model.state_dict()['encoder.layers.' + str(i) + '.sub_layer_connections.1.norm.weight'].numpy()
    globals()['encoder_' + str(i) + '_ffn_layer_norm_bias'] = \
        model.state_dict()['encoder.layers.' + str(i) + '.sub_layer_connections.1.norm.bias'].numpy()

    globals()['decoder_' + str(i) + '_ffn_layer_norm_weight'] = \
        model.state_dict()['decoder.layers.' + str(i) + '.sublayer.norm.weight'].numpy()
    globals()['decoder_' + str(i) + '_ffn_layer_norm_bias'] = \
        model.state_dict()['decoder.layers.' + str(i) + '.sublayer.norm.bias'].numpy()
#
num_pos_s = [0] * 12
position_pos_s = [[0]*2048 for i in range(0, 12)]
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
        score = torch.matmul(x.unsqueeze(0), torch.from_numpy(w).t())
        # print(score.shape)
        score = score.squeeze(0) - torch.from_numpy(b)
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
        score = score.squeeze(0) - torch.from_numpy(b)
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
with open('/Users/zx/Documents/model/wmt2014-ende/6000-4/ffn_used.txt', 'w') as f:
    position_pos_s = [[str(count) for count in pos] for pos in position_pos_s]
    position_pos_s = [' '.join(pos) for pos in position_pos_s]
    position_pos_s = '\n'.join(position_pos_s)
    f.write(position_pos_s)

# w = globals()['decoder_' + str(5) + '_w1_weight']
# for i in range(0, w.shape[0]):
#     print(LA.norm(w[i]))


# for i in range(0, 6):
#     for j in range(0, 6):
#         print(i, j, cosine_similarity(globals()['decoder_' + str(i) + '_w2_bias'].reshape(1, -1),
#                                       globals()['decoder_' + str(j) + '_w2_bias'].reshape(1, -1)))

# print(globals()['encoder_5_w2_bias'])

# similarity = cosine_similarity(encoder_4_w2_bias.reshape(1, -1), encoder_5_w2_bias.reshape(1, -1))
# print(similarity)


# print(encoder_w1_bias.shape)
# for v in range(encoder_w1_bias.size(0)):
#     bias = encoder_w1_bias[v].item()
#     if bias < 0:
#         num_neg += 1
# print(num_neg)
# print(encoder_w1_bias[v])

# print(original_model.state_dict())
# for i in range(0, 6):

# encoder_layer_feedforward_key = \
# model.state_dict()['decoder.layers.' + str(i) + '.feed_forward_layer.w_1.weight']
# encoder_layer_feedforward_value = \
# model.state_dict()['decoder.layers.' + str(i) + '.feed_forward_layer.w_2.weight'].transpose(-1, -2)

# encoder_layer_feedforward_key = encoder_layer_feedforward_key.numpy()
# encoder_layer_feedforward_value = encoder_layer_feedforward_value.numpy()
# print(key_sub.size())
# print(value_sub.size())
# print(torch.sum(torch.norm(key_sub, dim=-1)))
# print(torch.sum(torch.norm(value_sub, dim=-1)))
# print(np.average(key_norm))
# print(np.average(encoder_layer_feedforward_value))

# avg_value = np.mean(encoder_layer_feedforward_value, axis=0)
# print(avg_value)
# print(encoder_layer_feedforward_value.shape)
# for idx in range(0, encoder_layer_feedforward_value.shape[0]):
#     x = encoder_layer_feedforward_value[idx]
#     similarity = cosine_similarity(x.reshape(1, -1), avg_value.reshape(1, -1))
#     print(similarity)

# plt.axis([0.0, 20.0, 0.0, 20.0])
# plt.scatter(key_norm, value_norm, s=2)
# plt.show()
