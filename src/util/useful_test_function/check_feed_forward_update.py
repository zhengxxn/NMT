from util.model_build.make_model.make_transformer import make_transformer
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
}

vocab = {
    'src': [1] * 36549,
    'trg': [1] * 36549,
}

original_model = make_transformer(model_config, vocab)
finetuned_model = make_transformer(model_config, vocab)

original_model_path = '/Users/zx/Documents/model/wmt2014-ende/6000-4/loss_best_model'
finetuned_model_path = '/Users/zx/Documents/model/wmt-2-bible-clean/finetune-feedforward/bleu_best/model'
device = torch.device('cpu')

model_dict = original_model.state_dict()
pretrained_dict = torch.load(original_model_path)
model_dict.update(pretrained_dict)
original_model.load_state_dict(model_dict)

model_dict = finetuned_model.state_dict()
pretrained_dict = torch.load(finetuned_model_path, map_location=device)
model_dict.update(pretrained_dict)
finetuned_model.load_state_dict(model_dict)

# print(original_model.state_dict())
for i in range(0, 6):

    original_encoder_layer_feedforward_key = \
        original_model.state_dict()['decoder.layers.' + str(i) + '.feed_forward_layer.w_1.weight']
    finetuned_encoder_layer_feedforward_key = \
        finetuned_model.state_dict()['decoder.layers.' + str(i) + '.feed_forward_layer.w_1.weight']
    original_encoder_layer_feedforward_value = \
        original_model.state_dict()['decoder.layers.' + str(i) + '.feed_forward_layer.w_2.weight']
    finetuned_encoder_layer_feedforward_value = \
        finetuned_model.state_dict()['decoder.layers.' + str(i) + '.feed_forward_layer.w_2.weight']

    key_sub = original_encoder_layer_feedforward_key - finetuned_encoder_layer_feedforward_key
    value_sub = (original_encoder_layer_feedforward_value - finetuned_encoder_layer_feedforward_value).transpose(-1, -2)
    # print(key_sub.size())
    # print(value_sub.size())

    # print(torch.sum(torch.norm(key_sub, dim=-1)))
    # print(torch.sum(torch.norm(value_sub, dim=-1)))
    key_norm = torch.norm(key_sub, dim=-1).numpy()
    value_norm = torch.norm(value_sub, dim=-1).numpy()

    print(np.average(key_norm))
    print(np.average(value_norm))

    plt.axis([0.0, 20.0, 0.0, 20.0])
    plt.scatter(key_norm, value_norm, s=2)
    plt.show()
