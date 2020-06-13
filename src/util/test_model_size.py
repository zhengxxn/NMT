from util.model_build.make_model.make_transformer import make_transformer
from util.model_build.make_model.make_transformer_with_adapter import make_transformer_with_adapter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


model_config = {
    'head_num': 8,
    'feature_size': 512,
    'dropout_rate': 0.1,
    'feedforward_dim': 2048,
    'num_layers': 6,
    'generator_bias': False,
    'share_decoder_embedding': True,
    'share_enc_dec_embedding': True,

    'adapter_dict': ['iwslt'],
    'adapter_bottleneck_size': [2048],
}


vocab = {
    'src': [1] * 37000,
    'trg': [1] * 37000,
}

model = make_transformer(model_config, vocab)
#
for name, param in model.named_parameters():
    print(name)

# print(count_parameters(model))

# model2 = make_transformer_with_adapter(model_config, vocab)
# print(count_parameters(model2))

# count_model_1 = count_parameters(model)
# count_model_2 = count_parameters(model2)
# print(count_model_1)
# print(count_model_2)
# print((count_model_2 - count_model_1) / count_model_1)