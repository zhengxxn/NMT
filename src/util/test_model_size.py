from util.model_build.make_model.make_transformer import make_transformer
from util.model_build.make_model.make_transformer_with_adapter import make_transformer_with_adapter
from util.model_build.make_model.make_transformer_with_parallel_adapter import make_transformer_with_parallel_adapter
from util.model_build.make_model.make_transformer_with_stacked_adapter import make_transformer_with_stacked_adapter


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
    'domain_adapter_dict': {
        'iwslt': {
            'memory_count': 1024
        }
    },

    'adapter_dict': ['iwslt'],
    'adapter_bottleneck_size': [2048],
}


vocab = {
    'src': [1] * 37000,
    'trg': [1] * 37000,
}

model = make_transformer_with_stacked_adapter(model_config, vocab)
#

for name, param in model.named_parameters():
    print(name)

# count_feed_forward = 0
# for name, param in model.named_parameters():
#     if 'feed_forward' in name:
#         count_feed_forward += param.numel()
# print(count_feed_forward)
#
# count_self_attn = 0
# for name, param in model.named_parameters():
#     if 'self_attention' in name:
#         count_self_attn += param.numel()
# print(count_self_attn)
#
# count_cross_attn = 0
# for name, param in model.named_parameters():
#     if 'cross_attention' in name:
#         count_cross_attn += param.numel()
# print(count_cross_attn)
#
# count_embedding = 0
# for name, param in model.named_parameters():
#     if 'embedding' in name:
#         count_embedding += param.numel()
# print(count_embedding)


# count_t = count_parameters(model)
print(count_parameters(model))

# print(count_t)
# print(count_feed_forward / count_t)
# print(count_self_attn / count_t)
# print(count_cross_attn / count_t)
# print(count_embedding / count_t)
# print(count_feed_forward / (count_t - count_embedding))


# model2 = make_transformer_with_adapter(model_config, vocab)
# print(count_parameters(model2))

# count_model_1 = count_parameters(model)
# count_model_2 = count_parameters(model2)
# print(count_model_1)
# print(count_model_2)
# print((count_model_2 - count_model_1) / count_model_1)