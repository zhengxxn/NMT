import torch.nn as nn
import copy
from module.embedding.embedding_with_positional_encoding import Embeddings
from module.encoder.transformer_encoder_with_mix_adapter_update import \
    TransformerEncoderLayerWithMixAdapter, \
    TransformerEncoderWithMixAdapter
from module.decoder.transformer_decoder_with_mix_adapter_update import \
    TransformerDecoderLayerWithMixAdapter, \
    TransformerDecoderWithMixAdapter
from module.generator.simple_generator import SimpleGenerator
from module.attention.multihead_attention import MultiHeadedAttention
from module.attention.multihead_attention_with_cache import MultiHeadedAttentionWithCache
from module.feedforward.positional_wise_feed_forward import PositionWiseFeedForward
from module.adapter.mixture_of_adapter_update import MixtureOfAdapter
from model.transformer_with_mix_adapter_update import TransformerWithMixAdapter


def make_transformer_with_mix_adapter_update(model_config, vocab):
    attention = MultiHeadedAttention(
        head_num=model_config['head_num'],
        feature_size=model_config['feature_size'],
        dropout=model_config['dropout_rate']
    )
    attention_with_cache = MultiHeadedAttentionWithCache(
        head_num=model_config['head_num'],
        feature_size=model_config['feature_size'],
        dropout=model_config['dropout_rate']
    )
    feed_forward = PositionWiseFeedForward(
        input_dim=model_config['feature_size'],
        ff_dim=model_config['feedforward_dim'],
        dropout=model_config['dropout_rate']
    )

    adapter = None
    if 'domain_adapter_dict' in model_config:
        adapter = MixtureOfAdapter(
            domain_adapter_dict=model_config['domain_adapter_dict'],
            feature_size=model_config['feature_size'],
            dropout_rate=model_config['dropout_rate'],
            # domain_list=model_config['domain_list'],
            # domain_inner_gate_list=model_config['domain_inner_gate_list'],
            # gate_activate_func=model_config['adapter_gate_activate'],
            # stack_between_adapter_and_experts=model_config['stack_between_adapter_and_experts'] if
            # 'stack_between_adapter_and_experts' in model_config else False
        )

    enc_adapter = adapter
    if 'enc_domain_adapter_dict' in model_config:
        enc_adapter = MixtureOfAdapter(
            domain_adapter_dict=model_config['enc_domain_adapter_dict'],
            feature_size=model_config['feature_size'],
            dropout_rate=model_config['dropout_rate'],
        )

    dec_adapter = adapter
    if 'dec_domain_adapter_dict' in model_config:
        dec_adapter = MixtureOfAdapter(
            domain_adapter_dict=model_config['dec_domain_adapter_dict'],
            feature_size=model_config['feature_size'],
            dropout_rate=model_config['dropout_rate'],
        )

    model = TransformerWithMixAdapter(
        src_embedding_layer=Embeddings(
            vocab_size=len(vocab['src']),
            emb_size=model_config['feature_size'],
            dropout=model_config['dropout_rate'],
            max_len=5000
        ),
        trg_embedding_layer=Embeddings(
            vocab_size=len(vocab['trg']),
            emb_size=model_config['feature_size'],
            dropout=model_config['dropout_rate'],
            max_len=5000
        ),
        encoder=TransformerEncoderWithMixAdapter(
            layer=TransformerEncoderLayerWithMixAdapter(feature_size=model_config['feature_size'],
                                                        self_attention_layer=copy.deepcopy(attention),
                                                        feed_forward_layer=copy.deepcopy(feed_forward),
                                                        adapters=copy.deepcopy(enc_adapter),
                                                        dropout_rate=model_config['dropout_rate'],
                                                        ),
            feature_size=model_config['feature_size'],
            num_layers=model_config['num_layers'],
        ),
        decoder=TransformerDecoderWithMixAdapter(
            layer=TransformerDecoderLayerWithMixAdapter(feature_size=model_config['feature_size'],
                                                        self_attention_layer=copy.deepcopy(
                                                            attention_with_cache),
                                                        cross_attention_layer=copy.deepcopy(
                                                            attention_with_cache),
                                                        feed_forward_layer=copy.deepcopy(feed_forward),
                                                        adapters=copy.deepcopy(dec_adapter),
                                                        dropout_rate=model_config['dropout_rate'],
                                                        ),
            num_layers=model_config['num_layers'],
            feature_size=model_config['feature_size'],
        ),
        generator=SimpleGenerator(feature_size=model_config['feature_size'],
                                  vocab_size=len(vocab['trg']),
                                  bias=model_config['generator_bias']),
        vocab=vocab,
        share_decoder_embedding=model_config['share_decoder_embedding'],
        share_enc_dec_embedding=model_config['share_enc_dec_embedding'],
    )

    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)

    for name, param in model.named_parameters():
        if param.dim() > 1 and 'W' not in name and 'B' not in name:
            nn.init.xavier_uniform_(param)

        else:
            if param.dim() > 1:
                print('module self init', name, param.size())

    # for name, param in model.named_parameters():
    #     if param.dim() > 1 and 'adapter' in name:
    #     #     nn.init.zeros_(param)
    #     # if 'adapter' in name:
    #         print('init adapter', name)
    #         nn.init.xavier_uniform_(param, 0.001)

    return model
