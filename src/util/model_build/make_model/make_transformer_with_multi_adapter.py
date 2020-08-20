import torch.nn as nn
import copy
from module.embedding.embedding_with_positional_encoding import Embeddings
from module.encoder.transformer_encoder_with_parallel_adapter import TransformerEncoderWithParallelAdapter, \
    TransformerEncoderLayerWithParallelAdapter
from module.decoder.transformer_decoder_with_parallel_adapter import TransformerDecoderWithParallelAdapter, \
    TransformerDecoderLayerWithParallelAdapter
from module.generator.simple_generator import SimpleGenerator
from module.attention.multihead_attention import MultiHeadedAttention
from module.attention.multihead_attention_with_cache import MultiHeadedAttentionWithCache
from module.feedforward.positional_wise_feed_forward import PositionWiseFeedForward
from module.adapter.parallel_adapter import ParallelAdapter
from model.transformer_with_multi_adapter import TransformerWithMultiAdapter


def make_transformer_with_parallel_adapter(model_config, vocab):

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
    parallel_adapter_layer = ParallelAdapter(domain_adapter_dict=model_config['domain_adapter_dict'],
                                             feature_size=model_config['feature_size'],
                                             dropout_rate=model_config['dropout_rate'],
                                             max_domain_num=model_config['max_domain_num'],
                                             domain_idx_dict=model_config['domain_idx_dict'],
                                             weighted_sum=model_config['adapter_weighted_sum'],)

    model = TransformerWithMultiAdapter(
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
        encoder=TransformerEncoderWithParallelAdapter(
            layer=TransformerEncoderLayerWithParallelAdapter(feature_size=model_config['feature_size'],
                                                             self_attention_layer=copy.deepcopy(attention),
                                                             feed_forward_layer=copy.deepcopy(feed_forward),
                                                             parallel_adapter_layer=copy.deepcopy(parallel_adapter_layer),
                                                             dropout_rate=model_config['dropout_rate'],
                                                             layer_norm_rescale=model_config['layer_norm_rescale']),
            feature_size=model_config['feature_size'],
            num_layers=model_config['num_layers'],
            layer_norm_rescale=model_config['layer_norm_rescale']
        ),
        decoder=TransformerDecoderWithParallelAdapter(
            layer=TransformerDecoderLayerWithParallelAdapter(feature_size=model_config['feature_size'],
                                                             self_attention_layer=copy.deepcopy(attention_with_cache),
                                                             cross_attention_layer=copy.deepcopy(attention_with_cache),
                                                             feed_forward_layer=copy.deepcopy(feed_forward),
                                                             parallel_adapter_layer=copy.deepcopy(parallel_adapter_layer),
                                                             dropout_rate=model_config['dropout_rate'],
                                                             layer_norm_rescale=model_config['layer_norm_rescale']),
            num_layers=model_config['num_layers'],
            feature_size=model_config['feature_size'],
            layer_norm_rescale=model_config['layer_norm_rescale']
        ),
        generator=SimpleGenerator(feature_size=model_config['feature_size'],
                                  vocab_size=len(vocab['trg']),
                                  bias=model_config['generator_bias']),
        vocab=vocab,
        share_decoder_embedding=model_config['share_decoder_embedding'],
        share_enc_dec_embedding=model_config['share_enc_dec_embedding'],
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # for name, param in model.named_parameters():
    #     if param.dim() > 1 and 'adapter' in name:
    #         nn.init.zeros_(param)

    return model
