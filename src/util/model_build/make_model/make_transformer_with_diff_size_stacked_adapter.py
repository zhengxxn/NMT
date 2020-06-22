import torch.nn as nn
import copy
from module.embedding.embedding_with_positional_encoding import Embeddings

from module.encoder.transformer_encoder_with_diff_size_stacked_adapter import \
    TransformerEncoderLayerWithDiffSizeStackedAdapter, \
    TransformerEncoderWithDiffSizeStackedAdapter
from module.decoder.transformer_decoder_with_diff_size_stacked_adapter import \
    TransformerDecoderLayerWithDiffSizeStackedAdapter, \
    TransformerDecoderWithDiffSizeStackedAdapter

from module.generator.simple_generator import SimpleGenerator
from module.attention.multihead_attention import MultiHeadedAttention
from module.attention.multihead_attention_with_cache import MultiHeadedAttentionWithCache
from module.feedforward.positional_wise_feed_forward import PositionWiseFeedForward
from model.transformer_with_adapter import TransformerWithAdapter


def make_transformer_with_diff_size_stacked_adapter(model_config, vocab):
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

    encoder_layers = []
    decoder_layers = []
    for i in range(0, model_config['num_layers']):
        adapter_dict = {}
        if 'encoder.' + str(i) in model_config['domain_adapter_dict'].keys():
            adapter_dict = model_config['domain_adapter_dict']['encoder.' + str(i)]
        else:
            adapter_dict = model_config['domain_adapter_dict']['default']

        encoder_layer = TransformerEncoderLayerWithDiffSizeStackedAdapter(feature_size=model_config['feature_size'],
                                                                          self_attention_layer=copy.deepcopy(attention),
                                                                          feed_forward_layer=copy.deepcopy(
                                                                              feed_forward),
                                                                          domain_adapter_dict=adapter_dict,
                                                                          dropout_rate=model_config['dropout_rate'], )
        encoder_layers.append(encoder_layer)

        adapter_dict = {}
        if 'decoder.' + str(i) in model_config['domain_adapter_dict'].keys():
            adapter_dict = model_config['domain_adapter_dict']['decoder.' + str(i)]
        else:
            adapter_dict = model_config['domain_adapter_dict']['default']

        decoder_layer = TransformerDecoderLayerWithDiffSizeStackedAdapter(feature_size=model_config['feature_size'],
                                                                          self_attention_layer=copy.deepcopy(
                                                                              attention_with_cache),
                                                                          cross_attention_layer=copy.deepcopy(
                                                                              attention_with_cache),
                                                                          feed_forward_layer=copy.deepcopy(
                                                                              feed_forward),
                                                                          domain_adapter_dict=adapter_dict,
                                                                          dropout_rate=model_config['dropout_rate'])
        decoder_layers.append(decoder_layer)

    model = TransformerWithAdapter(
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
        encoder=TransformerEncoderWithDiffSizeStackedAdapter(
            layers=encoder_layers,
            feature_size=model_config['feature_size'],
            num_layers=model_config['num_layers'],
        ),
        decoder=TransformerDecoderWithDiffSizeStackedAdapter(
            layers=decoder_layers,
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

    # for name, param in model.named_parameters():
    #     print(name)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for name, param in model.named_parameters():
        # if param.dim() > 1 and 'adapter' in name:
        #     nn.init.zeros_(param)
        if 'memory_score_bias' in name:
            nn.init.xavier_uniform_(param)

    return model
