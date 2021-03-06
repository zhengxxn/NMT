import torch.nn as nn
import copy
from module.embedding.embedding_with_positional_encoding import Embeddings
from module.encoder.transformer_encoder_with_houlsby_adapter import TransformerEncoder, TransformerEncoderLayer
from module.decoder.transformer_decoder_with_houlsby_adapter import TransformerDecoder, TransformerDecoderLayer
from module.generator.simple_generator import SimpleGenerator
from module.attention.multihead_attention import MultiHeadedAttention
from module.attention.multihead_attention_with_cache import MultiHeadedAttentionWithCache
from module.feedforward.positional_wise_feed_forward import PositionWiseFeedForward
from model.transformer_with_adapter import TransformerWithAdapter


def make_transformer_with_houlsby_adapter(model_config, vocab):
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
        encoder=TransformerEncoder(
            layer=TransformerEncoderLayer(feature_size=model_config['feature_size'],
                                          self_attention_layer=copy.deepcopy(attention),
                                          feed_forward_layer=copy.deepcopy(feed_forward),
                                          adapter_dict=model_config['adapter_dict'],
                                          adapter_bottleneck_size=model_config['adapter_bottleneck_size'],
                                          dropout_rate=model_config['dropout_rate'], ),
            feature_size=model_config['feature_size'],
            num_layers=model_config['num_layers'],
            adapter_dict=model_config['adapter_dict'],
        ),
        decoder=TransformerDecoder(
            layer=TransformerDecoderLayer(feature_size=model_config['feature_size'],
                                          self_attention_layer=copy.deepcopy(attention_with_cache),
                                          cross_attention_layer=copy.deepcopy(attention_with_cache),
                                          feed_forward_layer=copy.deepcopy(feed_forward),
                                          adapter_dict=model_config['adapter_dict'],
                                          adapter_bottleneck_size=model_config['adapter_bottleneck_size'],
                                          dropout_rate=model_config['dropout_rate']),
            num_layers=model_config['num_layers'],
            feature_size=model_config['feature_size'],
            adapter_dict=model_config['adapter_dict'],
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

    for name, param in model.named_parameters():
        if 'adapter' in name:
            nn.init.zeros_(param)
        # if param.dim() > 1 and 'adapter' in name:
        #     nn.init.zeros_(param)

    # init layer norm
    # model.encoder.init_adapter_parameter()
    # model.decoder.init_adapter_parameter()

    return model
