import torch.nn as nn
import copy
from module.embedding.split_position_embedding import Embeddings
from module.encoder.transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from module.decoder.transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from module.generator.simple_generator import SimpleGenerator
from module.attention.multihead_attention import MultiHeadedAttention
from module.attention.multihead_attention_with_cache import MultiHeadedAttentionWithCache
from module.feedforward.positional_wise_feed_forward import PositionWiseFeedForward
from model.transformer_with_split_postion import TransformerWithSplitPosition


def make_transformer_with_split_position(model_config, vocab):
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

    model = TransformerWithSplitPosition(
        src_embedding_layer=Embeddings(emb_size=model_config['feature_size'],
                                       vocab_size=len(vocab['src']),
                                       dropout=model_config['dropout_rate'],
                                       linear_combination=model_config['position_linear_combination'],
                                       ),
        trg_embedding_layer=Embeddings(emb_size=model_config['feature_size'],
                                       vocab_size=len(vocab['trg']),
                                       dropout=model_config['dropout_rate'],
                                       linear_combination=model_config['position_linear_combination'],
                                       ),
        encoder=TransformerEncoder(
            layer=TransformerEncoderLayer(feature_size=model_config['feature_size'],
                                          self_attention_layer=copy.deepcopy(attention),
                                          feed_forward_layer=copy.deepcopy(feed_forward),
                                          dropout_rate=model_config['dropout_rate']),
            feature_size=model_config['feature_size'],
            num_layers=model_config['num_layers'],
        ),
        decoder=TransformerDecoder(
            layer=TransformerDecoderLayer(feature_size=model_config['feature_size'],
                                          self_attention_layer=copy.deepcopy(attention_with_cache),
                                          cross_attention_layer=copy.deepcopy(attention_with_cache),
                                          feed_forward_layer=copy.deepcopy(feed_forward),
                                          dropout_rate=model_config['dropout_rate']),
            num_layers=model_config['num_layers'],
            feature_size=model_config['feature_size'],
        ),
        generator=SimpleGenerator(feature_size=model_config['feature_size'], vocab_size=len(vocab['trg'])),
        vocab=vocab,
        share_decoder_embedding=model_config['share_decoder_embedding'],
        share_enc_dec_embedding=model_config['share_enc_dec_embedding'],
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
