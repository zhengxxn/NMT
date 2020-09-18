import torch.nn as nn
import copy
from module.embedding.embedding_with_positional_encoding import Embeddings
from module.encoder.transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from module.generator.simple_generator import SimpleGenerator
from module.attention.multihead_attention import MultiHeadedAttention
from module.feedforward.positional_wise_feed_forward import PositionWiseFeedForward
from model.transformer_language_model import TransformerLanguageModel


def make_transformer_language_model(model_config, vocab):
    attention = MultiHeadedAttention(
        head_num=model_config['head_num'],
        feature_size=model_config['feature_size'],
        dropout=model_config['dropout_rate']
    )
    feed_forward = PositionWiseFeedForward(
        input_dim=model_config['feature_size'],
        ff_dim=model_config['feedforward_dim'],
        dropout=model_config['dropout_rate']
    )

    model = TransformerLanguageModel(
        embedding_layer=Embeddings(
            vocab_size=len(vocab['text']),
            emb_size=model_config['feature_size'],
            dropout=model_config['dropout_rate'],
            max_len=5000
        ),
        decoder=TransformerEncoder(
            layer=TransformerEncoderLayer(feature_size=model_config['feature_size'],
                                          self_attention_layer=copy.deepcopy(attention),
                                          feed_forward_layer=copy.deepcopy(feed_forward),
                                          dropout_rate=model_config['dropout_rate'],
                                          layer_norm_rescale=model_config['layer_norm_rescale']),
            feature_size=model_config['feature_size'],
            num_layers=model_config['num_layers'],
            layer_norm_rescale=model_config['layer_norm_rescale'],
        ),
        generator=SimpleGenerator(feature_size=model_config['feature_size'],
                                  vocab_size=len(vocab['text']),
                                  bias=model_config['generator_bias']),
        vocab=vocab,
        share_embedding=model_config['share_embedding'],
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
