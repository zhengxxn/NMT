import torch.nn as nn
import copy
from module.embedding.embedding_with_positional_encoding import Embeddings
from module.encoder.transformer_encoder_with_classifier_adapter import \
    TransformerEncoderLayerWithClassifierAdapter, \
    TransformerEncoderWithClassifierAdapter
from module.decoder.transformer_decoder_with_classifier_adapter import \
    TransformerDecoderLayerWithClassifierAdapter, \
    TransformerDecoderWithClassifierAdapter
from module.generator.simple_generator import SimpleGenerator
from module.classifier.simple_classifier import SimpleClassifier
from module.classifier.cnn_classifier import CNNClassifier
from module.attention.multihead_attention import MultiHeadedAttention
from module.attention.multihead_attention_with_cache import MultiHeadedAttentionWithCache
from module.feedforward.positional_wise_feed_forward import PositionWiseFeedForward
from model.transformer_with_classifier_adapter import TransformerWithClassifierAdapter


def make_transformer_with_classifier_adapter(model_config, vocab):
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

    if model_config['classifier_type'] == 'simple':
        classifier = SimpleClassifier(input_dim=model_config['feature_size'],
                                      feature_size=model_config['classify_feature_size'],
                                      class_num=model_config['domain_class_num'], )
    elif model_config['classifier_type'] == 'cnn':
        classifier = CNNClassifier(num_class=model_config['domain_class_num'],
                                   input_dim=model_config['feature_size'],
                                   kernel_nums=model_config['kernel_nums'],
                                   kernel_sizes=model_config['kernel_sizes'],
                                   dropout_rate=model_config['dropout_rate'])
    else:
        classifier = None

    model = TransformerWithClassifierAdapter(
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
        encoder=TransformerEncoderWithClassifierAdapter(
            layer=TransformerEncoderLayerWithClassifierAdapter(feature_size=model_config['feature_size'],
                                                               self_attention_layer=copy.deepcopy(attention),
                                                               feed_forward_layer=copy.deepcopy(feed_forward),
                                                               domain_adapter_dict=model_config[
                                                                   'domain_adapter_dict'],
                                                               dropout_rate=model_config['dropout_rate'],
                                                               ),
            feature_size=model_config['feature_size'],
            num_layers=model_config['num_layers'],
            domain_label_dict=model_config['domain_dict']
        ),
        decoder=TransformerDecoderWithClassifierAdapter(
            layer=TransformerDecoderLayerWithClassifierAdapter(feature_size=model_config['feature_size'],
                                                               self_attention_layer=copy.deepcopy(
                                                                   attention_with_cache),
                                                               cross_attention_layer=copy.deepcopy(
                                                                   attention_with_cache),
                                                               feed_forward_layer=copy.deepcopy(feed_forward),
                                                               domain_adapter_dict=model_config[
                                                                   'domain_adapter_dict'],
                                                               dropout_rate=model_config['dropout_rate']),
            num_layers=model_config['num_layers'],
            feature_size=model_config['feature_size'],
            domain_label_dict=model_config['domain_dict'],
        ),
        generator=SimpleGenerator(feature_size=model_config['feature_size'],
                                  vocab_size=len(vocab['trg']),
                                  bias=model_config['generator_bias']),
        emb_classifier=copy.deepcopy(classifier),
        domain_mask=model_config['domain_mask'],
        vocab=vocab,
        share_decoder_embedding=model_config['share_decoder_embedding'],
        share_enc_dec_embedding=model_config['share_enc_dec_embedding'],
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for name, param in model.named_parameters():
        # if param.dim() > 1 and 'adapter' in name:
        #     nn.init.zeros_(param)
        if 'memory_score_bias' in name:
            nn.init.xavier_uniform_(param)

    return model
