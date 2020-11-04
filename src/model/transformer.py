import torch
import torch.nn as nn
from util.batch.transformer_batch import TrgTestBatch


class Transformer(nn.Module):

    def __init__(self,
                 src_embedding_layer,
                 trg_embedding_layer,
                 encoder,
                 decoder,
                 generator,
                 vocab,
                 share_decoder_embedding=False,
                 share_enc_dec_embedding=False,):
        super().__init__()

        self.src_embedding_layer = src_embedding_layer
        self.trg_embedding_layer = trg_embedding_layer
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

        self.vocab = vocab

        if share_enc_dec_embedding:
            assert vocab['src'] == vocab['trg']
            self.src_embedding_layer.embedding_layer.weight = \
                self.trg_embedding_layer.embedding_layer.weight
        if share_decoder_embedding:
            self.generator.proj.weight = self.trg_embedding_layer.embedding_layer.weight

    def forward(self, src, src_mask, trg_input, trg, trg_mask):
        # trg_input = batch.trg_input
        # trg_mask = batch.trg_mask
        # trg = batch.trg

        decoder_state = self.prepare_for_decode(src, src_mask)

        decoder_logits, _, _ = self.decode(trg_input, trg_mask, decoder_state)

        logit = self.generator(decoder_logits, return_logit=True)
        log_probs = torch.log_softmax(logit, -1)
        # loss = self.criterion(
        #     input=log_probs.contiguous().view(-1, log_probs.size(-1)),
        #     target=trg.contiguous().view(-1)
        # )

        return {'log_prob': log_probs, 'logit': logit}

    def prepare_for_decode(self, src, src_mask, test=None):
        # src = batch.src
        # src_mask = batch.src_mask

        assert src_mask is not None

        encoder_state = self.encode(src, src_mask)
        decoder_state = {
            'memory': encoder_state['memory'],
            'src_mask': src_mask,
            'input': None,
            'enc_attn_cache': None,
            'self_attn_cache': None,
        }
        return decoder_state

    def decode_step(self, input, decode_state):
        # input_embedding = self.trg_embedding_layer(input)

        if decode_state['input'] is None:
            # [batch size, 1]
            decode_state['input'] = input.unsqueeze(1)
        else:
            # [batch size * beam size, seq len]
            decode_state['input'] = torch.cat((decode_state['input'], input.unsqueeze(1)), dim=-1)

        batch = TrgTestBatch(decode_state['input'], pad=self.vocab['trg'].stoi['<pad>'])

        # [batch size, seq len]
        # trg_input = batch.trg_input[:, -1:]
        trg_input = batch.trg_input

        # [batch size, seq len, seq len]
        # todo: wait for improvement
        trg_mask = batch.trg_mask[:, -1:]
        # trg_mask = batch.trg_mask

        # if 'enc attn cache' not in decode state, it means in decoding
        # else it means before decoding, then it must be None
        if 'enc_attn_cache' not in decode_state.keys():
            enc_attn_cache_list = []
            self_attn_cache_list = []

            for i in range(self.decoder.num_layers):
                enc_attn_cache_list.append((decode_state['enc_attn_cache_' + str(i) + '_key'],
                                            decode_state['enc_attn_cache_' + str(i) + '_value']))
                self_attn_cache_list.append((decode_state['self_attn_cache_' + str(i) + '_key'],
                                             decode_state['self_attn_cache_' + str(i) + '_value']))

            decode_state['enc_attn_cache'] = enc_attn_cache_list
            decode_state['self_attn_cache'] = self_attn_cache_list
        else:
            assert decode_state['enc_attn_cache'] is None

        logits, new_enc_attn_cache, new_self_attn_cache = self.decode(trg_input, trg_mask, decode_state, test=True)

        logits = logits[:, -1, :]
        log_prob = self.generator(logits)

        for i in range(len(new_enc_attn_cache)):
            decode_state['enc_attn_cache_' + str(i) + '_key'] = new_enc_attn_cache[i][0]
            decode_state['enc_attn_cache_' + str(i) + '_value'] = new_enc_attn_cache[i][1]
            decode_state['self_attn_cache_' + str(i) + '_key'] = new_self_attn_cache[i][0]
            decode_state['self_attn_cache_' + str(i) + '_value'] = new_self_attn_cache[i][1]

        decode_state.pop('enc_attn_cache', None)
        decode_state.pop('self_attn_cache', None)

        return log_prob, decode_state

    def encode(self, src, src_mask):
        input_embedding = self.src_embedding_layer(src)
        encoder_state = self.encoder(input_embedding, src_mask)
        return encoder_state

    def decode(self, trg_input, trg_mask, decoder_state, test=False):
        trg_input_embedding = self.trg_embedding_layer(trg_input)

        if test:
            trg_input_embedding = trg_input_embedding[:, -1:]

        return self.decoder(trg_input_embedding,
                            decoder_state['memory'],
                            decoder_state['src_mask'],
                            trg_mask,
                            decoder_state['enc_attn_cache'],
                            decoder_state['self_attn_cache'])
