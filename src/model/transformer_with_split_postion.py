import torch
import torch.nn as nn
from util.batch.batch_with_split_position import TrgTestBatch


class TransformerWithSplitPosition(nn.Module):

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
        # self.mid_word_mask = torch.zeros((len(self.vocab['trg'])), dtype=torch.long, requires_grad=False)
        # self.create_position_word_mask()

        if share_enc_dec_embedding:
            assert vocab['src'] == vocab['trg']
            self.src_embedding_layer.embedding_layer.weight = \
                self.trg_embedding_layer.embedding_layer.weight
        if share_decoder_embedding:
            self.generator.proj.weight = self.trg_embedding_layer.embedding_layer.weight

    def create_position_word_mask(self, device):
        mid_word_mask = torch.zeros((len(self.vocab['trg'])), dtype=torch.long, requires_grad=False).to(device)
        for i, word in enumerate(self.vocab['trg'].itos):
            if len(word) > 2 and word[-2:] == '@@':
                mid_word_mask[i] = 1
        self.register_buffer('mid_word_mask', mid_word_mask)

    def forward(self, src, src_mask,
                trg_input, trg, trg_mask,
                src_inner_index, src_outer_index,
                trg_inner_index, trg_outer_index):

        decoder_state = self.prepare_for_decode(src=src, src_mask=src_mask,
                                                src_inner_position=src_inner_index, src_outer_position=src_outer_index)
        decoder_logits, _, _ = self.decode(trg_input=trg_input, trg_mask=trg_mask,
                                           trg_inner_position=trg_inner_index, trg_outer_position=trg_outer_index,
                                           decoder_state=decoder_state)
        log_probs = self.generator(decoder_logits)
        return {'log_prob': log_probs}

    def prepare_for_decode(self, src, src_mask, src_inner_position, src_outer_position, test=None):

        assert src_mask is not None

        encoder_state = self.encode(src, src_mask, src_inner_position, src_outer_position)
        decoder_state = {
            'memory': encoder_state['memory'],
            'src_mask': src_mask,
            'input': None,
            'trg_inner_position': None,
            'trg_outer_position': None,
            'current_is_mid_word': None,
            'enc_attn_cache': None,
            'self_attn_cache': None,
        }
        return decoder_state

    def decode_step(self, input, decode_state):
        # input_embedding = self.trg_embedding_layer(input)

        if decode_state['input'] is None:
            # [batch size, 1]
            decode_state['input'] = input.unsqueeze(1)
            decode_state['trg_inner_position'] = torch.zeros_like(decode_state['input']).\
                type_as(decode_state['input']).to(input.device)
            decode_state['trg_outer_position'] = torch.zeros_like(decode_state['input']).\
                type_as(decode_state['input']).to(input.device)
            decode_state['current_is_mid_word'] = torch.gather(self.mid_word_mask, 0, input)

            # prepare for inner_position and outer_position
        else:
            # [batch size * beam size, seq len]
            decode_state['input'] = torch.cat((decode_state['input'], input.unsqueeze(1)), dim=-1)

            # inner_position_index
            last_inner_position_index = decode_state['trg_inner_position'][:, -1]
            last_is_mid_word = decode_state['current_is_mid_word']
            inner_position = torch.where(last_is_mid_word == 1,
                                         last_inner_position_index + 1,
                                         torch.zeros_like(last_inner_position_index).type_as(last_inner_position_index)).unsqueeze(1)

            # outer_position_index
            last_outer_position_index = decode_state['trg_outer_position'][:, -1]
            outer_position = torch.where(last_is_mid_word == 1,
                                         last_outer_position_index,
                                         last_outer_position_index + 1).unsqueeze(1)

            # update state
            decode_state['current_is_mid_word'] = torch.gather(self.mid_word_mask, 0, input)
            decode_state['trg_inner_position'] = torch.cat((decode_state['trg_inner_position'], inner_position), dim=-1)
            decode_state['trg_outer_position'] = torch.cat((decode_state['trg_outer_position'], outer_position), dim=-1)

        batch = TrgTestBatch(decode_state['input'],
                             pad=self.vocab['trg'].stoi['<pad>'],
                             trg_inner_index=decode_state['trg_inner_position'],
                             trg_outer_index=decode_state['trg_outer_position'])

        # [batch size, seq len]
        # trg_input = batch.trg_input[:, -1:]
        trg_input = batch.trg_input

        # [batch size, seq len, seq len]
        # todo: wait for improvement
        trg_mask = batch.trg_mask[:, -1:]
        # trg_mask = batch.trg_mask
        trg_inner_index = batch.trg_inner_index
        trg_outer_index = batch.trg_outer_index

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

        logits, new_enc_attn_cache, new_self_attn_cache = self.decode(trg_input=trg_input, trg_mask=trg_mask,
                                                                      trg_inner_position=trg_inner_index, trg_outer_position=trg_outer_index,
                                                                      decoder_state=decode_state, test=True)

        logits = logits[:, -1, :]
        log_prob = self.generator(logits)

        for i in range(len(new_enc_attn_cache)):
            decode_state['enc_attn_cache_' + str(i) + '_key'] = new_enc_attn_cache[i][0]
            decode_state['enc_attn_cache_' + str(i) + '_value'] = new_enc_attn_cache[i][1]
            decode_state['self_attn_cache_' + str(i) + '_key'] = new_self_attn_cache[i][0]
            decode_state['self_attn_cache_' + str(i) + '_value'] = new_self_attn_cache[i][1]

        decode_state.pop('enc_attn_cache', None)
        decode_state.pop('self_attn_cache', None)

        # decode_state['enc_attn_cache'] = new_enc_attn_cache
        # decode_state['self_attn_cache'] = new_self_attn_cache

        return log_prob, decode_state

    def encode(self, src, src_mask, src_inner_position, src_outer_position):
        input_embedding = self.src_embedding_layer(src,
                                                   inner_position_index=src_inner_position,
                                                   outer_position_index=src_outer_position)
        encoder_state = self.encoder(input_embedding, src_mask)
        return encoder_state

    def decode(self, trg_input, trg_mask, trg_inner_position, trg_outer_position, decoder_state, test=False):
        trg_input_embedding = self.trg_embedding_layer(trg_input,
                                                       inner_position_index=trg_inner_position,
                                                       outer_position_index=trg_outer_position)

        if test:
            trg_input_embedding = trg_input_embedding[:, -1:]

        return self.decoder(trg_input_embedding,
                            decoder_state['memory'],
                            decoder_state['src_mask'],
                            trg_mask,
                            decoder_state['enc_attn_cache'],
                            decoder_state['self_attn_cache'])
