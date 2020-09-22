import torch
import torch.nn as nn
from util.batch.transformer_batch import TrgTestBatch


class TransformerWithMixAdapter(nn.Module):

    def __init__(self,
                 src_embedding_layer,
                 trg_embedding_layer,
                 encoder,
                 decoder,
                 generator,
                 vocab,
                 share_decoder_embedding=False,
                 share_enc_dec_embedding=False,
                 ):

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

        self.mix_output = False
        self.used_domain_list = None
        self.target_domain = None
        # self.ref_domain_list = None

    def forward(self, src, src_mask, trg_input, trg, trg_mask, target_domain,
                mix_output: bool = False,
                used_domain_list: list = None, ):

        decoder_state = self.prepare_for_decode(src, src_mask, target_domain,
                                                mix_output=mix_output,
                                                used_domain_list=used_domain_list, )

        enc_adapter_output = decoder_state.pop('enc_adapter_output')
        enc_mix_gate = decoder_state.pop('enc_mix_gate')

        decode_result = self.decode(trg_input, trg_mask, decoder_state,
                                    target_domain=self.target_domain,
                                    mix_output=self.mix_output,
                                    used_domain_list=self.used_domain_list,
                                    )

        logit = self.generator(decode_result['logits'])
        log_probs = torch.log_softmax(logit, -1)

        return {'logit': logit,
                'log_prob': log_probs,
                'enc_adapter_output': enc_adapter_output,
                'enc_mix_gate': enc_mix_gate,
                'dec_adapter_output': decode_result['adapter_output'],
                'dec_mix_gate': decode_result['mix_gate']}

    def classify_forward(self, src, src_mask):

        input_embedding = self.src_embedding_layer(src)
        emb_classify_logits = self.emb_classifier(input_embedding, src_mask,
                                                  self.classify_domain_mask)  # [batch, class num]

        return {'emb_classify_logits': emb_classify_logits}

    def prepare_for_decode(self, src, src_mask, target_domain,
                           mix_output: bool = False,
                           used_domain_list: list = None, ):

        assert src_mask is not None

        self.target_domain = target_domain
        self.mix_output = mix_output
        self.used_domain_list = used_domain_list

        encoder_state = self.encode(src, src_mask, target_domain, mix_output, used_domain_list)
        decoder_state = {
            'memory': encoder_state['memory'],
            'enc_adapter_output': encoder_state['adapter_output'],
            'enc_mix_gate': encoder_state['mix_gate'],
            'src_mask': src_mask,
            # the variables below are for decoding
            'input': None,
            'enc_attn_cache': None,
            'self_attn_cache': None,
            'dec_adapter_output': None,
            'dec_mix_gate': None,
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

        # if the 'dec_mix_gate' and 'dec_mix_gate_0' ara both not in the state, then we do not return them, but should
        # set the default value to None
        if 'dec_mix_gate' not in decode_state.keys() and 'dec_mix_gate_0' not in decode_state.keys():
            decode_state['dec_mix_gate'] = None
        # if 'dec_mix_gate_0' then we need to cat them every decoding step
        elif 'dec_mix_gate' not in decode_state.keys() and 'dec_mix_gate_0' in decode_state.keys():
            dec_mix_gate_list = []
            for i in range(self.decoder.num_layers):
                dec_mix_gate_list.append(decode_state.get('dec_mix_gate_{}'.format(i), None))
            decode_state['dec_mix_gate'] = dec_mix_gate_list
        # for the first step
        else:
            assert decode_state.get('dec_mix_gate') is None

        if 'dec_adapter_output' not in decode_state.keys() and 'dec_adapter_output_0' not in decode_state.keys():
            decode_state['dec_adapter_output'] = None
        elif 'dec_adapter_output' not in decode_state.keys() and 'dec_adapter_output_0' in decode_state.keys():
            dec_adapter_output = []
            for i in range(self.decoder.num_layers):
                dec_adapter_output.append(decode_state.get('dec_adapter_output_{}'.format(i), None))
            decode_state['dec_adapter_output'] = dec_adapter_output
        else:
            assert decode_state.get('dec_adapter_output') is None

        decode_result = self.decode(trg_input,
                                    trg_mask,
                                    decode_state,
                                    target_domain=self.target_domain,
                                    mix_output=self.mix_output,
                                    used_domain_list=self.used_domain_list,
                                    test=True)

        logits = decode_result['logits']
        adapter_output = decode_result['adapter_output']  # list or None
        gate = decode_result['mix_gate']  # list or None
        new_enc_attn_cache = decode_result['enc_attn_cache_list']
        new_self_attn_cache = decode_result['new_self_attn_cache_list']

        logits = logits[:, -1, :]
        log_prob = self.generator(logits)

        for i in range(len(new_enc_attn_cache)):
            decode_state['enc_attn_cache_' + str(i) + '_key'] = new_enc_attn_cache[i][0]
            decode_state['enc_attn_cache_' + str(i) + '_value'] = new_enc_attn_cache[i][1]
            decode_state['self_attn_cache_' + str(i) + '_key'] = new_self_attn_cache[i][0]
            decode_state['self_attn_cache_' + str(i) + '_value'] = new_self_attn_cache[i][1]

            if adapter_output is not None:
                decode_state['dec_adapter_output_{}'.format(i)] = adapter_output[i]
            if gate is not None:
                decode_state['dec_mix_gate_{}'.format(i)] = gate[i]

        decode_state.pop('enc_attn_cache', None)
        decode_state.pop('self_attn_cache', None)
        decode_state.pop('dec_adapter_output', None)
        decode_state.pop('dec_mix_gate', None)

        return log_prob, decode_state

    def encode(self, src,
               src_mask,
               target_domain,
               mix_output: bool = False,
               used_domain_list: list = None,):

        input_embedding = self.src_embedding_layer(src)
        encoder_state = self.encoder(input_embedding, src_mask, target_domain=target_domain,
                                     mix_output=mix_output,
                                     used_domain_list=used_domain_list,)
        return encoder_state

    def decode(self, trg_input, trg_mask, decoder_state,
               target_domain,
               mix_output: bool = False,
               used_domain_list: list = None,
               test=False,
               ):

        trg_input_embedding = self.trg_embedding_layer(trg_input)

        if test:
            trg_input_embedding = trg_input_embedding[:, -1:]

        return self.decoder(trg_input_embedding,
                            decoder_state['memory'],
                            decoder_state['src_mask'],
                            trg_mask,
                            decoder_state['enc_attn_cache'],
                            decoder_state['self_attn_cache'],
                            decoder_state['dec_adapter_output'],
                            decoder_state['dec_mix_gate'],
                            target_domain=target_domain,
                            mix_output=mix_output,
                            used_domain_list=used_domain_list,
                            )
