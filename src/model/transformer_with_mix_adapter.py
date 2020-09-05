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
                 emb_classifier,
                 vocab,
                 classify_domain_mask,
                 share_decoder_embedding=False,
                 share_enc_dec_embedding=False,
                 domain_list: list = None,
                 max_domain_num: int = 0,
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

        self.classify_domain_mask = torch.ByteTensor(classify_domain_mask)
        self.domain_list = domain_list
        self.max_domain_num = max_domain_num
        self.emb_classifier = emb_classifier

        self.mix_output = False
        self.used_domain_list = None
        self.mix_weight = None
        self.domain_mask = None

        self.target_domain = None
        # self.ref_domain_list = None

    def forward(self, src, src_mask, trg_input, trg, trg_mask, target_domain,
                mix_output: bool = False,
                used_domain_list: list = None,
                mix_weight: torch.Tensor = None,
                domain_mask: torch.Tensor = None):

        decoder_state = self.prepare_for_decode(src, src_mask, target_domain,
                                                mix_output=mix_output,
                                                used_domain_list=used_domain_list,
                                                mix_weight=mix_weight,
                                                domain_mask=domain_mask,
                                                require_adapter_output=True)

        if mix_output:
            decoder_logits, adapter_output, _, _, decode_mix_weight = self.decode(trg_input, trg_mask, decoder_state,
                                                                                  target_domain=self.target_domain,
                                                                                  mix_output=self.mix_output,
                                                                                  used_domain_list=self.used_domain_list,
                                                                                  mix_weight=self.mix_weight,
                                                                                  domain_mask=self.domain_mask,
                                                                                  )
        else:
            decoder_logits, adapter_output, _, _ = self.decode(trg_input, trg_mask, decoder_state,
                                                               target_domain=self.target_domain,
                                                               mix_output=self.mix_output,
                                                               used_domain_list=self.used_domain_list,
                                                               mix_weight=self.mix_weight,
                                                               domain_mask=self.domain_mask,
                                                               )

        logit = self.generator(decoder_logits, return_logit=True)
        log_probs = torch.log_softmax(logit, -1)

        return {'logit': logit,
                'log_prob': log_probs,
                'enc_adapter_output': decoder_state['encoder_adapter_output'],
                'dec_adapter_output': adapter_output,
                'enc_mix_weight': decoder_state['enc_calculate_mix_weight'] if mix_output else None,
                'dec_mix_weght': decode_mix_weight if mix_output else None}

    def classify_forward(self, src, src_mask):

        input_embedding = self.src_embedding_layer(src)
        emb_classify_logits = self.emb_classifier(input_embedding, src_mask,
                                                  self.classify_domain_mask)  # [batch, class num]

        return {'emb_classify_logits': emb_classify_logits}

    def prepare_for_decode(self, src, src_mask, target_domain,
                           mix_output: bool = False,
                           used_domain_list: list = None,
                           mix_weight: torch.Tensor = None,
                           domain_mask: torch.Tensor = None,
                           require_adapter_output=False):

        assert src_mask is not None

        self.target_domain = target_domain
        self.mix_output = mix_output
        self.used_domain_list = used_domain_list
        self.mix_weight = mix_weight
        self.domain_mask = domain_mask

        # print('model ', target_domain)
        encoder_state = self.encode(src, src_mask, target_domain, mix_output, used_domain_list, mix_weight, domain_mask)
        decoder_state = {
            'memory': encoder_state['memory'],
            'src_mask': src_mask,
            'input': None,
            'enc_attn_cache': None,
            'self_attn_cache': None,
        }

        if not isinstance(target_domain, str) and not isinstance(target_domain, list):
            decoder_state['target_domain'] = target_domain

        if require_adapter_output:
            decoder_state['encoder_adapter_output'] = encoder_state['adapter_output']

        if mix_output is True:
            decoder_state['enc_calculate_mix_weight'] = encoder_state['encoder_calculate_mix_weights']

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

        if self.mix_output:
            logits, adapter_output, new_enc_attn_cache, new_self_attn_cache, _ = self.decode(trg_input,
                                                                                             trg_mask,
                                                                                             decode_state,
                                                                                             target_domain=decode_state[
                                                                                                 'target_domain'] \
                                                                                                 if 'target_domain' in decode_state.keys() else self.target_domain,
                                                                                             mix_output=self.mix_output,
                                                                                             used_domain_list=self.used_domain_list,
                                                                                             mix_weight=self.mix_weight,
                                                                                             domain_mask=self.domain_mask,
                                                                                             test=True)

        else:
            logits, adapter_output, new_enc_attn_cache, new_self_attn_cache = self.decode(trg_input,
                                                                                          trg_mask,
                                                                                          decode_state,
                                                                                          target_domain=decode_state[
                                                                                              'target_domain'] \
                                                                                              if 'target_domain' in decode_state.keys() else self.target_domain,
                                                                                          test=True)

        logits = logits[:, -1, :]
        log_prob = self.generator(logits)

        for i in range(len(new_enc_attn_cache)):
            decode_state['enc_attn_cache_' + str(i) + '_key'] = new_enc_attn_cache[i][0]
            decode_state['enc_attn_cache_' + str(i) + '_value'] = new_enc_attn_cache[i][1]
            decode_state['self_attn_cache_' + str(i) + '_key'] = new_self_attn_cache[i][0]
            decode_state['self_attn_cache_' + str(i) + '_value'] = new_self_attn_cache[i][1]

        decode_state.pop('enc_attn_cache', None)
        decode_state.pop('self_attn_cache', None)
        decode_state.pop('enc_calculate_mix_weight', None)

        return log_prob, decode_state

    def encode(self, src,
               src_mask,
               target_domain,
               mix_output: bool = False,
               used_domain_list: list = None,
               mix_weight: torch.Tensor = None,
               domain_mask: torch.Tensor = None):

        input_embedding = self.src_embedding_layer(src)
        encoder_state = self.encoder(input_embedding, src_mask, target_domain=target_domain,
                                     mix_output=mix_output,
                                     used_domain_list=used_domain_list,
                                     mix_weight=mix_weight,
                                     domain_mask=domain_mask)
        return encoder_state

    def decode(self, trg_input, trg_mask, decoder_state,
               target_domain,
               mix_output: bool = False,
               used_domain_list: list = None,
               mix_weight: torch.Tensor = None,
               domain_mask: torch.Tensor = None,
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
                            target_domain=target_domain,
                            mix_output=mix_output,
                            used_domain_list=used_domain_list,
                            mix_weight=mix_weight,
                            domain_mask=domain_mask
                            )
