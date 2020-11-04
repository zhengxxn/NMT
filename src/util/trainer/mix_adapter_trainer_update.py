from util.trainer.transformer_trainer import Trainer
from util.decoding.mix_adapter_decoding import beam_search
from tqdm import tqdm
import torch
import torch.nn as nn
import math
import numpy as np


def update_loss_dict(long_term_loss_dict, current_loss_dict, long_term_tag):
    # update the epoch or step loss dict with the current step loss dict
    for description in current_loss_dict:

        long_term_description = '{}_{}'.format(long_term_tag, description)

        if long_term_description not in long_term_loss_dict:
            long_term_loss_dict[long_term_description] = current_loss_dict[description]
        else:
            long_term_loss_dict[long_term_description] += current_loss_dict[description]


def set_dict_zero(dic):
    # for initialize the step loss dict
    for k in dic:
        dic[k] = 0


class Mix_Adapter_Trainer(Trainer):

    def __init__(self,
                 model,
                 criterion,
                 validation_criterion,
                 vocab,
                 optimizer,
                 lr_scheduler,
                 train_iterators,
                 validation_iterators,
                 validation_test_iterators,
                 optimizer_config,
                 train_config,
                 validation_config,
                 record_config,
                 device,
                 ):

        super().__init__(
            model,
            criterion,
            validation_criterion,
            vocab,
            optimizer,
            lr_scheduler,
            train_iterators,
            validation_iterators,
            validation_test_iterators,
            optimizer_config,
            train_config,
            validation_config,
            record_config,
            device, )

        self.target_domain = train_config['target_domain']
        # self.distillation_criterion = distillation_criterion

        self.used_domain_list = train_config['used_domain_list']
        # self.used_inner_gate = train_config['used_inner_gate']

        self.train_stage = train_config['stage']
        self.mix_output = False
        self.go_through_shared_adapter = False

        if self.train_stage == 'train_single_adapter':
            self.mix_output = False
        elif self.train_stage == 'train_gate':
            self.mix_output = True
        elif self.train_stage == 'mix_adapter':
            self.mix_output = True

        elif self.train_stage == 'knowledge_distillation':
            self.ref_domain_dict = train_config['ref_domain_dict']
            self.kd_loss_factor = train_config['kd_loss_factor']
            self.kl_criterion = torch.nn.KLDivLoss(reduction='none')

        elif self.train_stage == 'adaptive_knowledge_distillation':
            self.adaptive_level = train_config['adaptive_level']
            self.ref_domain_dict = train_config['ref_domain_dict']
            self.kd_criterion = torch.nn.KLDivLoss(reduction='none')
            self.calculate_kd_factor_criterion = torch.nn.NLLLoss(reduction='none',
                                                                  ignore_index=self.vocab['trg'].stoi['<pad>'])

        elif self.train_stage == 'hidden_knowledge_distillation':
            self.ref_domain_dict = train_config['ref_domain_dict']
            self.kd_loss_factor = train_config['kd_loss_factor']
            self.kl_criterion = torch.nn.KLDivLoss(reduction='none')
            self.hidden_kl_criterion = torch.nn.MSELoss(reduction='sum')

        elif self.train_stage == 'sent_kd':
            self.loss_factor = train_config['loss_factor']

        elif self.train_stage == 'train_shared_adapter':

            self.go_through_shared_adapter = True
            self.train_dataset_domain = train_config['dataset_domain']
            self.classify_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
            # self.differ_criterion = nn.MSELoss()

        elif self.train_stage == 'train_mix_layer_for_adapter':

            self.mix_output = True
            self.classify_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
            self.train_dataset_domain = train_config['dataset_domain']
            self.domain_label = train_config['domain_label']

            self.enc_mix_loss_type = train_config.get('enc_mix_loss_type', None)
            self.dec_mix_loss_type = train_config.get('dec_mix_loss_type', None)
            self.enc_mix_loss_factor = train_config.get('enc_mix_loss_factor', None)
            self.dec_mix_loss_factor = train_config.get('dec_mix_loss_factor', None)

        else:
            assert 0 == 1

        self.step_loss_dict = {'sum_loss': 0, 'sum_tokens': 0}

    def train_step(self, model, batches: list):
        """
        a step includes a forward and a backward, and we rebatch it with some batch options
        :param batches: [domain1 batch, domain2 batch ...]
        :param model:
        :return: sum loss, sum tokens
        """

        if self.train_stage == 'train_shared_adapter':
            # list[[b, s, h], [b, s, h], ...]
            # self.train_shared_adapter_step(new_batch, target_adapter_result, i)
            pass

        if self.train_stage == 'knowledge_distillation':
            return self.train_kd_step(model, batches)

        if self.train_stage == 'sent_kd':
            return self.train_sent_kd_step(model, batches)

        if self.train_stage == 'train_mix_layer_for_adapter':
            return self.train_mix_layer_step(model, batches)

        if self.train_stage == 'hidden_knowledge_distillation':
            return self.train_hidden_kd_step(model, batches)

        sum_tokens = 0
        sum_loss = 0

        for i, batch in enumerate(batches):
            new_batch = self.rebatch(batch)
            target_adapter_result = model.forward(new_batch.src, new_batch.src_mask,
                                                  new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                                  self.target_domain,
                                                  mix_output=self.mix_output,
                                                  used_domain_list=self.used_domain_list,
                                                  # go_through_shared_adapter=self.go_through_shared_adapter
                                                  )
            log_prob = target_adapter_result['log_prob']

            loss = self.criterion(
                log_prob.contiguous().view(-1, log_prob.size(-1)),
                new_batch.trg.contiguous().view(-1)
            )

            sum_loss += loss.item()
            sum_tokens += new_batch.ntokens.item()

            if self.criterion.reduction == 'mean':
                loss = loss / new_batch.ntokens.item()

            loss.backward()

        return {'sum_loss': sum_loss, 'sum_tokens': sum_tokens}

    def validation_step(self, batch):
        new_batch = self.rebatch(batch)
        log_prob = self.model.forward(new_batch.src, new_batch.src_mask,
                                      new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                      self.target_domain,
                                      mix_output=self.mix_output,
                                      used_domain_list=self.used_domain_list,
                                      )['log_prob']
        loss = self.validation_criterion(
            log_prob.contiguous().view(-1, log_prob.size(-1)),
            new_batch.trg.contiguous().view(-1)
        )
        return loss.item(), new_batch.ntokens.item()

    def validation_decoding_step(self, batch):
        search_results = beam_search(
            model=self.model,
            src=batch.src,
            sos_index=self.vocab['trg'].stoi['<sos>'],
            eos_index=self.vocab['trg'].stoi['<eos>'],
            pad_index=self.vocab['trg'].stoi['<pad>'],
            target_domain=self.target_domain,
            max_len=self.decoding_max_steps,
            beam_size=self.decoding_beam_size,
            length_penalty=self.decoding_length_penalty,
            alpha=self.decoding_alpha,
            mix_output=self.mix_output,
            used_domain_list=self.used_domain_list,
            use_multiple_gpu=self.use_multiple_gpu,
        )

        return search_results

    def run_epoch(self, model, data_iterators):

        self.model.train()
        for data_iterator in data_iterators:
            data_iterator.init_epoch()

        cur_step_in_epoch = 0
        epoch_loss_dict = {'epoch_sum_loss': 0, 'epoch_sum_tokens': 0}

        with tqdm(zip(*tuple(data_iterators))) as bar:
            bar.set_description("training epoch: " + str(self.current_epoch))

            for batches in bar:

                # end epoch
                if cur_step_in_epoch != 0 and data_iterators[0].state_dict()['iterations_this_epoch'] == 1:
                    break

                if self.batch_count == 0:
                    self.optimizer.zero_grad()

                loss_dict = self.train_step(self.model, batches)  # for each token

                self.batch_count = (self.batch_count + 1) % self.update_batch_count
                if self.batch_count == 0:
                    self.current_step += 1
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    if self.lr_scheduler['base_on'] == 'step':
                        self.lr_scheduler['scheduler'].step()
                    cur_step_in_epoch += 1

                update_loss_dict(epoch_loss_dict, loss_dict, 'epoch')
                update_loss_dict(self.step_loss_dict, loss_dict, 'step')

                bar.set_postfix({'loss': '{0:1.5f}'.format(loss_dict['sum_loss'] / loss_dict['sum_tokens']),
                                 'average_train_loss': '{0:1.5f}'.format(
                                     epoch_loss_dict['epoch_sum_loss'] / epoch_loss_dict['epoch_sum_tokens']),
                                 'best_validation_loss': '{0:1.5f}'.format(self.best_validation_loss)})
                bar.update()

                # validation:
                if self.current_step != 0 and self.batch_count == 0:
                    if self.current_step >= self.start_loss_validation_on_steps \
                            and self.current_step % self.loss_validation_frequency == 0:
                        self.loss_validation()
                        self.write_train_loss(self.step_loss_dict, 'step', self.current_step)  # update
                        set_dict_zero(self.step_loss_dict)
                        self.writer.add_scalar(tag='lr_rate',
                                               scalar_value=[group['lr'] for group in self.optimizer.param_groups][0],
                                               global_step=self.current_step)
                        self.model.train()

                    if self.current_step >= self.start_bleu_validation_on_steps \
                            and self.current_step % self.bleu_validation_frequency == 0:
                        self.bleu_validation()
                        self.model.train()

                    if self.current_step >= self.save_checkpoint_start_on_steps \
                            and self.current_step % self.save_checkpoint_frequency == 0 and self.checkpoint_num > 0:
                        index = self.current_checkpoint_index
                        self.save_checkpoint(self.checkpoint_model_path[index],
                                             self.checkpoint_optimizer_path[index],
                                             self.checkpoint_lr_scheduler_path[index],
                                             self.checkpoint_save_optimizer,
                                             self.checkpoint_save_lr_scheduler)
                        self.current_checkpoint_index = (self.current_checkpoint_index + 1) % self.checkpoint_num

        return epoch_loss_dict

    def train(self):
        """
        :return:
        """
        self.model.train()
        for self.current_epoch in range(self.current_epoch, self.epoch_num):
            epoch_loss_dict = self.run_epoch(self.model, self.train_iterators)
            self.write_train_loss(epoch_loss_dict, loss_tag='epoch', record_step=self.current_epoch)

    def train_shared_adapter_step(self, batch, target_adapter_result, target_label):

        enc_adapter_output_wo_res_connect = target_adapter_result['enc_adapter_output_wo_res_connect']
        dec_adapter_output_wo_res_connect = target_adapter_result['dec_adapter_output_wo_res_connect']
        enc_adapter_output_wo_res_connect = torch.stack(enc_adapter_output_wo_res_connect,
                                                        dim=0)  # [l, b, s, h]
        dec_adapter_output_wo_res_connect = torch.stack(dec_adapter_output_wo_res_connect,
                                                        dim=0)  # [l, b, s, h]
        enc_adapter_output_norm = torch.norm(enc_adapter_output_wo_res_connect, dim=-1) / math.sqrt(512)
        dec_adapter_output_norm = torch.norm(dec_adapter_output_wo_res_connect, dim=-1) / math.sqrt(512)

        enc_classify_logits = target_adapter_result['enc_classify_logits']
        dec_classify_logits = target_adapter_result['dec_classify_logits']
        enc_classify_logits = torch.stack(enc_classify_logits, dim=0)  # [l, b, s, 2]
        dec_classify_logits = torch.stack(dec_classify_logits, dim=0)  # [l, b, s, 2]
        enc_classify_entropy = torch.distributions.categorical.Categorical \
            (torch.softmax(enc_classify_logits, -1)).entropy()  # [l, b, s]
        dec_classify_entropy = torch.distributions.categorical.Categorical \
            (torch.softmax(dec_classify_logits, -1)).entropy()  # [l, b, s]
        enc_classify_entropy = enc_classify_entropy.masked_fill(batch.src_mask == 0, value=0)
        dec_classify_entropy = dec_classify_entropy.masked_fill(batch.trg_input_mask == 0, value=0)

        num_layers = enc_classify_logits.size(0)
        enc_classify_logits = enc_classify_logits.view(-1, enc_classify_logits.size(-1))
        dec_classify_logits = dec_classify_logits.view(-1, dec_classify_logits.size(-1))

        enc_classify_target = torch.zeros_like(batch.src).unsqueeze(0).fill_(target_label)  # [1, b, s]
        dec_classify_target = torch.zeros_like(batch.trg_input).unsqueeze(0).fill_(target_label)  # [1, b, s]
        enc_classify_target = enc_classify_target.masked_fill(batch.src_mask == 0, value=-1)
        dec_classify_target = dec_classify_target.masked_fill(batch.trg_input_mask == 0, value=-1)
        enc_classify_target = enc_classify_target.repeat(num_layers, 1, 1)
        dec_classify_target = dec_classify_target.repeat(num_layers, 1, 1)

        enc_classify_loss = self.classify_criterion(enc_classify_logits, enc_classify_target)
        dec_classify_loss = self.classify_criterion(dec_classify_logits, dec_classify_target)

    def train_kd_step(self, model, batches: list):

        loss_dict = {'sum_loss': 0, 'sum_tokens': 0}

        for batch in batches:
            new_batch = self.rebatch(batch)
            ref_logit = {}

            # for ref domain
            model.eval()
            with torch.no_grad():
                for ref_domain in self.ref_domain_dict.keys():
                    ref_result = model.forward(new_batch.src, new_batch.src_mask,
                                               new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                               ref_domain)
                    ref_logit[ref_domain] = ref_result['logit']

            # for target domain
            model.train()
            target_adapter_result = model.forward(new_batch.src, new_batch.src_mask,
                                                  new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                                  self.target_domain)

            log_prob = target_adapter_result['log_prob']
            target_domain_logit = target_adapter_result['logit']

            # translation_loss
            translation_loss = self.criterion(
                log_prob.contiguous().view(-1, log_prob.size(-1)),
                new_batch.trg.contiguous().view(-1)
            )

            # generate trg mask
            trg_mask = new_batch.trg.ne(self.vocab['trg'].stoi['<pad>'])

            # calculate_kd_loss
            kd_loss = 0
            for ref_domain in self.ref_domain_dict.keys():
                ref_domain_logit = ref_logit[ref_domain]
                temperature = self.ref_domain_dict[ref_domain]['temperature']
                factor = self.ref_domain_dict[ref_domain]['factor']

                unreduced_domain_kd_loss = self.kl_criterion(
                    (target_domain_logit / temperature).log_softmax(-1),
                    (ref_domain_logit / temperature).softmax(-1)).sum(-1)

                unreduced_domain_kd_loss = unreduced_domain_kd_loss.masked_fill(trg_mask == 0, value=0)
                current_domain_kd_loss = factor * (temperature ** 2) * unreduced_domain_kd_loss.sum()

                loss_dict['{}_kd_loss'.format(ref_domain)] = current_domain_kd_loss.item()
                kd_loss = kd_loss + current_domain_kd_loss

            loss = (1 - self.kd_loss_factor) * translation_loss + self.kd_loss_factor * kd_loss

            if self.criterion.reduction == 'mean':
                loss = loss / new_batch.ntokens.item()

            loss.backward()

            loss_dict['kd_loss'] = kd_loss.item()
            loss_dict['translation_loss'] = translation_loss.item()
            loss_dict['sum_loss'] += loss.item()
            loss_dict['sum_tokens'] += new_batch.ntokens.item()

        # {'sum_loss': , 'sum_tokens': , 'kd_loss': , 'translation_loss': , 'domain_kd_loss': ...}
        return loss_dict

    def train_hidden_kd_step(self, model, batches):

        loss_dict = {'sum_loss': 0, 'sum_tokens': 0}

        for batch in batches:
            new_batch = self.rebatch(batch)
            ref_logit = {}
            ref_enc_hidden = {}
            ref_dec_hidden = {}

            # for ref domain
            model.eval()
            with torch.no_grad():
                for ref_domain in self.ref_domain_dict.keys():
                    ref_result = model.forward(new_batch.src, new_batch.src_mask,
                                               new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                               ref_domain)
                    ref_logit[ref_domain] = ref_result['logit']
                    ref_enc_hidden[ref_domain] = torch.cat(ref_result['enc_adapter_output'], dim=0)  # [L, B, S, H]
                    ref_dec_hidden[ref_domain] = torch.cat(ref_result['dec_adapter_output'], dim=0)  # [L, B, S, H]

            # for target domain
            model.train()
            target_adapter_result = model.forward(new_batch.src, new_batch.src_mask,
                                                  new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                                  self.target_domain)

            log_prob = target_adapter_result['log_prob']
            target_domain_logit = target_adapter_result['logit']
            target_domain_enc_hidden = torch.cat(target_adapter_result['enc_adapter_output'], dim=0)
            target_domain_dec_hidden = torch.cat(target_adapter_result['dec_adapter_output'], dim=0)

            # translation_loss
            translation_loss = self.criterion(
                log_prob.contiguous().view(-1, log_prob.size(-1)),
                new_batch.trg.contiguous().view(-1)
            )
            # do average
            translation_loss = translation_loss / new_batch.ntokens.item()

            # generate trg mask
            trg_mask = new_batch.trg.ne(self.vocab['trg'].stoi['<pad>'])

            # calculate_kd_loss
            kd_loss = 0
            for ref_domain in self.ref_domain_dict.keys():
                ref_domain_logit = ref_logit[ref_domain]
                temperature = self.ref_domain_dict[ref_domain]['temperature']
                logit_factor = self.ref_domain_dict[ref_domain]['logit_factor']
                enc_hidden_factor = self.ref_domain_dict[ref_domain]['enc_hidden_factor']
                dec_hidden_factor = self.ref_domain_dict[ref_domain]['enc_hidden_factor']

                unreduced_domain_kd_loss = self.kl_criterion(
                    (target_domain_logit / temperature).log_softmax(-1),
                    (ref_domain_logit / temperature).softmax(-1)).sum(-1)
                unreduced_domain_kd_loss = unreduced_domain_kd_loss.masked_fill(trg_mask == 0, value=0)
                reduced_domain_kd_loss = unreduced_domain_kd_loss.sum() / new_batch.ntokens.item()

                # enc hidden kd
                target_domain_enc_hidden = target_domain_enc_hidden / target_domain_enc_hidden.norm(dim=-1).unsqueeze(
                    -1)
                enc_hidden_counts = np.prod(list(target_domain_enc_hidden.size())[:-1])
                ref_enc_hidden[ref_domain] = ref_enc_hidden[ref_domain] / ref_enc_hidden[ref_domain].norm(
                    dim=-1).unsqueeze(-1)
                reduced_domain_enc_hidden_kd_loss = self.hidden_kl_criterion(
                    target_domain_enc_hidden, ref_enc_hidden[ref_domain]
                ) / enc_hidden_counts

                # dec hidden kd
                target_domain_dec_hidden = target_domain_dec_hidden / target_domain_dec_hidden.norm(dim=-1).unsqueeze(
                    -1)
                dec_hidden_counts = np.prod(list(target_domain_dec_hidden.size())[:-1])
                ref_dec_hidden[ref_domain] = ref_dec_hidden[ref_domain] / ref_dec_hidden[ref_domain].norm(
                    dim=-1).unsqueeze(-1)
                reduced_domain_dec_hidden_kd_loss = self.hidden_kl_criterion(
                    target_domain_dec_hidden, ref_dec_hidden[ref_domain]
                ) / dec_hidden_counts

                # kd loss sum
                current_domain_kd_loss = logit_factor * (temperature ** 2) * reduced_domain_kd_loss + \
                                         enc_hidden_factor * reduced_domain_enc_hidden_kd_loss + dec_hidden_factor * reduced_domain_dec_hidden_kd_loss

                loss_dict['logit_kd_loss'] = reduced_domain_kd_loss.item() * new_batch.ntokens.item()
                loss_dict['enc_hidden_loss'] = reduced_domain_enc_hidden_kd_loss.item() * enc_hidden_counts
                loss_dict['dec_hidden_loss'] = reduced_domain_dec_hidden_kd_loss.item() * dec_hidden_counts
                loss_dict['enc_hidden_counts'] = enc_hidden_counts
                loss_dict['dec_hidden_counts'] = dec_hidden_counts
                kd_loss = kd_loss + current_domain_kd_loss

            loss = (1 - self.kd_loss_factor) * translation_loss + self.kd_loss_factor * kd_loss

            # if self.criterion.reduction == 'mean':
            #     loss = loss / new_batch.ntokens.item()
            loss.backward()

            loss_dict['kd_loss'] = kd_loss.item() * new_batch.ntokens.item()
            loss_dict['translation_loss'] = translation_loss.item() * new_batch.ntokens.item()
            loss_dict['sum_loss'] += loss.item() * new_batch.ntokens.item()
            loss_dict['sum_tokens'] += new_batch.ntokens.item()

        # {'sum_loss': , 'sum_tokens': , 'kd_loss': , 'translation_loss': , 'domain_kd_loss': ...}
        return loss_dict

    def train_sent_kd_step(self, model, batches: list):
        """
        suppose the first batch is real data,
        and the batches after are synthetic data generated from different models(domains)

        :param model:
        :param batches:
        :return:
        """

        loss_dict = {'sum_loss': 0, 'sum_tokens': 0}

        for i, batch in enumerate(batches):
            new_batch = self.rebatch(batch)

            result = model.forward(new_batch.src, new_batch.src_mask,
                                   new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                   self.target_domain)

            log_prob = result['log_prob']
            # translation_loss

            loss = self.criterion(
                log_prob.contiguous().view(-1, log_prob.size(-1)),
                new_batch.trg.contiguous().view(-1)
            )

            factor = self.loss_factor[i]
            weight_loss = factor * loss

            if self.criterion.reduction == 'mean':
                weight_loss = weight_loss / new_batch.ntokens.item()

            weight_loss.backward()

            loss_dict['dataset_{}_loss'.format(i)] = loss.item()
            loss_dict['dataset_{}_tokens'.format(i)] = new_batch.ntokens.item()
            loss_dict['sum_loss'] += weight_loss.item()
            loss_dict['sum_tokens'] += new_batch.ntokens.item()

        # {'sum_loss': , 'sum_tokens': , 'dataset_0_loss': , 'dataset_1_loss': , 'dataset_2_loss': ...}
        return loss_dict

    def adaptive_mix_kd_step(self, model, batches):

        loss_dict = {'sum_loss': 0, 'sum_tokens': 0}

        for batch in batches:
            new_batch = self.rebatch(batch)
            ref_logit = {}
            ref_neg_ppl = {}
            ref_factor = {}

            # for ref domain
            model.eval()
            with torch.no_grad():
                for ref_domain in self.ref_domain_dict.keys():
                    ref_result = model.forward(new_batch.src, new_batch.src_mask,
                                               new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                               ref_domain)
                    ref_logit[ref_domain] = ref_result['logit']  # [B, S, V]

                    # batch-wise adaptive kd
                    current_ref_log_prob = ref_result['log_prob']
                    current_ref_loss = self.calculate_kd_factor_criterion(
                        current_ref_log_prob.contiguous(),
                        new_batch.trg.contiguous()
                    )  # [B, S]

                    if self.adaptive_level == 'batch':
                        # batch-wise adaptive kd
                        current_ref_loss = current_ref_loss.sum() / new_batch.ntokens
                        ref_neg_ppl[ref_domain] = - torch.exp(current_ref_loss)  # [1]

                    elif self.adaptive_level == 'sent':
                        # sent-wise adaptive kd
                        sent_len = new_batch.ne(self.vocab['trg'].stoi['<pad>']).sum(dim=-1)  # [B]
                        current_ref_loss = current_ref_loss.sum(dim=-1) / sent_len
                        ref_neg_ppl[ref_domain] = - torch.exp(current_ref_loss)  # [B]

                    elif self.adaptive_level == 'token':
                        # token-wise adaptive kd
                        ref_neg_ppl[ref_domain] = - torch.exp(current_ref_loss)  # [B, S]

            # calculate factor for each domain logit
            neg_ppl_list = []
            for ref_domain in self.ref_domain_dict:
                neg_ppl_list.append(ref_neg_ppl[ref_domain])
            neg_ppl_factor = torch.stack(neg_ppl_list, dim=-1)  # [..., D]
            neg_ppl_factor = torch.softmax(neg_ppl_factor, dim=-1)  # [..., D]
            for idx, ref_domain in enumerate(self.ref_domain_dict):
                ref_factor[ref_domain] = neg_ppl_factor.select(dim=-1, index=idx)  # [...]

            # for target domain
            model.train()
            target_adapter_result = model.forward(new_batch.src, new_batch.src_mask,
                                                  new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                                  self.target_domain)

            log_prob = target_adapter_result['log_prob']
            target_domain_logit = target_adapter_result['logit']

            # translation_loss
            translation_loss = self.criterion(
                log_prob.contiguous().view(-1, log_prob.size(-1)),
                new_batch.trg.contiguous().view(-1)
            )

            # generate trg mask
            trg_mask = new_batch.trg.ne(self.vocab['trg'].stoi['<pad>'])

            # calculate_kd_loss
            kd_loss = 0
            for ref_domain in self.ref_domain_dict.keys():
                ref_domain_logit = ref_logit[ref_domain]
                temperature = self.ref_domain_dict[ref_domain]['temperature']

                unreduced_domain_kd_loss = self.kl_criterion(
                    (target_domain_logit / temperature).log_softmax(-1),
                    (ref_domain_logit / temperature).softmax(-1)).sum(-1)  # [B, S]

                unreduced_domain_kd_loss = unreduced_domain_kd_loss.masked_fill(trg_mask == 0, value=0)
                if self.adaptive_level == 'batch':
                    current_domain_kd_loss = ref_factor[ref_domain] * (temperature ** 2) * unreduced_domain_kd_loss.sum()
                elif self.adaptive_level == 'sent':
                    current_domain_kd_loss = (ref_factor[ref_domain] * (temperature ** 2) * unreduced_domain_kd_loss.sum(dim=-1)).sum()
                elif self.adaptive_level == 'token':
                    current_domain_kd_loss = (ref_factor[ref_domain] * (temperature ** 2) * unreduced_domain_kd_loss).sum()

                loss_dict['{}_kd_loss'.format(ref_domain)] = current_domain_kd_loss.item()
                kd_loss = kd_loss + current_domain_kd_loss

            loss = (1 - self.kd_loss_factor) * translation_loss + self.kd_loss_factor * kd_loss

            if self.criterion.reduction == 'mean':
                loss = loss / new_batch.ntokens.item()

            loss.backward()

            loss_dict['kd_loss'] = kd_loss.item()
            loss_dict['translation_loss'] = translation_loss.item()
            loss_dict['sum_loss'] += loss.item()
            loss_dict['sum_tokens'] += new_batch.ntokens.item()

        # {'sum_loss': , 'sum_tokens': , 'kd_loss': , 'translation_loss': , 'domain_kd_loss': ...}
        return loss_dict

    def train_mix_layer_step(self, model, batches: list):

        loss_dict = {'sum_loss': 0, 'sum_tokens': 0, 'enc_sum_counts': 0, 'dec_sum_counts': 0}

        for i, batch in enumerate(batches):

            new_batch = self.rebatch(batch)
            result = model.forward(new_batch.src, new_batch.src_mask,
                                   new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                   self.target_domain, self.mix_output)

            enc_mix_layer_logits = result['enc_mix_layer_logits']
            dec_mix_layer_logits = result['dec_mix_layer_logits']
            log_prob = result['log_prob']

            # translation_loss
            translation_loss = self.criterion(
                log_prob.contiguous().view(-1, log_prob.size(-1)),
                new_batch.trg.contiguous().view(-1)
            )
            # get summation loss
            loss_dict['translation_loss'] = loss_dict.get('translation_loss', 0) + translation_loss.item()
            # do average for convenient compute with other loss
            translation_loss = translation_loss / new_batch.ntokens.item()

            current_dataset_domain = self.train_dataset_domain[i]
            current_dataset_label = self.domain_label[current_dataset_domain]
            # enc_classify_loss

            enc_classify_loss = 0
            if self.enc_mix_loss_type is not None:
                if self.enc_mix_loss_type == 'token':
                    enc_classify_label = torch.zeros_like(new_batch.src).fill_(current_dataset_label).to(
                        new_batch.src.device)
                    src_mask = new_batch.src.ne(self.vocab['src'].stoi['<pad>'])
                    enc_classify_label = enc_classify_label.masked_fill(src_mask == 0, -1).long()

                else:  # sent level
                    enc_classify_label = torch.zeros(new_batch.src.size(0)).fill_(current_dataset_label).to(
                        new_batch.src.device).long()

                enc_classify_label = enc_classify_label.expand(len(enc_mix_layer_logits), *(enc_classify_label.size()))
                enc_mix_layer_logits = torch.stack(enc_mix_layer_logits, dim=0)
                enc_mix_layer_log_probs = torch.log_softmax(enc_mix_layer_logits, dim=-1)
                enc_classify_loss = self.classify_criterion(
                    enc_mix_layer_log_probs.view(-1, enc_mix_layer_log_probs.size(-1)),
                    enc_classify_label.contiguous().view(-1))
                enc_counts = enc_classify_label.contiguous().view(-1).ne(-1).sum()

                loss_dict['enc_classify_loss'] = loss_dict.get('enc_classify_loss', 0) + enc_classify_loss.item()
                loss_dict['enc_sum_counts'] = loss_dict.get('enc_sum_counts', 0) + enc_counts.item()

                enc_classify_loss = self.enc_mix_loss_factor * enc_classify_loss / enc_counts

            # dec_classify_loss

            dec_classify_loss = 0
            if self.dec_mix_loss_type is not None:
                dec_classify_label = torch.zeros_like(new_batch.trg).fill_(current_dataset_label).to(
                    new_batch.trg.device)
                dec_mask = new_batch.trg.ne(self.vocab['trg'].stoi['<pad>'])
                dec_classify_label = dec_classify_label.masked_fill(dec_mask == 0, -1)
                dec_classify_label = dec_classify_label.expand(len(dec_mix_layer_logits),
                                                               *(dec_classify_label.size())).long()
                dec_mix_layer_logits = torch.stack(dec_mix_layer_logits, dim=0)
                dec_mix_layer_log_probs = torch.log_softmax(dec_mix_layer_logits, dim=-1)
                dec_classify_loss = self.classify_criterion(
                    dec_mix_layer_log_probs.view(-1, dec_mix_layer_log_probs.size(-1)),
                    dec_classify_label.contiguous().view(-1))

                dec_counts = dec_classify_label.contiguous().view(-1).ne(-1).sum()
                loss_dict['dec_classify_loss'] = loss_dict.get('dec_classify_loss', 0) + dec_classify_loss.item()
                loss_dict['dec_sum_counts'] = loss_dict.get('dec_sum_counts', 0) + dec_counts.item()

                dec_classify_loss = self.dec_mix_loss_factor * dec_classify_loss / dec_counts

            loss = translation_loss + enc_classify_loss + dec_classify_loss
            # if self.criterion.reduction == 'mean':
            #     loss = loss / new_batch.ntokens.item()

            loss.backward()

            loss_dict['sum_loss'] += loss.item() * new_batch.ntokens.item()
            loss_dict['sum_tokens'] += new_batch.ntokens.item()

        return loss_dict

    def write_train_loss(self, loss_dict, loss_tag, record_step):
        """

        :param record_step:
        :param loss_dict:
        :param loss_tag: 'epoch' or 'step'
        :return:
        """

        sum_tokens_description = '{}_sum_tokens'.format(loss_tag)
        self.writer.add_scalar(tag='{}_train_loss'.format(loss_tag),
                               scalar_value=loss_dict['{}_sum_loss'.format(loss_tag)] / loss_dict[
                                   sum_tokens_description],
                               global_step=record_step)

        if self.train_stage == 'sent_kd':
            for description in loss_dict:
                description_split = description.split('_')
                if description_split[1] == 'dataset' and description_split[
                    -1] == 'loss':  # epoch_dataset_0_loss epoch_dataset_
                    self.writer.add_scalar(tag=description,
                                           scalar_value=loss_dict[description] / loss_dict[
                                               description.replace('loss', 'tokens')],
                                           global_step=record_step)

        if self.train_stage == 'knowledge_distillation' or self.train_stage == 'hidden_knowledge_distillation':
            for description in loss_dict:
                if description.endswith('kd_loss') or description.endswith('translation_loss'):
                    self.writer.add_scalar(tag=description,
                                           scalar_value=loss_dict[description] / loss_dict[sum_tokens_description],
                                           global_step=record_step)

        if self.train_stage == 'hidden_knowledge_distillation':
            for description in loss_dict:
                if description.endswith('kd_loss') or description.endswith('translation_loss'):
                    self.writer.add_scalar(tag=description,
                                           scalar_value=loss_dict[description] / loss_dict[sum_tokens_description],
                                           global_step=record_step)
                elif description.endswith('hidden_loss'):
                    self.writer.add_scalar(tag=description,
                                           scalar_value=loss_dict[description] / loss_dict[
                                               description.replace('loss', 'counts')],
                                           global_step=record_step)

        if self.train_stage == 'train_mix_layer_for_adapter':
            for description in loss_dict:
                if description.endswith('translation_loss'):
                    self.writer.add_scalar(tag=description,
                                           scalar_value=loss_dict[description] / loss_dict[sum_tokens_description],
                                           global_step=record_step)
                elif description.endswith('enc_classify_loss') or description.endswith('dec_classify_loss'):
                    self.writer.add_scalar(tag=description,
                                           scalar_value=loss_dict[description] / loss_dict[
                                               description.replace('classify_loss', 'sum_counts')],
                                           global_step=record_step)
