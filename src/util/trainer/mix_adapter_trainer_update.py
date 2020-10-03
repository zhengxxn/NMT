from util.trainer.transformer_trainer import Trainer
from util.decoding.mix_adapter_decoding import beam_search
from tqdm import tqdm
import torch
import torch.nn as nn
import math


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
            self.kl_criterion = torch.nn.KLDivLoss(reduction='none')

        elif self.train_stage == 'train_shared_adapter':

            self.go_through_shared_adapter = True
            self.train_dataset_domain = train_config['dataset_domain']
            self.classify_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
            # self.differ_criterion = nn.MSELoss()

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

        sum_tokens = 0
        sum_loss = 0

        for i, batch in enumerate(batches):
            new_batch = self.rebatch(batch)
            target_adapter_result = model.forward(new_batch.src, new_batch.src_mask,
                                                  new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                                  self.target_domain,
                                                  mix_output=self.mix_output,
                                                  used_domain_list=self.used_domain_list,
                                                  go_through_shared_adapter=self.go_through_shared_adapter)
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

                # todo: need mask
                unreduced_domain_kd_loss = self.kl_criterion(
                    (target_domain_logit / temperature).log_softmax(-1),
                    (ref_domain_logit / temperature).softmax(-1)).sum(-1)

                unreduced_domain_kd_loss = unreduced_domain_kd_loss.masked_fill(trg_mask==0, value=0)
                current_domain_kd_loss = factor * (temperature ** 2) * unreduced_domain_kd_loss.sum()

                loss_dict['{}_kd_loss'.format(ref_domain)] = current_domain_kd_loss.item()
                kd_loss = kd_loss + current_domain_kd_loss

            loss = translation_loss + kd_loss

            if self.criterion.reduction == 'mean':
                loss = loss / new_batch.ntokens.item()

            loss.backward()

            loss_dict['kd_loss'] = kd_loss.item()
            loss_dict['translation_loss'] = translation_loss.item()
            loss_dict['sum_loss'] += loss.item()
            loss_dict['sum_tokens'] += new_batch.ntokens.item()

        # {'sum_loss': , 'sum_tokens': , 'kd_loss': , 'translation_loss': , 'domain_kd_loss': ...}
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
                               scalar_value=loss_dict['{}_sum_loss'.format(loss_tag)] / loss_dict[sum_tokens_description],
                               global_step=record_step)

        if self.train_stage == 'knowledge_distillation':
            for description in loss_dict:
                if description[-7:] == 'kd_loss':
                    self.writer.add_scalar(tag=description,
                                           scalar_value=loss_dict[description] / loss_dict[sum_tokens_description],
                                           global_step=record_step)
