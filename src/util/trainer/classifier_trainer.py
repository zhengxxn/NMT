from util.batch.transformer_batch import TrainingBatch
from util.convenient_funcs import create_path
import numpy as np

from tensorboardX import SummaryWriter

import torch
from torch import nn
from tqdm import tqdm


class ClassifierTrainer:
    def __init__(self,
                 model,
                 criterion,
                 vocab,
                 optimizer,
                 lr_scheduler,
                 train_iterators,
                 train_iterators_domain_list,
                 validation_iterators,
                 validation_iterators_domain_list,
                 domain_dict,
                 optimizer_config,
                 train_config,
                 validation_config,
                 record_config,
                 device,
                 ):

        self.device = device
        self.model = model
        self.vocab = vocab
        self.criterion = criterion

        # iterators
        self.train_iterators = train_iterators
        self.validation_iterators = validation_iterators
        self.train_iterators_domain_list = train_iterators_domain_list
        self.validation_iterators_domain_list = validation_iterators_domain_list
        self.domain_dict = domain_dict

        # optimizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_clip = optimizer_config['grad_clip']

        # train process
        self.epoch_num = train_config['epoch_num']
        self.update_batch_count = train_config['update_batch_count']
        self.use_multiple_gpu = train_config['use_multiple_gpu']
        self.current_epoch = 0
        self.current_step = 0

        # loss_validation:
        self.start_loss_validation_on_steps = validation_config['loss_validation']['start_on_steps']
        self.loss_validation_frequency = validation_config['loss_validation']['frequency']

        # record
        # training record
        self.training_record_path = record_config['training_record_path']
        self.writer = SummaryWriter(self.training_record_path + '/visualization')
        self.average_train_loss = 0
        self.best_validation_loss = 10000.

        # best model save
        self.model_record_path = record_config['model_record']['path']
        create_path(self.model_record_path + '/loss_best')
        self.best_loss_model_path = self.model_record_path + '/loss_best/model'
        self.best_loss_optimizer_path = self.model_record_path + '/loss_best/optimizer'
        self.best_loss_lr_scheduler_path = self.model_record_path + '/loss_best/lr_scheduler'

        self.save_loss_best = record_config['model_record']['best_model_save']['loss_best']
        self.best_model_save_optimizer = record_config['model_record']['best_model_save']['save_optimizer']
        self.best_model_save_lr_scheduler = record_config['model_record']['best_model_save']['save_lr_scheduler']

        # checkpoint save
        self.save_checkpoint_start_on_steps = record_config['model_record']['last_checkpoint_save']['start_on_steps']
        self.save_checkpoint_frequency = record_config['model_record']['last_checkpoint_save']['frequency']
        self.checkpoint_num = record_config['model_record']['last_checkpoint_save']['save_checkpoint_count']
        # create_path(self.model_record_path + '/checkpoint')
        self.checkpoint_path = [self.model_record_path + '/checkpoint' + str(i) for i in range(0, self.checkpoint_num)]
        for checkpoint_path in self.checkpoint_path:
            create_path(checkpoint_path)
        self.checkpoint_model_path = [path + '/model' for path in self.checkpoint_path]
        self.checkpoint_optimizer_path = [path + '/optimizer' for path in self.checkpoint_path]
        self.checkpoint_lr_scheduler_path = [path + '/lr_scheduler' for path in self.checkpoint_path]

        self.checkpoint_save_optimizer = record_config['model_record']['last_checkpoint_save']['save_optimizer']
        self.checkpoint_save_lr_scheduler = record_config['model_record']['last_checkpoint_save']['save_lr_scheduler']

        # Option
        self.step_total_loss = 0
        self.step_total_samples = 0
        self.batch_count = 0
        self.current_checkpoint_index = 0

    def train_step(self, model, batches: list):
        """
        a step includes a forward and a backward, and we rebatch it with some batch options
        :param batches: [domain1 batch, domain2 batch ...]
        :param model:
        :return: sum loss, sum tokens
        """
        sum_samples = 0
        sum_loss = 0

        for batch, domain in zip(batches, self.train_iterators_domain_list):
            new_batch = self.rebatch(batch)

            result = model.classify_forward(new_batch.src, new_batch.src_mask)
            emb_classify_logits = result['emb_classify_logits']

            trg = torch.zeros(new_batch.src.size(0)).long().to(self.device).fill_(self.domain_dict[domain])

            loss = self.criterion(emb_classify_logits, trg)
            sum_loss += loss.item()
            sum_samples += new_batch.src.size(0)

            if self.criterion.reduction == 'mean':
                loss = loss / new_batch.src.size(0)

            loss.backward()

        return sum_loss, sum_samples

    def validation_step(self, batch, domain):
        new_batch = self.rebatch(batch)

        result = self.model.classify_forward(new_batch.src, new_batch.src_mask)

        emb_classify_logits = result['emb_classify_logits']

        trg = torch.zeros(new_batch.src.size(0)).long().to(self.device).fill_(self.domain_dict[domain])

        emb_predicted = torch.max(emb_classify_logits, dim=-1)[1]
        emb_predict_true_count = (emb_predicted == trg).sum().item()

        emb_classify_loss = self.criterion(emb_classify_logits, trg)

        return emb_classify_loss.item(),  emb_predict_true_count, new_batch.src.size(0)

    def train(self):
        """
        :return:
        """
        self.model.train()
        for self.current_epoch in range(self.current_epoch, self.epoch_num):
            train_average_loss = self.run_epoch(self.model, self.train_iterators)

            self.writer.add_scalar(tag='epoch_train_average_loss',
                                   scalar_value=train_average_loss,
                                   global_step=self.current_epoch)

    def validation(self):
        average_loss_list = []
        total_samples, total_emb_true_count = 0, 0

        self.model.eval()
        with torch.no_grad():

            for validation_iterator, domain in zip(self.validation_iterators, self.validation_iterators_domain_list):
                sum_emb_loss = 0
                sum_emb_true_count = 0
                sum_samples = 0

                with tqdm(validation_iterator) as bar:
                    bar.set_description("loss validation")
                    for batch in bar:
                        emb_loss, emb_true_count, n_samples = self.validation_step(batch, domain)
                        sum_emb_loss += emb_loss
                        sum_emb_true_count += emb_true_count
                        sum_samples += n_samples

                        bar.set_postfix({'loss': '{0:1.5f}'.format(emb_loss / n_samples),
                                         'best_validation_loss': '{0:1.5f}'.format(self.best_validation_loss)})
                        bar.update()

                average_emb_loss = sum_emb_loss / sum_samples
                emb_acc = float(sum_emb_true_count) / sum_samples

                total_samples += sum_samples
                total_emb_true_count += sum_emb_true_count

                average_loss_list.append(average_emb_loss)

                self.writer.add_scalar(tag='validation_loss_' + domain, scalar_value=average_emb_loss,
                                       global_step=self.current_step)

                self.writer.add_scalar(tag='validation_emb_loss_' + domain, scalar_value=average_emb_loss,
                                       global_step=self.current_step)

                self.writer.add_scalar(tag='validation_emb_acc_' + domain, scalar_value=emb_acc,
                                       global_step=self.current_step)

        self.writer.add_scalar(tag='average_emb_acc',
                               scalar_value=float(total_emb_true_count) / total_samples,
                               global_step=self.current_step)

        current_loss = np.average(average_loss_list)
        if current_loss < self.best_validation_loss:
            self.best_validation_loss = current_loss
            self.writer.add_scalar(tag='best_validation_loss',
                                   scalar_value=self.best_validation_loss,
                                   global_step=self.current_step)
            if self.save_loss_best:
                self.save_checkpoint(self.best_loss_model_path,
                                     self.best_loss_optimizer_path,
                                     self.best_loss_lr_scheduler_path,
                                     self.best_model_save_optimizer,
                                     self.best_model_save_lr_scheduler)

        if self.lr_scheduler['name'] == 'reduce_lr_on_loss':
            self.lr_scheduler['scheduler'].step(current_loss)

        return average_loss_list

    def run_epoch(self, model, data_iterators):

        self.model.train()
        for data_iterator in data_iterators:
            data_iterator.init_epoch()

        cur_step_in_epoch = 0

        with tqdm(zip(*tuple(data_iterators))) as bar:
            bar.set_description("training epoch: " + str(self.current_epoch))

            epoch_total_samples = 0
            epoch_total_loss = 0

            for batches in bar:
                # end epoch
                if cur_step_in_epoch != 0 and data_iterators[0].state_dict()['iterations_this_epoch'] == 1:
                    break

                if self.batch_count == 0:
                    self.optimizer.zero_grad()

                loss, nsamples = self.train_step(self.model, batches)  # for each token

                self.batch_count = (self.batch_count + 1) % self.update_batch_count
                if self.batch_count == 0:
                    self.current_step += 1
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    if self.lr_scheduler['base_on'] == 'step':
                        self.lr_scheduler['scheduler'].step()
                    cur_step_in_epoch += 1

                epoch_total_loss += loss
                epoch_total_samples += nsamples

                self.step_total_loss += loss
                self.step_total_samples += nsamples

                bar.set_postfix({'loss': '{0:1.5f}'.format(loss / nsamples),
                                 'average_train_loss': '{0:1.5f}'.format(epoch_total_loss / epoch_total_samples),
                                 'best_validation_loss': '{0:1.5f}'.format(self.best_validation_loss)})
                bar.update()

                # validation:
                if self.current_step != 0 and self.batch_count == 0:

                    if self.current_step >= self.start_loss_validation_on_steps \
                            and self.current_step % self.loss_validation_frequency == 0:
                        self.validation()
                        self.writer.add_scalar(tag='step_train_loss',
                                               scalar_value=self.step_total_loss / self.step_total_samples,
                                               global_step=self.current_step)
                        self.step_total_loss, self.step_total_samples = 0., 0
                        self.writer.add_scalar(tag='lr_rate',
                                               scalar_value=[group['lr'] for group in self.optimizer.param_groups][0],
                                               global_step=self.current_step)
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

        epoch_train_loss = epoch_total_loss / epoch_total_samples
        self.average_train_loss = epoch_train_loss
        return epoch_train_loss

    def rebatch(self, batch, option=None):
        return TrainingBatch(batch, pad=self.vocab['trg'].stoi['<pad>'])

    def save_checkpoint(self, model_path, optimizer_path, lr_scheduler_path, save_optimizer, save_lr_scheduler):

        torch.save(self.model.state_dict(), model_path)
        if save_optimizer:
            torch.save(self.optimizer.state_dict(), optimizer_path)
        if save_lr_scheduler:
            torch.save(self.lr_scheduler['scheduler'].state_dict(), lr_scheduler_path)
