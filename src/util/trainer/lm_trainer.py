from util.batch.lm_batch import TrainingBatch
from util.convenient_funcs import create_path

from tensorboardX import SummaryWriter

import torch
import math
from torch import nn
from tqdm import tqdm


class Trainer:
    def __init__(self,
                 model,
                 criterion,
                 validation_criterion,
                 vocab,
                 optimizer,
                 lr_scheduler,
                 train_iterators,
                 validation_iterators,
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
        self.validation_criterion = validation_criterion

        # iterators
        self.train_iterators = train_iterators
        self.validation_iterators = validation_iterators

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
        self.output_path = self.training_record_path + '/output'
        create_path(self.output_path)

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
        self.step_total_tokens = 0
        self.batch_count = 0
        self.current_checkpoint_index = 0

    def train_step(self, model, batches: list):
        """
        a step includes a forward and a backward, and we rebatch it with some batch options
        :param batches: [domain1 batch, domain2 batch ...]
        :param model:
        :return: sum loss, sum tokens
        """
        sum_tokens = 0
        sum_loss = 0

        for batch in batches:
            new_batch = self.rebatch(batch)
            log_prob = model.forward(new_batch.inp, new_batch.trg_mask)['log_prob']
            loss = self.criterion(
                log_prob.contiguous().view(-1, log_prob.size(-1)),
                new_batch.trg.contiguous().view(-1)
            )

            sum_loss += loss.item()
            sum_tokens += new_batch.ntokens.item()

            if self.criterion.reduction == 'mean':
                loss = loss / new_batch.ntokens.item()

            loss.backward()

        return sum_loss, sum_tokens

    def validation_step(self, batch):
        new_batch = self.rebatch(batch)
        log_prob = self.model.forward(new_batch.inp,
                                      new_batch.trg_mask)['log_prob']
        loss = self.validation_criterion(
            log_prob.contiguous().view(-1, log_prob.size(-1)),
            new_batch.trg.contiguous().view(-1)
        )

        return loss.item(), new_batch.ntokens.item()

    def train(self):

        self.model.train()
        for self.current_epoch in range(self.current_epoch, self.epoch_num):
            train_average_loss = self.run_epoch(self.model, self.train_iterators)

            self.writer.add_scalar(tag='epoch_train_average_loss',
                                   scalar_value=train_average_loss,
                                   global_step=self.current_epoch)

    def loss_validation(self):
        average_loss_list = []

        self.model.eval()
        with torch.no_grad():
            for validation_iterator in self.validation_iterators:
                sum_loss = 0
                sum_tokens = 0

                with tqdm(validation_iterator) as bar:
                    bar.set_description("loss validation")
                    for batch in bar:

                        loss, n_tokens = self.validation_step(batch)
                        sum_loss += loss
                        sum_tokens += n_tokens

                        bar.set_postfix({'loss': '{0:1.5f}'.format(loss / n_tokens),
                                         'best_validation_loss': '{0:1.5f}'.format(self.best_validation_loss)})
                        bar.update()

                average_loss = sum_loss / sum_tokens
                average_loss_list.append(average_loss)

        current_loss = average_loss_list[0]
        self.writer.add_scalar(tag='validation_loss', scalar_value=current_loss, global_step=self.current_step)
        self.writer.add_scalar(tag='validation_ppl', scalar_value=math.exp(current_loss), global_step=self.current_step)

        if current_loss < self.best_validation_loss:
            self.best_validation_loss = current_loss
            self.writer.add_scalar(tag='best_validation_loss', scalar_value=self.best_validation_loss,
                                   global_step=self.current_step)
            self.writer.add_scalar(tag='best_validation_ppl', scalar_value=math.exp(self.best_validation_loss),
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

    def validation(self, description: str, step_num: int):
        pass

    def run_epoch(self, model, data_iterators):

        self.model.train()
        for data_iterator in data_iterators:
            data_iterator.init_epoch()

        cur_step_in_epoch = 0

        with tqdm(zip(*tuple(data_iterators))) as bar:
            bar.set_description("training epoch: " + str(self.current_epoch))

            epoch_total_tokens = 0
            epoch_total_loss = 0

            for batches in bar:
                # end epoch
                if cur_step_in_epoch != 0 and data_iterators[0].state_dict()['iterations_this_epoch'] == 1:
                    break

                if self.batch_count == 0:
                    self.optimizer.zero_grad()

                loss, ntokens = self.train_step(self.model, batches)  # for each token

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
                epoch_total_tokens += ntokens

                self.step_total_loss += loss
                self.step_total_tokens += ntokens

                bar.set_postfix({'loss': '{0:1.5f}'.format(loss / ntokens),
                                 'average_train_loss': '{0:1.5f}'.format(epoch_total_loss / epoch_total_tokens),
                                 'best_validation_loss': '{0:1.5f}'.format(self.best_validation_loss)})
                bar.update()

                # validation:
                if self.current_step != 0 and self.batch_count == 0:
                    if self.current_step >= self.start_loss_validation_on_steps \
                            and self.current_step % self.loss_validation_frequency == 0:
                        self.loss_validation()
                        self.writer.add_scalar(tag='step_train_loss',
                                               scalar_value=self.step_total_loss / self.step_total_tokens,
                                               global_step=self.current_step)
                        self.step_total_loss, self.step_total_tokens = 0, 0
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

        epoch_train_loss = epoch_total_loss / epoch_total_tokens
        self.average_train_loss = epoch_train_loss
        return epoch_train_loss

    def rebatch(self, batch, option=None):
        return TrainingBatch(batch, pad=self.vocab['text'].stoi['<pad>'])

    def save_checkpoint(self, model_path, optimizer_path, lr_scheduler_path, save_optimizer, save_lr_scheduler):

        torch.save(self.model.state_dict(), model_path)
        if save_optimizer:
            torch.save(self.optimizer.state_dict(), optimizer_path)
        if save_lr_scheduler:
            torch.save(self.lr_scheduler['scheduler'].state_dict(), lr_scheduler_path)
