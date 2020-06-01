from util.trainer.transformer_trainer import Trainer
from util.batch.batch_with_split_position import TrainingBatch
from util.decoding.transformer_split_position_decoding import beam_search
import torch


class Split_Position_Trainer(Trainer):

    def __init__(self,
                 model,
                 criterion,
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
                 device,)

        self.src_subword_mask = torch.zeros((len(self.vocab['src'])), dtype=torch.long, requires_grad=False).to(self.device)
        self.trg_subword_mask = torch.zeros((len(self.vocab['trg'])), dtype=torch.long, requires_grad=False).to(self.device)
        for i, word in enumerate(self.vocab['src'].itos):
            if len(word) > 2 and word[-2:] == '@@':
                self.src_subword_mask[i] = 1
        for i, word in enumerate(self.vocab['trg'].itos):
            if len(word) > 2 and word[-2:] == '@@':
                self.trg_subword_mask[i] = 1
        # self.register_buffer('mid_word_mask', mid_word_mask)

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
            log_prob = model.forward(new_batch.src, new_batch.src_mask,
                                     new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                     new_batch.src_inner_index, new_batch.src_outer_index,
                                     new_batch.trg_inner_index, new_batch.trg_outer_index)['log_prob']
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
        log_prob = self.model.forward(new_batch.src, new_batch.src_mask,
                                      new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                      new_batch.src_inner_index, new_batch.src_outer_index,
                                      new_batch.trg_inner_index, new_batch.trg_outer_index)['log_prob']
        loss = self.criterion(
            log_prob.contiguous().view(-1, log_prob.size(-1)),
            new_batch.trg.contiguous().view(-1)
        )
        return loss.item(), new_batch.ntokens.item()

    def validation_decoding_step(self, batch):
        search_results = beam_search(
            model=self.model,
            src=batch.src,
            src_subword_mask=self.src_subword_mask,
            sos_index=self.vocab['trg'].stoi['<sos>'],
            eos_index=self.vocab['trg'].stoi['<eos>'],
            pad_index=self.vocab['trg'].stoi['<pad>'],
            max_len=self.decoding_max_steps,
            beam_size=self.decoding_beam_size,
            length_penalty=self.decoding_length_penalty,
            alpha=self.decoding_alpha,
            use_multiple_gpu=self.use_multiple_gpu,
        )

        return search_results

    def rebatch(self, batch, option=None):
        return TrainingBatch(batch=batch,
                             src_subword_mask=self.src_subword_mask,
                             trg_subword_mask=self.trg_subword_mask,
                             pad=self.vocab['trg'].stoi['<pad>'])
