from util.trainer.transformer_trainer import Trainer
from util.decoding.adapter_decoding import beam_search


class Adapter_Trainer(Trainer):

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
                 target_domain,
                 ref_domains=None,
                 distillation_criterion=None,
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
            device, )

        if ref_domains is None:
            ref_domains = []
        self.target_domain = target_domain
        self.ref_domains = ref_domains
        self.distillation_criterion = distillation_criterion

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
            target_adapter_result = model.forward(new_batch.src, new_batch.src_mask,
                                                  new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                                  self.target_domain)
            log_prob = target_adapter_result['log_prob']

            for ref_domain in self.ref_domains:
                ref_result = model.forward(new_batch.src, new_batch.src_mask,
                                           new_batch.trg_input, new_batch.trg, new_batch.trg_mask,
                                           ref_domain)

            loss = self.criterion(
                log_prob.contiguous().view(-1, log_prob.size(-1)),
                new_batch.trg.contiguous().view(-1)
            )
            # loss = model.forward(new_batch)['loss']  # the loss is summation of all tokens

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
                                      self.target_domain)['log_prob']
        loss = self.criterion(
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
            use_multiple_gpu=self.use_multiple_gpu,
        )

        return search_results
