from util.trainer.transformer_trainer import Trainer
import torch


class Kd_Trainer(Trainer):

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
                 ref_model,
                 ref_temperature,
                 ref_factor,
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

        self.ref_model = ref_model
        self.ref_temperature = ref_temperature
        self.ref_factor = ref_factor
        self.kl_criterion = torch.nn.KLDivLoss(reduction='none')
        self.ref_model.eval()

    def train_step(self, model, batches: list):
        """
        a step includes a forward and a backward, and we  rebatch it with some batch options
        :param batches: [domain1 batch, domain2 batch ...]
        :param model:
        :return: sum loss, sum tokens
        """
        sum_tokens = 0
        sum_loss = 0

        for batch in batches:
            new_batch = self.rebatch(batch)

            with torch.no_grad():
                ref_result = self.ref_model.forward(new_batch.src, new_batch.src_mask,
                                                    new_batch.trg_input, new_batch.trg, new_batch.trg_mask, )
                ref_logit = ref_result['logit']

            result = model.forward(new_batch.src, new_batch.src_mask,
                                   new_batch.trg_input, new_batch.trg, new_batch.trg_mask, )

            log_prob = result['log_prob']
            logit = result['logit']

            translation_loss = self.criterion(
                log_prob.contiguous().view(-1, log_prob.size(-1)),
                new_batch.trg.contiguous().view(-1)
            )

            temperature = self.ref_temperature
            factor = self.ref_factor

            unreduced_kd_loss = self.kl_criterion(
                (logit / temperature).log_softmax(-1),
                (ref_logit / temperature).softmax(-1)).sum(-1)

            trg_mask = new_batch.trg.ne(self.vocab['trg'].stoi['<pad>'])
            unreduced_kd_loss = unreduced_kd_loss.masked_fill(trg_mask == 0, value=0)
            kd_loss = factor * (temperature ** 2) * unreduced_kd_loss.sum()

            loss = (1 - self.ref_factor) * translation_loss + self.ref_factor * kd_loss
            # loss = model.forward(new_batch)['loss']  # the loss is summation of all tokens

            sum_loss += loss.item()
            sum_tokens += new_batch.ntokens.item()

            if self.criterion.reduction == 'mean':
                loss = loss / new_batch.ntokens.item()

            loss.backward()

        return sum_loss, sum_tokens
