from util.tester.transformer_tester import TransformerTester
from util.decoding.adapter_decoding import beam_search


class TransformerAdapterTester(TransformerTester):

    def decoding_step(self, batch):
        search_results = beam_search(
            model=self.model,
            src=batch.src,
            target_domain=self.config['Test']['target_domain'],
            sos_index=self.vocab['trg'].stoi['<sos>'],
            eos_index=self.vocab['trg'].stoi['<eos>'],
            pad_index=self.vocab['trg'].stoi['<pad>'],
            max_len=self.config['Test']['beam_search']['max_steps'],
            beam_size=self.config['Test']['beam_search']['beam_size'],
            length_penalty=self.config['Test']['beam_search']['length_penalty'],
            alpha=self.config['Test']['beam_search']['alpha'],
        )

        return search_results

    def test_step(self, batch):
        new_batch = self.rebatch(batch)
        log_prob = self.model.forward(new_batch.src,
                                      new_batch.src_mask,
                                      new_batch.trg_input,
                                      new_batch.trg,
                                      new_batch.trg_mask,
                                      target_domain=self.config['Test']['target_domain'])['log_prob']
        loss = self.test_criterion(
            log_prob.contiguous(),
            new_batch.trg.contiguous(),
        )  # [batch]

        return loss
