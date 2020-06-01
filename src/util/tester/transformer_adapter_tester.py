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
