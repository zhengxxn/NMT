from util.tester.transformer_tester import TransformerTester
from util.decoding.transformer_split_position_decoding import beam_search
import torch


class TransformerSplitPositionTester(TransformerTester):

    def __init__(self, config, device, model_name):
        super().__init__(config, device, model_name)

        self.src_subword_mask = torch.zeros((len(self.vocab['src'])), dtype=torch.long, requires_grad=False).to(device)
        for i, word in enumerate(self.vocab['src'].itos):
            if len(word) > 2 and word[-2:] == '@@':
                self.src_subword_mask[i] = 1

    def decoding_step(self, batch):
        search_results = beam_search(
            model=self.model,
            src=batch.src,
            src_subword_mask=self.src_subword_mask,
            sos_index=self.vocab['trg'].stoi['<sos>'],
            eos_index=self.vocab['trg'].stoi['<eos>'],
            pad_index=self.vocab['trg'].stoi['<pad>'],
            max_len=self.config['Test']['beam_search']['max_steps'],
            beam_size=self.config['Test']['beam_search']['beam_size'],
            length_penalty=self.config['Test']['beam_search']['length_penalty'],
            alpha=self.config['Test']['beam_search']['alpha'],
        )

        return search_results
