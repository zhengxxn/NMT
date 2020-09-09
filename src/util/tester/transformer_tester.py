import torch
from util.data_loader.mt_data_loader import MTDataLoader
from util.model_builder import ModelBuilder
from util.decoding.transformer_decoding import beam_search
from util.convenient_funcs import create_path, tensor2str, de_bpe, get_path_prefix
from util.batch.transformer_batch import SrcTestBatch
from module.criterion.test_nll_loss import TestNLLLoss
from util.batch.transformer_batch import TrainingBatch
from tqdm import tqdm
import os
import sacrebleu


class TransformerTester:

    def __init__(self, config, device, model_name):
        self.config = config
        self.device = device

        mt_data_loader = MTDataLoader(config)
        mt_data_loader.load_datasets(load_train=False, load_dev=False, load_test=True)
        mt_data_loader.build_vocab()
        mt_data_loader.build_iterators(device=device, build_train=False, build_dev=False, build_test=True)

        self.vocab = mt_data_loader.vocab
        self.test_iterators = mt_data_loader.test_iterators

        model_builder = ModelBuilder()
        model = model_builder.build_model(model_name=model_name,
                                          model_config=config['Model'],
                                          vocab=self.vocab,
                                          device=device,
                                          load_pretrained=False,
                                          pretrain_path=None)

        save_model_state_dict = torch.load(config['Test']['model_path'])
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        save_in_multi_gpu = config['Test']['save_in_multi_gpu']

        if save_in_multi_gpu:
            for k, v in save_model_state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
        else:

            model_dict = model.state_dict()
            model_dict.update(save_model_state_dict)
            model.load_state_dict(model_dict)
            # model.load_state_dict(save_model_state_dict)

        self.target_language = config['Test']['target_language']
        self.test_ref_file_paths = config['Test']['refs']
        self.detruecase_script = config['Test']['detruecase_script']
        self.detokenize_script = config['Test']['detokenize_script']

        model = model.to(device)
        self.model = model

        self.test_criterion = TestNLLLoss(size=len(self.vocab['trg']),
                                          padding_idx=self.vocab['trg'].stoi['<pad>'])

    def decoding_step(self, batch):

        search_results = beam_search(
            model=self.model,
            src=batch.src,
            sos_index=self.vocab['trg'].stoi['<sos>'],
            eos_index=self.vocab['trg'].stoi['<eos>'],
            pad_index=self.vocab['trg'].stoi['<pad>'],
            max_len=self.config['Test']['beam_search']['max_steps'],
            beam_size=self.config['Test']['beam_search']['beam_size'],
            length_penalty=self.config['Test']['beam_search']['length_penalty'],
            alpha=self.config['Test']['beam_search']['alpha'],
        )

        return search_results

    def decoding(self):
        for test_iterator, test_output_path, test_ref_file_path in \
                zip(self.test_iterators, self.config['Test']['output_path'], self.test_ref_file_paths):

            create_path(get_path_prefix(test_output_path))
            self.model.eval()
            with torch.no_grad():

                hypotheses = []
                with open(test_ref_file_path, 'r', encoding='utf-8') as f:
                    references = f.read().splitlines()

                with tqdm(test_iterator) as bar:
                    bar.set_description("inference")

                    for batch in bar:
                        # [batch size, max len]
                        search_results = self.decoding_step(batch)
                        prediction = search_results['prediction']

                        for i in range(prediction.size(0)):
                            hypotheses.append(tensor2str(prediction[i], self.vocab['trg']))

                if self.config['Vocab']['use_bpe']:
                    hypotheses = [de_bpe(sent) for sent in hypotheses]
                    # references = [de_bpe(sent) for sent in references]

                # if config['Validation']['Bleu']['level'] == 'character':
                #     hypotheses = [sacrebleu.tokenize_zh(sent) for sent in hypotheses]
                #     references = [sacrebleu.tokenize_zh(sent) for sent in references]

                test_initial_output_path = test_output_path + '.initial'
                detruecase_path = test_output_path + '.detc'
                # detokenize_path = test_output_path + '.detok'
                with open(test_initial_output_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(hypotheses))
                    # detruecase
                    # os.system(detruecase_script + ' < ' + test_initial_output_path + ' > ' + detruecase_path)
                    os.system(self.detokenize_script + ' -l ' + self.target_language + ' < ' + test_initial_output_path +
                              ' > ' + test_output_path)
                with open(test_output_path, 'r', encoding='utf-8') as f:
                    hypotheses = f.read().splitlines()

                bleu_score = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=self.config['Test']['tokenize'])

                print('some examples')
                for i in range(3):
                    print("hyp: ", hypotheses[i])
                    print("ref: ", references[i])

                print()
                print('bleu scores: ', bleu_score)
                print()

    def test_step(self, batch):
        new_batch = self.rebatch(batch)
        log_prob = self.model.forward(new_batch.src,
                                      new_batch.src_mask,
                                      new_batch.trg_input,
                                      new_batch.trg,
                                      new_batch.trg_mask)['log_prob']
        loss = self.test_criterion(
            log_prob.contiguous(),
            new_batch.trg.contiguous(),
        )  # [batch]

        return loss

    def rebatch(self, batch, option=None):
        return TrainingBatch(batch, pad=self.vocab['trg'].stoi['<pad>'])

    def test_loss(self):

        loss_list = []

        self.model.eval()
        with torch.no_grad():
            for test_iterator in self.test_iterators:
                cur_data_loss = []
                with tqdm(test_iterator) as bar:
                    bar.set_description("loss validation")
                    for batch in bar:

                        loss = self.test_step(batch).tolist()
                        cur_data_loss = cur_data_loss + loss
                loss_list.append(cur_data_loss)
        return loss_list

    def visualize_hidden_state(self, save_file):

        for test_iterator in self.test_iterators:
            self.model.eval()
            with torch.no_grad():
                with tqdm(test_iterator) as bar:
                    bar.set_description("visualize hidden state")
                    hidden_state_list = []
                    for batch in bar:
                        new_batch = SrcTestBatch(batch.src, self.vocab['src'].stoi['<pad>'])
                        state = self.model.prepare_for_decode(new_batch.src, new_batch.src_mask)

                        # [batch size, seq len, hidden size]
                        hidden_state = state['memory']

                        # [batch size, hidden size]
                        sum_hidden_state = torch.sum(hidden_state, dim=1)

                        # seq len
                        seq_len = torch.sum(new_batch.src_mask.squeeze(1).type_as(new_batch.src), -1).unsqueeze(-1)
                        # print(seq_len)

                        # [batch size, hidden size]
                        average_hidden_state = sum_hidden_state / seq_len.type_as(sum_hidden_state)

                        hidden_state_list.append(average_hidden_state)

                    hidden = torch.cat(hidden_state_list, dim=0)

            torch.save(hidden, save_file)