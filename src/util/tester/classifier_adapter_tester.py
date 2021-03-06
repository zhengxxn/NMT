from util.decoding.adapter_decoding import beam_search
import torch
from util.data_loader.mt_data_loader import MTDataLoader
from util.model_builder import ModelBuilder
from util.convenient_funcs import create_path, tensor2str, de_bpe, get_path_prefix
from tqdm import tqdm
import os
import sacrebleu
from util.batch.transformer_batch import SrcTestBatch


class ClassifierAdapterTester:

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
            model.load_state_dict(save_model_state_dict)

        self.target_language = config['Test']['target_language']
        self.test_ref_file_paths = config['Test']['refs']
        self.detruecase_script = config['Test']['detruecase_script']
        self.detokenize_script = config['Test']['detokenize_script']

        model = model.to(device)
        model.domain_mask = model.domain_mask.to(device)
        self.model = model

    def decoding_step(self, batch, target_domain):

        search_results = beam_search(
            model=self.model,
            src=batch.src,
            target_domain=target_domain,
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

                        if self.config['Test']['target_domain'] is None:
                            new_batch = SrcTestBatch(batch.src, self.vocab['src'].stoi['<pad>'])
                            result = self.model.classify_forward(new_batch.src, new_batch.src_mask)
                            logits = result['emb_classify_logits']
                            logits = torch.softmax(logits, dim=-1)
                            target_domain_prob, target_domain = torch.max(logits, -1)
                            for i in range(0, target_domain_prob.size(0)):
                                if target_domain_prob[i] < 0.90 and target_domain[i].item() != 1:
                                    print('change')
                                    target_domain[i] = 1
                        else:
                            target_domain = self.config['Test']['target_domain']

                        search_results = self.decoding_step(batch, target_domain)
                        prediction = search_results['prediction']

                        for i in range(prediction.size(0)):
                            hypotheses.append(tensor2str(prediction[i], self.vocab['trg']))

                if self.config['Vocab']['use_bpe']:
                    hypotheses = [de_bpe(sent) for sent in hypotheses]

                test_initial_output_path = test_output_path + '.initial'
                with open(test_initial_output_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(hypotheses))
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
