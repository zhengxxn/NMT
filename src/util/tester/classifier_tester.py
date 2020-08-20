import torch
from util.data_loader.mt_data_loader import MTDataLoader
from util.model_builder import ModelBuilder
from util.convenient_funcs import create_path, get_path_prefix
from tqdm import tqdm
from util.batch.transformer_batch import SrcTestBatch


class ClassifierTester:

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

        model = model.to(device)
        model.domain_mask = model.domain_mask.to(device)
        self.model = model
        self.domain_dict = config['Model']['domain_dict']

    def test(self):
        for test_iterator, test_output_path in \
                zip(self.test_iterators, self.config['Test']['output_path']):

            create_path(get_path_prefix(test_output_path))
            self.model.eval()
            with torch.no_grad():

                hypotheses = []

                with tqdm(test_iterator) as bar:
                    bar.set_description("inference")

                    for batch in bar:
                        # [batch size, max len]
                        new_batch = SrcTestBatch(batch.src, self.vocab['src'].stoi['<pad>'])

                        result = self.model.classify_forward(new_batch.src, new_batch.src_mask, None, train=False)
                        logits = result['emb_classify_logits']
                        prediction = torch.max(logits, dim=-1)[1]
                        for i in range(0, prediction.size(0)):
                            predict = prediction[i].item()
                            for domain in self.domain_dict:
                                if self.domain_dict[domain] == predict:
                                    hypotheses.append(domain)

                with open(test_output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(hypotheses))