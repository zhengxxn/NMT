from util.data_loader.data_fields import mt_data_wo_length_fields
from util.data_loader.dataset import load_datasets, combine_datasets
from util.data_loader.vocab.mt_vocab import build_vocabs
from util.data_loader.iterator.get_iterator import get_train_iterator, get_dev_iterator, get_test_iterator


class MTDataLoader:
    def __init__(self, config):

        self.config = config
        self.train_datasets = None
        self.dev_datasets = None
        self.test_datasets = None
        self.vocab = None
        self.train_iterators = None
        self.dev_iterators = None
        self.test_iterators = None
        self.dev_test_iterators = None

        self.data_fields = mt_data_wo_length_fields()

    def load_datasets(self, load_train=False, load_dev=False, load_test=False):

        if load_train:
            self.train_datasets = load_datasets(paths=self.config['Dataset']['train_dataset_path'],
                                                data_fields=self.data_fields,
                                                filter_len=self.config['Dataset']['filter_len'])
            if self.config['Dataset']['combine_train_datasets']:
                self.train_datasets = [combine_datasets(datasets=self.train_datasets,
                                                        data_fields=self.data_fields,
                                                        filter_len=self.config['Dataset']['filter_len'])]

        if load_dev:
            self.dev_datasets = load_datasets(paths=self.config['Dataset']['dev_dataset_path'],
                                              data_fields=self.data_fields)

        if load_test:
            self.test_datasets = load_datasets(self.config['Dataset']['test_dataset_path'],
                                               data_fields=self.data_fields)

    def build_vocab(self, ):
        self.vocab = build_vocabs(data_fields={'src': self.data_fields[0][1], 'trg': self.data_fields[1][1]},
                                  # list( tuple(name, field) )
                                  path={'src': self.config['Vocab']['src']['file'],
                                        'trg': self.config['Vocab']['trg']['file']},
                                  max_size={'src': self.config['Vocab']['src']['max_size'],
                                            'trg': self.config['Vocab']['trg']['max_size']},
                                  special_tokens={'src': ['<unk>', '<pad>', '<sos>', '<eos>'],
                                                  'trg': ['<unk>', '<pad>', '<sos>', '<eos>']})

    def build_iterators(self, device, build_train=False, build_dev=False, build_test=False):

        if build_train:
            self.train_iterators = [get_train_iterator(dataset=dataset,
                                                       batch_size=self.config['Train']['batch_size'],
                                                       device=device,
                                                       token_level=True)
                                    for dataset in self.train_datasets]

        if build_dev:
            self.dev_iterators = [get_dev_iterator(dataset=dataset,
                                                   batch_size=self.config['Validation']['batch_size'],
                                                   device=device)
                                  for dataset in self.dev_datasets]
            self.dev_test_iterators = [get_test_iterator(dataset=dataset,
                                                         batch_size=self.config['Validation']['batch_size'],
                                                         device=device) for dataset in self.dev_datasets]

        if build_test:
            self.test_iterators = [get_test_iterator(test_dataset,
                                                     batch_size=self.config['Test']['batch_size'],
                                                     device=device)
                                   for test_dataset in self.test_datasets]
