import yaml
import torch
import sys
from util.convenient_funcs import create_path, set_random_seed
from util.data_loader.mt_data_loader import MTDataLoader
from torch.utils.data.dataloader import DataLoader

global max_src_in_batch, max_tgt_in_batch

from torchtext.data.functional import numericalize_tokens_from_iterator


# ids_iter = numericalize_tokens_from_iterator(vocab['src'].stoi, [train_dataset.examples[0].src])


def sent_2_tensor(sent, vocab: dict):
    id_list = [vocab.get(token, vocab['<unk>']) for token in sent]
    return torch.tensor(id_list, dtype=torch.long).unsqueeze(0)


def my_collate_fn(examples: list, vocab):
    src_sents = [example.src for example in examples]
    trg_sents = [example.trg for example in examples]
    src_ids = [sent_2_tensor(sent, vocab['src'].stoi) for sent in src_sents]
    trg_ids = [sent_2_tensor(sent, vocab['trg'].stoi) for sent in trg_sents]
    src_tensor = torch.stack(src_ids, vocab['src'].stoi['<pad>'])
    trg_tensor = torch.stack(trg_ids, vocab['trg'].stoi['<pad>'])
    return src_tensor


def main():
    # config_file_path = sys.argv[1]
    config_file_path = '/Users/zx/Documents/GitHub/NMT/config/translation-transformer/medical-ende.yaml'

    print('read config')
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)
        # create_path(config['Record']['training_record_path'])

    # set random seed
    set_random_seed(config['Train']['random_seed'])

    # ================================================================================== #
    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set the data fields dict['src': (name, field), 'trg': (name, field)]

    # load dataset
    print('load dataset ...')
    mt_data_loader = MTDataLoader(config)
    mt_data_loader.load_datasets(load_train=True, load_dev=True, load_test=False)
    mt_data_loader.build_vocab()
    vocab = mt_data_loader.vocab
    # print(vocab['src'].stoi)

    train_dataset = mt_data_loader.train_datasets[0]
    # print(train_dataset[10].src)
    data_loader = DataLoader(train_dataset, batch_size=4, collate_fn=lambda x: my_collate_fn(x, vocab))
    #
    # print(train_dataset.examples[0].src)
    #
    # from torchtext.data.functional import numericalize_tokens_from_iterator
    # ids_iter = numericalize_tokens_from_iterator(vocab['src'].stoi, [train_dataset.examples[0].src])
    # for ids in ids_iter:
    #     print([num for num in ids])
    i = 0
    for batch in data_loader:
        if i < 3:
            print(batch)
        else:
            break
        i = i + 1
    # from torchtext.experimental.functional import sequential_transforms


if __name__ == "__main__":
    main()
