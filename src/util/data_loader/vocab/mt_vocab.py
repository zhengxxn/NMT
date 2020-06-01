from torchtext.vocab import Vocab
from collections import Counter


def generate_vocab_counter(file):
    c = Counter()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            word, freq = line.split(' ')
            c[word] = int(freq)

    return c


def build_vocabs(data_fields: dict,
                 path: dict,
                 max_size: dict,
                 special_tokens: dict):
    data_fields['src'].vocab = Vocab(counter=generate_vocab_counter(path['src']),
                                     max_size=max_size['src'],
                                     specials=special_tokens['src'])
    data_fields['trg'].vocab = Vocab(counter=generate_vocab_counter(path['trg']),
                                     max_size=max_size['trg'],
                                     specials=special_tokens['trg'])
    assert data_fields['src'].vocab.stoi['<pad>'] == data_fields['trg'].vocab.stoi['<pad>']

    vocab = {'src': data_fields['src'].vocab, 'trg': data_fields['trg'].vocab}
    return vocab
