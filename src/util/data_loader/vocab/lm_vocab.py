from torchtext.vocab import Vocab
from .mt_vocab import generate_vocab_counter


def build_vocabs(data_fields: dict,
                 path: str,
                 max_size: int,
                 special_tokens: list):

    data_fields['text'].vocab = Vocab(counter=generate_vocab_counter(path),
                                      max_size=max_size,
                                      specials=special_tokens)

    vocab = {'text': data_fields['text'].vocab}
    return vocab
