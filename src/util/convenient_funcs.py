import pandas as pd
import re
from pathlib import Path
from collections import Counter
import torch
import random
import numpy as np


def tensor2str(prediction, vocab):
    str = []

    for i in range(0, prediction.size(0)):

        ch = vocab.itos[prediction[i]]

        if ch == '<eos>':
            break
        else:
            str.append(ch)

    return " ".join(str)


def convert_xml_to_plaintext(src_file, trg_file):
    with open(src_file, 'r') as f:
        with open(trg_file, 'w') as wf:
            newlines = []
            lines = f.readlines()
            for (i, line) in enumerate(lines):
                newline = re.sub('<seg id=\"[0-9]+\"> | </seg>', '', line, 2)
                if '<' not in newline:
                    newlines.append(newline)

            wf.writelines(newlines)


def save_to_tsv(file_path_1, file_path_2, tsv_file_path, domain=None):

    with open(file_path_1, encoding='utf-8') as f:
        src = f.read().split('\n')[:-1]
    with open(file_path_2, encoding='utf-8') as f:
        trg = f.read().split('\n')[:-1]

    if domain is not None:
        raw_data = {'src': [line for line in src], 'trg': [line for line in trg], 'domain': [domain for line in src]}
    else:
        raw_data = {'src': [line for line in src], 'trg': [line for line in trg]}
    df = pd.DataFrame(raw_data)
    df.to_csv(tsv_file_path, index=False, sep='\t')


def new_save_to_tsv(config, tsv_file_path):

    raw_data = {}
    for key in config.keys():
        file_name = config[key]
        with open(file_name, encoding='utf-8') as f:
            lines = f.read().split('\n')  #[:-1] # .splitlines()
            print(len(lines))
            # split('\n')[:-1]
            value = [line for line in lines if len(line) > 0]
            raw_data[key] = value

    df = pd.DataFrame(raw_data)
    df.to_csv(tsv_file_path, index=False, sep='\t')


def clean_and_save_to_tsv(config, tsv_file_path):

    file_length = 0
    raw_data = {}
    for key in config.keys():
        file_name = config[key]
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')  #[:-1] # .splitlines()
            if len(lines[-1]) == 0:
                lines = lines[:-1]
            # split('\n')[:-1]
            value = [line for line in lines]
            # print(value[-1])
            file_length = len(value)
            raw_data[key] = value

    remove_idx = []
    for i in range(0, file_length):
        src_line = raw_data['src'][i]
        trg_line = raw_data['trg'][i]
        long_length = len(src_line.split(' ')) if len(src_line.split(' ')) > len(trg_line.split(' ')) else len(trg_line.split(' '))
        short_length = len(src_line.split(' ')) + len(trg_line.split(' ')) - long_length
        if long_length / short_length > 9:
            remove_idx.append(i)

    print(len(remove_idx))
    remove_idx.reverse()
    for index in remove_idx:
        del raw_data['src'][index]
        del raw_data['trg'][index]

    print(len(raw_data['src']))
    print(len(raw_data['trg']))

    df = pd.DataFrame(raw_data)
    df.to_csv(tsv_file_path, index=False, sep='\t')


def get_path_prefix(path):
    return re.sub('/[^/]+$', '', path, 1)


def create_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def de_bpe(str):
    return re.sub(r'@@ |@@ ?$', '', str)


def generate_vocab_counter(file):
    c = Counter()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            word, freq = line.split(' ')
            c[word] = int(freq)

    return c


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

