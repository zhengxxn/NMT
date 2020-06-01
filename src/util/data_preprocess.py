import os
import sys

import yaml
from tqdm import tqdm

from util.convenient_funcs import new_save_to_tsv


def open_vocab(path):
    with open(path, 'r') as f:
        raw_vocab = f.readlines()
        vocab = {}
        for pair in raw_vocab:
            word, count = tuple(pair.split(' '))
            vocab[word] = count
    return vocab


def write_vocab(vocab, path):
    with open(path, 'w') as f:
        lines = []
        for k, v in vocab.items():
            lines.append(str(k)+' '+str(v))
        f.writelines(lines)


def transfer_count_to_rate(vocab):
    total_count = 0

    for word in vocab:
        total_count += int(vocab[word])

    for word in vocab:
        vocab[word] = int(vocab[word]) / total_count

    return vocab


def _split_vocab(domain1_vocab_path, domain2_vocab_path, entire_vocab_path,
                 domain1_frequent_vocab_path, domain2_frequent_vocab_path, common_vocab_path,
                 threshold=0):

    domain1_vocab = open_vocab(domain1_vocab_path)
    domain2_vocab = open_vocab(domain2_vocab_path)
    entire_vocab = open_vocab(entire_vocab_path)

    print('domain1 vocab len ', len(domain1_vocab.keys()))
    print('domain2 vocab len ', len(domain2_vocab.keys()))
    print('entire len', len(entire_vocab.keys()))

    common_vocab = {}
    domain1_frequent_vocab = {}  # frequent in domain1 vocab, unfrequent (or never) in domain2 vocab
    domain2_frequent_vocab = {}  # frequent in domain2 vocab, unfrequent (or never) in domain1 vocab

    domain1_vocab = transfer_count_to_rate(domain1_vocab)
    domain2_vocab = transfer_count_to_rate(domain2_vocab)
    mean_domain1_frequency = 1 / len(domain1_vocab.keys())
    mean_domain2_frequency = 1 / len(domain2_vocab.keys())

    for word in entire_vocab.keys():
        word_domain1_frequency = domain1_vocab[word] if word in domain1_vocab.keys() else 0
        word_domain2_frequency = domain2_vocab[word] if word in domain2_vocab.keys() else 0

        if word_domain1_frequency >= mean_domain1_frequency and word_domain2_frequency >= mean_domain2_frequency:
            common_vocab[word] = entire_vocab[word]
        elif word_domain1_frequency >= mean_domain1_frequency and word_domain2_frequency < mean_domain2_frequency:
            rate = (word_domain1_frequency - word_domain2_frequency) / word_domain1_frequency
            if rate >= threshold:
                domain1_frequent_vocab[word] = entire_vocab[word]
            else:
                common_vocab[word] = entire_vocab[word]
        elif word_domain1_frequency < mean_domain1_frequency and word_domain2_frequency >= mean_domain2_frequency:
            rate = (word_domain2_frequency - word_domain1_frequency) / word_domain2_frequency
            if rate >= threshold:
                domain2_frequent_vocab[word] = entire_vocab[word]
            else:
                common_vocab[word] = entire_vocab[word]
        else:
            if word_domain2_frequency == 0:
                domain1_frequent_vocab[word] = entire_vocab[word]
            elif word_domain1_frequency == 0:
                domain2_frequent_vocab[word] = entire_vocab[word]
            else:
                common_vocab[word] = entire_vocab[word]

        # if domain1_frequency > threshold and domain2_frequency > threshold:
        #     common_vocab[word] = entire_vocab[word]
        # elif domain1_frequency > threshold >= domain2_frequency:
        #     domain1_frequent_vocab[word] = entire_vocab[word]
        # elif domain1_frequency <= threshold < domain2_frequency:
        #     domain2_frequent_vocab[word] = entire_vocab[word]
        # else:
        #     common_vocab[word] = entire_vocab[word]

    # domain2_frequent_vocab = {k: v for k, v in entire_vocab.items() if k not in common_vocab}
    # domain2_frequent_vocab = {k: v for k, v in domain2_frequent_vocab.items() if k not in domain1_frequent_vocab}

    print('domain1_frequent_vocab len:', len(domain1_frequent_vocab.keys()))
    print('domain2_frequent_vocab len', len(domain2_frequent_vocab.keys()))
    print('common_vocab', len(common_vocab.keys()))
    print('sum ', len(domain1_frequent_vocab.keys()) + len(domain2_frequent_vocab.keys()) + len(common_vocab.keys()))

    write_vocab(domain1_frequent_vocab, domain1_frequent_vocab_path)
    write_vocab(domain2_frequent_vocab, domain2_frequent_vocab_path)
    write_vocab(common_vocab, common_vocab_path)


def split_vocab(configs):
    for config in configs:
        _split_vocab(domain1_vocab_path=config['domain1_vocab'],
                     domain2_vocab_path=config['domain2_vocab'],
                     entire_vocab_path=config['entire_vocab'],
                     domain1_frequent_vocab_path=config['domain1_frequent_vocab_path'],
                     domain2_frequent_vocab_path=config['domain2_frequent_vocab_path'],
                     common_vocab_path=config['common_vocab_path'],
                     threshold=config['threshold'])


def _data_label(domain_vocab_path_list: list,
                domain_word_label_list: list,
                input_file_path,
                output_file_path):
    """

    :param domain_vocab_path_list: [common vocab path, domain1 vocab path, domain2 vocab path, ...]
    :param domain_word_label_list: [common vocab word label, domain1 vocab word label, domain2 vocab word label, ...]
    :param input_file_path:
    :param output_file_path:
    :return:
    """

    domain_vocab_list = [open_vocab(domain_vocab_path) for domain_vocab_path in domain_vocab_path_list]

    with open(input_file_path, 'r') as f_read:
        with open(output_file_path, 'w', encoding='utf-8') as f_write:
            raw_data = f_read.readlines()
            label_data = []
            for sample in raw_data:
                sample_split = sample[:-1].split(' ')
                label_spilt = []
                for word in sample_split:
                    labeled = False
                    for domain_vocab, domain_label in zip(domain_vocab_list, domain_word_label_list):
                        if word in domain_vocab:
                            label_spilt.append(str(domain_label))
                            labeled = True
                            continue
                    if not labeled:
                        label_spilt.append(str(domain_word_label_list[0]))
                label_data.append(' '.join(label_spilt) + '\n')

            f_write.writelines(label_data)


def data_label(configs):
    for config in configs:
        for data_path, label_path in zip(config['data_path'],
                                         config['label_path']):
            _data_label(domain_vocab_path_list=config['domain_vocab_path_list'],
                        domain_word_label_list=config['domain_word_label_list'],
                        input_file_path=data_path,
                        output_file_path=label_path,)


def _append_suffix(data_path: list, suffix: list, output_path: list):
    # sed 's/$/ <laws> <eos>/' test.txt > test.txt
    for in_path, out_path in zip(data_path, output_path):

        cmd_str = "sed 's/$/ " + " ".join(suffix) + "/' " + in_path + " > " + out_path
        print(cmd_str)
        os.system(cmd_str)


def append_suffix(configs):
    for config in configs:
        _append_suffix(config['data_path'], config['suffix'], config['output_path'])


def concat(config):
    com_path = ' '.join(config['files'])
    cmd_str = 'cat ' + com_path + ' > ' + config['obj_file']
    print(cmd_str)
    os.system(cmd_str)


def lowercase(configs):
    lowercase_script = '../scripts/lowercase.perl '

    for config in configs:
        language = config['language']
        dir_path = config['dir']
        pre_lower_path = [dir_path + '/' + file for file in config['files']]
        lower_path = [path[:-3] + '.lower' + path[-3:] for path in pre_lower_path]

        for s_file, t_file in zip(pre_lower_path, lower_path):
            cmd_str = lowercase_script + \
                      ' < ' + s_file + ' > ' + \
                      t_file
            print(cmd_str)
            os.system(cmd_str)


def tokenize(configs):
    """
    do normalize-punctuation and tokenize
    :param configs:
    :return:
    """
    tokenize_script = '../scripts/tokenizer/tokenizer.perl '
    normalize_script = '../scripts/tokenizer/normalize-punctuation.perl'
    remove_non_printing_char_script = '../scripts/tokenizer/remove-non-printing-char.perl'

    for config in configs:
        language = config['language']
        dir_path = config['dir']
        pre_token_path = [dir_path + '/' + file for file in config['files']]
        token_path = [path[:-3] + '.tok' + path[-3:] for path in pre_token_path]

        for s_file, t_file in zip(pre_token_path, token_path):

            temp_t_file = t_file + '.temp'

            # cmd_str = 'cat ' + s_file + ' | ' + \
            #           normalize_script + ' -l ' + language + ' | ' + \
            #           remove_non_printing_char_script + ' | ' + \
            #           tokenize_script + ' -threads 8 -l ' + language + \
            #           ' > ' + temp_t_file

            cmd_str = 'cat ' + s_file + ' | ' + \
                      normalize_script + ' -l ' + language + ' | ' + \
                      tokenize_script + ' -threads 8 -l ' + language + \
                      ' > ' + temp_t_file

            # cmd_str = 'cat ' + s_file + ' | ' + \
            #           tokenize_script + ' -threads 8 -l ' + language + \
            #           ' > ' + temp_t_file

            print(cmd_str)
            os.system(cmd_str)

            with open(s_file, 'r') as f:
                length = str(len(f.read().splitlines()))
            # length = os.popen('wc -l ' + s_file).read().strip().split()[0]

            cmd_str = 'head -n ' + length + ' ' + temp_t_file + ' > ' + t_file
            print(cmd_str)
            os.system(cmd_str)

            cmd_str = 'rm ' + temp_t_file
            print(cmd_str)
            os.system(cmd_str)


def train_truecase(configs):
    train_truecase_script = '../scripts/recaser/train-truecaser.perl'
    for config in configs:
        s_file = ' '.join(config['files'])
        model = config['model']
        # cmd_str = # 'cat ' + s_file + ' | ' +\
        cmd_str = train_truecase_script + ' -corpus ' + s_file + ' -model ' + model

        print(cmd_str)
        os.system(cmd_str)


def truecase(configs):
    truecase_script = '../scripts/recaser/truecase.perl'
    for config in configs:
        s_file = config['file']
        model = config['model']
        output_file = config['output_file']
        cmd_str = truecase_script + ' < ' + s_file + ' > ' + output_file + ' -model ' + model

        print(cmd_str)
        os.system(cmd_str)


def clean(configs):

    clean_script = '../scripts/clean-corpus-n.perl '

    for config in configs:
        min_len = config['min_len']
        max_len = config['max_len']
        src_language = config['src_language']
        trg_language = config['trg_language']
        file_prefix = config['file_prefix']
        ratio = config['ratio']
        clean_file_prefix = file_prefix + '.clean'

        cmd_str = clean_script + ' -ratio ' + str(ratio) + ' ' + \
                  file_prefix + \
                  " " + src_language + " " + trg_language + " " + \
                  clean_file_prefix + \
                  " " + str(min_len) + " " + str(max_len)
        print(cmd_str)
        os.system(cmd_str)


def apply_bpe(configs):

    # apply_bpe_script = '../scripts/subword-nmt-master/subword_nmt/apply_bpe.py '
    apply_bpe_script = ' subword-nmt applt-bpe '

    for config in configs:
        dir_path = config['dir']
        src_apply_path = [dir_path + '/' + file for file in config['files']]

        code_file = config['code_file']
        for file in src_apply_path:
            cmd_str = apply_bpe_script + \
                      " -c " + code_file + \
                      " < " + file + " > " + \
                      file[:-3] + '.bpe' + file[-3:]
            print(cmd_str)
            os.system(cmd_str)


def learn_bpe(configs):
    # learn_bpe_script = '../scripts/subword-nmt-master/subword_nmt/learn_bpe.py '
    learn_bpe_script = ' subword-nmt learn-bpe'

    for config in configs:
        bpe_operation = config['bpe_operation']
        code_file = config['code_file']
        dir_path = config['dir']
        src_train_path = [dir_path + '/' + file for file in config['files']]

        if len(src_train_path) == 1:
            cmd_str = learn_bpe_script + " -s " + str(bpe_operation) + \
                      " < " + src_train_path[0] + " > " + \
                      code_file

            print(cmd_str)
            os.system(cmd_str)

        else:
            comb_path = ""
            for i, file in enumerate(src_train_path):
                assert os.path.exists(file)
                comb_path += file + " "
            cmd_str = "cat " + comb_path + " | " + \
                      learn_bpe_script + " -s " + str(bpe_operation) + \
                      ' -o ' + code_file

            print(cmd_str)
            os.system(cmd_str)


def get_vocab(configs):
    # get_vocab_script = '../scripts/subword-nmt-master/subword_nmt/get_vocab.py'
    get_vocab_script = ' subword-nmt get-vocab '

    for config in configs:
        dir_path = config['dir']
        files = [dir_path + '/' + file for file in config['files']]
        vocab_file = config['vocab_file']

        if len(files) == 1:
            cmd_str = get_vocab_script + ' --input ' + files[0] + \
                      ' --output ' + vocab_file
            print(cmd_str)
            os.system(cmd_str)
        else:
            comb_path = ''
            for i, file in enumerate(files):
                assert os.path.exists(file)
                comb_path += file + ' '
            cmd_str = 'cat ' + comb_path + ' | ' + \
                      get_vocab_script + ' -o ' + vocab_file

            print(cmd_str)
            os.system(cmd_str)


def get_position_index(configs):
    """
    get outer and inner positional index of seq. for example:
    i ha@@ ve a pen@@ ci@@ l
    outer index: [0, 1, 1, 2, 3, 3, 3]
    inner index: [0, 0, 1, 0, 0, 1, 2]
    :param configs:
    :return:
    """
    for config in configs:
        outer_position_index = []
        inner_position_index = []
        with open(config['dir'] + '/' + config['raw_file'], 'r') as f:
            raw_file = f.read().splitlines()
        for seq in tqdm(raw_file):
            current_seq_outer_position_index = []
            current_seq_inner_position_index = []
            current_outer_position_index = 0
            current_inner_position_index = 0
            if config['is_target']:
                current_seq_outer_position_index.append(str(0))
                current_seq_inner_position_index.append(str(0))
                current_outer_position_index = 1
                current_inner_position_index = 0

            seq_split = seq.split(' ')
            for word in seq_split:
                current_seq_outer_position_index.append(str(current_outer_position_index))
                current_seq_inner_position_index.append(str(current_inner_position_index))

                if len(word) > 2 and word[-2:] == '@@':
                    current_inner_position_index += 1
                else:
                    current_inner_position_index = 0
                    current_outer_position_index += 1

            # if config['is_target']:
            #     current_seq_outer_position_index.append(str(current_outer_position_index))
            #     current_seq_inner_position_index.append(str(current_inner_position_index))

            outer_position_index.append(" ".join(current_seq_outer_position_index))
            inner_position_index.append(" ".join(current_seq_inner_position_index))

        with open(config['dir'] + '/' + config['outer_index_file'], 'w') as f:
            f.write("\n".join(outer_position_index))
        with open(config['dir'] + '/' + config['inner_index_file'], 'w') as f:
            f.write("\n".join(inner_position_index))


def save_tsv(configs):

    for config in configs:
        tsv_path = config['tsv_path']
        new_save_to_tsv(config['tsv_format'], tsv_path)


def action(name, config):
    if name == 'lower':
        lowercase(config['lowercase'])

    elif name == 'token':
        tokenize(config['tokenize'])

    elif name == 'clean':
        clean(config['clean'])

    elif name == 'learn_bpe':
        learn_bpe(config['learn_bpe'])

    elif name == 'apply_bpe':
        apply_bpe(config['apply_bpe'])

    elif name == 'get_vocab':
        get_vocab(config['get_vocab'])

    elif name == 'concat':
        concat(config['concat'])

    elif name == 'split_vocab':
        split_vocab(config['split_vocab'])

    elif name == 'data_label':
        data_label(config['data_label'])

    elif name == 'save_tsv':
        save_tsv(config['save_tsv'])

    elif name == 'append_suffix':
        append_suffix(config['append_suffix'])

    elif name == 'train_truecase':
        train_truecase(config['train_truecase'])

    elif name == 'truecase':
        truecase(config['truecase'])

    elif name == 'get_position_index':
        get_position_index(config['get_position_index'])


def main():
    config_file_path = sys.argv[1]
    actions = str(sys.argv[2]).split('->')

    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)
        for ac in actions:
            action(ac, config)


if __name__ == "__main__":
    main()
