import yaml

import torch
import torch.nn as nn

# from util._make_model import make_model, print_model
from util.model_builder.make_model.make_transformer_with_adapter import make_model
from util.data_loader.data_fields import mt_data_fields
from util.data_loader.dataset import load_datasets
from util.data_loader.vocab import build_vocabs
from util._iterator import get_dev_iterator, get_test_iterator
from util.batch.transformer_batch import TrainingBatch
from tensorboardX import SummaryWriter
from util.decoding.transformer_decoding import beam_search
from util.convenient_funcs import tensor2str, create_path, de_bpe
import sacrebleu
import copy


import numpy as np
import random
import sys
import os

import torchtext.data as data

global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """

    :param new: new example to add
    :param count: current count of examples in the batch
    :param sofar: current effective batch size
    :return: the new effective batch size resulting from adding that example to a batch
    """
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def step(model, batch, pad_token):

    sum_tokens = 0
    sum_loss = 0
    num = 0

    # for batch in batches:
    new_batch = rebatch(batch, pad_token)
    loss = model.forward(new_batch)['loss']  # the loss is summation of all tokens

    # sum_loss += loss.item()
    # sum_tokens += new_batch.ntokens.item()
    # loss = loss / new_batch.ntokens.item()
    # loss.backward()

    return loss, new_batch.ntokens.item()


def rebatch(batch, pad_token):
    return TrainingBatch(batch, pad=pad_token)


def loss_validation(model, validation_iterator, pad_token):
    model.eval()
    with torch.no_grad():
        sum_loss = 0
        sum_tokens = 0

        for batch in validation_iterator:
            new_batch = rebatch(batch, pad_token)
            result = model.forward(new_batch)
            loss = result['loss']

            sum_loss += loss.item()
            sum_tokens += new_batch.ntokens.item()

        average_loss = sum_loss / sum_tokens

    return average_loss


def bleu_validation(model, validation_test_iterator, vocab,
                    valid_ref_file_path, output_path,
                    detruecase_script, detokenize_script, target_language
                    ):
    # bleu_scores = []

    model.eval()
    with torch.no_grad():

        hypotheses = []

        with open(valid_ref_file_path, 'r', encoding='utf-8') as f:
            references = f.read().splitlines()
            # references = [" ".join(example.trg) for example in validation_dataset.examples]

        for batch in validation_test_iterator:

            batch = rebatch(batch, vocab['trg'].stoi['<pad>'])
            sorted_index = sorted(range(len(batch.src_length)), key=lambda i: batch.src_length[i],
                                  reverse=True)

            sorted_src = batch.src[sorted_index]
            sorted_lengths = batch.src_length[sorted_index]
            search_results = beam_search(
                model=model,
                src=sorted_src,
                sos_index=vocab['trg'].stoi['<sos>'],
                eos_index=vocab['trg'].stoi['<eos>'],
                pad_index=vocab['trg'].stoi['<pad>'],
                max_len=150,
                beam_size=4,
            )
            prediction = search_results['prediction']

            original_prediction = prediction.clone()
            original_prediction[sorted_index] = prediction

            for i in range(original_prediction.size(0)):
                hypotheses.append(tensor2str(original_prediction[i], vocab['trg']))

            # if self.use_bpe:
        hypotheses = [de_bpe(sent) for sent in hypotheses]

        # then should detruecase and detokenize
        valid_initial_output_path = output_path + '/valid.output.initial'
        detruecase_path = output_path + '/valid.output.detc'
        # detokenize_path = self.output_path + '/valid.output.detok'
        valid_output_path = output_path + '/valid.output'
        with open(valid_initial_output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(hypotheses))
            # detruecase
            os.system(detruecase_script + ' < ' + valid_initial_output_path + ' > ' + detruecase_path)
            os.system(detokenize_script + ' -l ' + target_language + ' < ' + detruecase_path +
                      ' > ' + valid_output_path)
        with open(valid_output_path, 'r', encoding='utf-8') as f:
            hypotheses = f.read().splitlines()

        bleu_score = sacrebleu.corpus_bleu(hypotheses, [references]).score
        # bleu_score = sacrebleu.raw_corpus_bleu(hypotheses, [references]).score
        # bleu_scores.append(bleu_score)

        sample = [random.randint(0, len(hypotheses) - 1) for i in range(3)]
        print('some examples')
        for i in range(3):
            print("hyp: ", hypotheses[sample[i]])
            print("ref: ", references[sample[i]])

        print()
        print('bleu scores: ', bleu_score)
        print()

    return bleu_score


def main():

    torch.manual_seed(3)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3)
    random.seed(3)
    np.random.seed(3)
    torch.backends.cudnn.deterministic = True

    config_file_path = sys.argv[1]

    print('read config')
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)
        create_path(config['Record']['path'])

    # ================================================================================== #
    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set the data fields dict['src': (name, field), 'trg': (name, field)]
    data_fields = mt_data_fields()

    # load dataset
    print('load dataset ...')
    meta_train_datasets = load_datasets(paths=config['Dataset']['meta_train_dataset_path'],
                                        data_fields=data_fields, filter_len=config['Dataset']['filter_len'])

    meta_dev_datasets = load_datasets(paths=config['Dataset']['meta_dev_dataset_path'],
                                      data_fields=data_fields, filter_len=config['Dataset']['filter_len'])

    validation_dataset = load_datasets(paths=config['Dataset']['validation_dataset_path'], data_fields=data_fields)[0]

    # load vocab
    print('begin load vocab ...')
    vocab = build_vocabs(data_fields={'src': data_fields[0][1], 'trg': data_fields[1][1]},  # list( tuple(name, field) )
                         path={'src': config['Vocab']['src']['file'], 'trg': config['Vocab']['trg']['file']},
                         max_size={'src': config['Vocab']['src']['max_size'], 'trg': config['Vocab']['trg']['max_size']},
                         special_tokens={'src': ['<unk>', '<pad>', '<sos>', '<eos>'],
                                         'trg': ['<unk>', '<pad>', '<sos>', '<eos>']})

    meta_train_iterators = [MyIterator(dataset=dataset,
                                       batch_size=config['meta_train']['batch_size'],
                                       device=device, repeat=False,
                                       sort_key=lambda x: (len(x.src), len(x.trg)),
                                       batch_size_fn=batch_size_fn, train=True, shuffle=True)
                            for dataset in meta_train_datasets]

    meta_dev_iterators = [MyIterator(dataset=dataset,
                                     batch_size=config['meta_dev']['batch_size'],
                                     device=device, repeat=False,
                                     sort_key=lambda x: (len(x.src), len(x.trg)),
                                     batch_size_fn=batch_size_fn, train=True, shuffle=True)
                          for dataset in meta_dev_datasets]

    validation_iterator = get_dev_iterator(dataset=validation_dataset, batch_size=config['Validation']['batch_size'],
                                           device=device)
    validation_test_iterator = get_test_iterator(dataset=validation_dataset, batch_size=config['Validation']['batch_size'],
                                                 device=device)

    # init or load the model
    model = make_model(
        vocab=vocab,
        model_config=config['Model']
    )

    if config['meta_train']['load_exist_model']:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(config['meta_train']['model_load_path'])
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.current_domain = config['meta_train']['current_domain']

    # fix except adapter

    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # print(parameters)
    # print(model.state_dict().keys())

    create_path(config['Record']['path'] + '/visualization')
    writer = SummaryWriter(config['Record']['path'] + '/visualization')

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=config['Optimizer']['meta_learning_rate'])
    model_optimizer = torch.optim.Adam(model.parameters(), lr=config['Optimizer']['model_learning_rate'])

    os.system('cp ' + config_file_path + ' ' + config['Record']['path'] + '/model_config.txt')

    turn_num = config['meta_train']['turn_num']
    task_num = config['meta_train']['task_num']
    detruecase_script = config['Dataset']['detruecase_script']
    detokenize_script = config['Dataset']['detokenize_script']
    validation_per_step = config['Validation']['per_steps']
    validation_ref = config['Validation']['ref']
    best_bleu_score = 0.0
    create_path(config['Record']['path'] + '/output')
    create_path(config['Record']['model_record_path'] + '/model')
    best_model_path = config['Record']['model_record_path'] + '/model/best_model'

    model.train()
    for i in range(0, turn_num):

        loss_tasks = []
        init_state = copy.deepcopy(model.state_dict())

        for j in range(0, task_num):
            for k in range(0, len(meta_train_iterators)):

                model.load_state_dict(init_state)
                model_optimizer.zero_grad()

                current_meta_train_iterator = meta_train_iterators[k]
                current_meta_test_iterator = meta_dev_iterators[k]

                support_set_batch = next(iter(current_meta_train_iterator))
                query_set_batch = next(iter(current_meta_test_iterator))

                support_loss, n_tokens = step(model, support_set_batch, pad_token=vocab['trg'].stoi['<pad>'])
                support_loss = support_loss / n_tokens
                support_loss.backward()
                model_optimizer.step()

                model.eval()
                query_loss, n_tokens = step(model, query_set_batch, pad_token=vocab['trg'].stoi['<pad>'])
                query_loss = query_loss / n_tokens
                loss_tasks.append(query_loss)
                model.train()

        model.load_state_dict(init_state)
        meta_optimizer.zero_grad()

        meta_loss = torch.stack(loss_tasks).sum(0)
        meta_loss.backward()
        print('meta loss', meta_loss.item())

        # for name, param in model.named_parameters():
        #     if 'adapter' in name:
        #         print(name)
        #         print(param.grad)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        meta_optimizer.step()

        # writer.add_scalar(tag='meta_loss', scalar_value=meta_loss.item(), global_step=i)
        if i % validation_per_step == 0:
            avg_valid_loss = \
                loss_validation(model, validation_iterator, pad_token=vocab['trg'].stoi['<pad>'])
            bleu_score = bleu_validation(model, validation_test_iterator, vocab, validation_ref,
                                         config['Record']['path'] + '/output',
                                         detruecase_script,
                                         detokenize_script,
                                         config['Dataset']['target_language'])
            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score
                torch.save(model.state_dict(), best_model_path)

            writer.add_scalar(tag='meta_loss', scalar_value=meta_loss.item(), global_step=i)
            writer.add_scalar(tag='bleu_score', scalar_value=bleu_score, global_step=i)
            writer.add_scalar(tag='validation_loss', scalar_value=avg_valid_loss, global_step=i)

            model.train()
        # init_state = copy.deepcopy(model.state_dict())


if __name__ == "__main__":
    main()
