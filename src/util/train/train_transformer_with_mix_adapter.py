import yaml
import torch

from util.convenient_funcs import create_path, set_random_seed
from util.data_loader.mt_data_loader import MTDataLoader
from util.model_builder import ModelBuilder

from util.trainer.adapter_trainer import Adapter_Trainer
from util.trainer.classifier_trainer import ClassifierTrainer
from util.trainer.mix_adapter_trainer import Mix_Adapter_Trainer
from util.trainer.kd_trainer import Kd_Adapter_Trainer

import sys
import os

global max_src_in_batch, max_tgt_in_batch


def main():
    config_file_path = sys.argv[1]

    print('read config')
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)
        create_path(config['Record']['training_record_path'])

    # ================================================================================== #
    # set the device
    set_random_seed(config['Train']['random_seed'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    print('load dataset ...')
    mt_data_loader = MTDataLoader(config)
    mt_data_loader.load_datasets(load_train=True, load_dev=True, load_test=False)
    mt_data_loader.build_vocab()
    mt_data_loader.build_iterators(device=device, build_train=True, build_dev=True, build_test=False)

    vocab = mt_data_loader.vocab
    train_iterators = mt_data_loader.train_iterators
    train_iterator_domain = config['Dataset']['train_dataset_domain']
    dev_iterators = mt_data_loader.dev_iterators
    dev_test_iterators = mt_data_loader.dev_test_iterators
    dev_iterator_domain = config['Dataset']['dev_dataset_domain']

    model_builder = ModelBuilder()
    model = model_builder.build_model(model_name='transformer_with_mix_adapter',
                                      model_config=config['Model'],
                                      vocab=vocab,
                                      device=device,
                                      load_pretrained=config['Train']['load_exist_model'],
                                      pretrain_path=config['Train']['model_load_path'])
    model.classify_domain_mask = model.classify_domain_mask.to(device)

    criterion = model_builder.build_criterion(criterion_config=config['Criterion'], vocab=vocab)
    validation_criterion = model_builder.build_criterion(criterion_config={
        'name': 'kl_divergence',
        'label_smoothing': 0,
    }, vocab=vocab)

    # training_domain = config['Train']['training_domain']
    training_stage = config['Train']['stage']

    if training_stage == 'classify':
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif training_stage == 'mixture_of_experts':
        for name, param in model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif training_stage == 'train_gate':
        for name, param in model.named_parameters():
            if 'inner_gate' in name and any(domain in name for domain in train_iterator_domain):
                param.requires_grad = True
            else:
                param.requires_grad = False

    else:
        # update specific domain adapter
        for name, param in model.named_parameters():
            if 'adapter' in name and any(domain in name for domain in train_iterator_domain):
                param.requires_grad = True
            else:
                param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = model_builder.build_optimizer(parameters=parameters,
                                              optimizer_config=config['Optimizer'],
                                              load_pretrained=config['Train']['load_optimizer'],
                                              pretrain_path=config['Train']['optimizer_path'])
    # make optimizer
    lr_scheduler = model_builder.build_lr_scheduler(optimizer=optimizer,
                                                    lr_scheduler_config=config['Optimizer']['lr_scheduler'],
                                                    load_pretrained=config['Train']['load_lr_scheduler'],
                                                    pretrain_path=config['Train']['lr_scheduler_path']
                                                    )

    os.system('cp ' + config_file_path + ' ' + config['Record']['training_record_path'] + '/model_config.txt')

    # parameters=filter(lambda p: p.requires_grad, model.parameters()))
    if training_stage == 'classify':
        trainer = ClassifierTrainer(
            model=model,
            criterion=criterion,
            vocab=vocab,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_iterators=train_iterators,
            train_iterators_domain_list=train_iterator_domain,
            validation_iterators=dev_iterators,
            validation_iterators_domain_list=dev_iterator_domain,
            domain_dict=config['Model']['domain_dict'],
            optimizer_config=config['Optimizer'],
            train_config=config['Train'],
            validation_config=config['Validation'],
            record_config=config['Record'],
            device=device,

        )
    elif training_stage == 'mix_adapter_translation' or training_stage == 'train_gate' or training_stage == 'mixture_of_experts':
        trainer = Mix_Adapter_Trainer(
            model=model,
            criterion=criterion,
            validation_criterion=validation_criterion,
            vocab=vocab,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_iterators=train_iterators,
            validation_iterators=dev_iterators,
            validation_test_iterators=dev_test_iterators,
            optimizer_config=config['Optimizer'],
            train_config=config['Train'],
            validation_config=config['Validation'],
            record_config=config['Record'],
            device=device,
            target_domain=config['Train']['target_domain'],
            used_domain_list=config['Train']['used_domain_list'],
            used_inner_gate=config['Train']['used_inner_gate'],
        )
    elif training_stage == 'kd':
        trainer = Kd_Adapter_Trainer(
            model=model,
            criterion=criterion,
            validation_criterion=validation_criterion,
            vocab=vocab,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_iterators=train_iterators,
            validation_iterators=dev_iterators,
            validation_test_iterators=dev_test_iterators,
            optimizer_config=config['Optimizer'],
            train_config=config['Train'],
            validation_config=config['Validation'],
            record_config=config['Record'],
            device=device,
            target_domain=config['Train']['target_domain'],
            ref_domain_dict=config['Train']['kd_ref_domain']
        )
    else:
        trainer = Adapter_Trainer(
            model=model,
            criterion=criterion,
            validation_criterion=validation_criterion,
            vocab=vocab,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_iterators=train_iterators,
            validation_iterators=dev_iterators,
            validation_test_iterators=dev_test_iterators,
            optimizer_config=config['Optimizer'],
            train_config=config['Train'],
            validation_config=config['Validation'],
            record_config=config['Record'],
            device=device,
            target_domain=config['Train']['target_domain'],
        )

    trainer.train()


if __name__ == "__main__":
    main()