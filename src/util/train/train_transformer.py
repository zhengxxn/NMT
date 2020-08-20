import yaml
import torch
from util.trainer.transformer_trainer import Trainer
from util.convenient_funcs import create_path, set_random_seed
from util.data_loader.mt_data_loader import MTDataLoader
from util.model_builder import ModelBuilder
import sys
import os
global max_src_in_batch, max_tgt_in_batch


def main():
    config_file_path = sys.argv[1]

    print('read config')
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)
        create_path(config['Record']['training_record_path'])

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
    mt_data_loader.build_iterators(device=device, build_train=True, build_dev=True, build_test=False)

    vocab = mt_data_loader.vocab
    train_iterators = mt_data_loader.train_iterators
    dev_iterators = mt_data_loader.dev_iterators
    dev_test_iterators = mt_data_loader.dev_test_iterators

    # make model
    model_builder = ModelBuilder()
    model = model_builder.build_model(model_name='transformer',
                                      model_config=config['Model'],
                                      vocab=vocab,
                                      device=device,
                                      load_pretrained=config['Train']['load_exist_model'],
                                      pretrain_path=config['Train']['model_load_path'])
    print('trained parameters: ')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    # make criterion
    criterion = model_builder.build_criterion(criterion_config=config['Criterion'], vocab=vocab)

    # make optimizer
    optimizer = model_builder.build_optimizer(parameters=model.parameters(),
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

    trainer = Trainer(
        model=model,
        criterion=criterion,
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
    )

    trainer.train()


if __name__ == "__main__":
    main()
