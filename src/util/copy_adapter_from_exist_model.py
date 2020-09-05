import yaml

import torch
import numpy as np
from util.convenient_funcs import create_path, get_path_prefix
from util.data_loader.mt_data_loader import MTDataLoader
from util.model_builder import ModelBuilder
import sys


def main():
    torch.manual_seed(3333)
    np.random.seed(3333)

    config_file_path = sys.argv[1]

    print('read config')
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)

    # ================================================================================== #
    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set the data fields
    mt_data_loader = MTDataLoader(config)
    mt_data_loader.build_vocab()
    vocab = mt_data_loader.vocab

    model_builder = ModelBuilder()

    model = model_builder.build_model(model_name=config['copy_adapter']['model_name'],
                                      model_config=config['Model'],
                                      vocab=vocab,
                                      device=device,
                                      load_pretrained=False,
                                      pretrain_path=None)
    model_dict = model.state_dict()

    # load from trained model
    load_model_dict = torch.load(config['copy_adapter']['load_model_path'])
    model_dict.update(load_model_dict)

    # copy adapter parameters according to dict
    for copy_item in config['copy_adapter']['copy_dict']:
        src_adapter_domain = copy_item['src']
        trg_adapter_domain = copy_item['trg']

        for parameter_name in model_dict.keys():
            if trg_adapter_domain in parameter_name:
                src_adapter_parameter_name = parameter_name.replace(trg_adapter_domain, src_adapter_domain)
                # copy value
                model_dict[parameter_name] = model_dict[src_adapter_parameter_name]
                print(parameter_name, src_adapter_parameter_name)

    model.load_state_dict(model_dict)
    create_path(get_path_prefix(config['copy_adapter']['save_path']))
    torch.save(model.state_dict(), config['copy_adapter']['save_path'])


if __name__ == "__main__":
    main()
