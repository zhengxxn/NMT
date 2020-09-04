import yaml

import torch
import numpy as np
from util.convenient_funcs import create_path, get_path_prefix
from util.data_loader.mt_data_loader import MTDataLoader
from util.model_builder import ModelBuilder
import sys


def check_inconsistent(model_dicts):
    model_dict_0 = model_dicts[0]
    for k, v in model_dict_0.items():
        for i in range(1, len(model_dicts)):
            model_dict_current = model_dicts[i]
            if k in model_dict_current.keys():
                v_c = model_dict_current[k]
                if not torch.equal(v, v_c):
                    print(k)


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

    model = model_builder.build_model(model_name=config['load_multiple_model']['model_name'],
                                      model_config=config['Model'],
                                      vocab=vocab,
                                      device=device,
                                      load_pretrained=False,
                                      pretrain_path=None)
    model_dict = model.state_dict()

    load_model_dicts = [torch.load(model_path) for model_path in config['load_multiple_model']['load_path']]
    check_inconsistent(load_model_dicts)

    for load_model_dict in load_model_dicts:
        model_dict.update(load_model_dict)

    model.load_state_dict(model_dict)
    create_path(get_path_prefix(config['load_multiple_model']['save_path']))
    torch.save(model.state_dict(), config['load_multiple_model']['save_path'])


if __name__ == "__main__":
    main()
