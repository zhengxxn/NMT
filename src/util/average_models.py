import yaml

import torch
import numpy as np
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

    model = model_builder.build_model(model_name=config['AverageModel']['model_name'],
                                      model_config=config['Model'],
                                      vocab=vocab,
                                      device=device,
                                      load_pretrained=False,
                                      pretrain_path=None)
    model_dict = model.state_dict()
    models = []
    for model_path in config['AverageModel']['load_path']:
        _model = model_builder.build_model(model_name=config['AverageModel']['model_name'],
                                           model_config=config['Model'],
                                           vocab=vocab,
                                           device=device,
                                           load_pretrained=False,
                                           pretrain_path=None)

        save_model_state_dict = torch.load(model_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        save_in_multi_gpu = config['AverageModel']['save_in_multi_gpu']
        if save_in_multi_gpu:
            for k, v in save_model_state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
            _model.load_state_dict(new_state_dict)
        else:
            save_model_state_dict = {k: v for k, v in save_model_state_dict.items() if k in model_dict}
            _model.load_state_dict(save_model_state_dict)

        models.append(_model)

    for ps in zip(*[m.parameters() for m in [model] + models]):
        ps[0].data.copy_(torch.stack([w.data for w in ps[1:]]).sum(0) / len(ps[1:]))

    torch.save(model.state_dict(), config['AverageModel']['save_path'])


if __name__ == "__main__":
    main()
