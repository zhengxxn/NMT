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

    model = model_builder.build_model(model_name=config['generate_adapter']['model_name'],
                                      model_config=config['Model'],
                                      vocab=vocab,
                                      device=device,
                                      load_pretrained=False,
                                      pretrain_path=None)
    model_dict = model.state_dict()

    # load from trained model
    load_model_dict = torch.load(config['generate_adapter']['load_model_path'])
    model_dict.update(load_model_dict)
    model.load_state_dict(model_dict)

    # from xx-generate to xx
    for generate_item in config['generate_adapter']['generate_dict']:
        src_adapter_name = generate_item['src']
        trg_adapter_name = generate_item['trg']

        for i in range(0, len(model.encoder.layers)):
            print(i)
            w1_weight, w1_bias, w2_weight, w2_bias = model.encoder.layers[i].adapters.adapter_layers[src_adapter_name].generate_param(model.encoder.layers[i].adapters.adapter_layers)
            model.encoder.layers[i].adapters.adapter_layers[trg_adapter_name].w_1.weight.data = w1_weight.transpose(0, 1)
            model.encoder.layers[i].adapters.adapter_layers[trg_adapter_name].w_1.bias.data = w1_bias
            model.encoder.layers[i].adapters.adapter_layers[trg_adapter_name].w_2.weight.data = w2_weight.transpose(0, 1)
            model.encoder.layers[i].adapters.adapter_layers[trg_adapter_name].w_2.bias.data = w2_bias

        for i in range(0, len(model.encoder.layers)):
            print(i)
            w1_weight, w1_bias, w2_weight, w2_bias = model.decoder.layers[i].adapters.adapter_layers[src_adapter_name].generate_param(model.decoder.layers[i].adapters.adapter_layers)
            model.decoder.layers[i].adapters.adapter_layers[trg_adapter_name].w_1.weight.data = w1_weight.transpose(0, 1)
            model.decoder.layers[i].adapters.adapter_layers[trg_adapter_name].w_1.bias.data = w1_bias
            model.decoder.layers[i].adapters.adapter_layers[trg_adapter_name].w_2.weight.data = w2_weight.transpose(0, 1)
            model.decoder.layers[i].adapters.adapter_layers[trg_adapter_name].w_2.bias.data = w2_bias

    create_path(get_path_prefix(config['generate_adapter']['save_path']))

    model_dict = model.state_dict()

    for generate_item in config['generate_adapter']['generate_dict']:
        src_adapter_name = generate_item['src']
        trg_adapter_name = generate_item['trg']
        for parameter_name in model_dict.keys():
            if trg_adapter_name in parameter_name and 'sublayer_connection' in parameter_name and 'generate' not in parameter_name:

                src_adapter_parameter_name = parameter_name.replace(trg_adapter_name, src_adapter_name)
                # copy value
                print(parameter_name, src_adapter_parameter_name)
                model_dict[parameter_name] = model_dict[src_adapter_parameter_name]

    model_dict = {k: v for k, v in model_dict.items() if 'generate' not in k}
    torch.save(model_dict, config['generate_adapter']['save_path'])


if __name__ == "__main__":
    main()
