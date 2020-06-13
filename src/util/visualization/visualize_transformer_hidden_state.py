import yaml
import torch
import numpy as np
import sys
from util.tester.transformer_tester import TransformerTester


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

    tester = TransformerTester(config, device, model_name='transformer')
    tester.visualize_hidden_state(config['visualize_file'])


if __name__ == "__main__":
    main()
