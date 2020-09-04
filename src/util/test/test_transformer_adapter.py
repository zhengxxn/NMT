import yaml
import torch
import numpy as np
import sys
from util.tester.transformer_adapter_tester import TransformerAdapterTester


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

    model_name = config['Test']['model_name']
    tester = TransformerAdapterTester(config, device, model_name=model_name)
    tester.decoding()


if __name__ == "__main__":
    main()
