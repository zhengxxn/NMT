import yaml
import torch
import numpy as np
import sys
from util.tester.transformer_multi_adapter_tester import TransformerMultiAdapterTester


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

    tester = TransformerMultiAdapterTester(config, device, model_name='transformer_with_parallel_adapter')
    tester.decoding()


if __name__ == "__main__":
    main()
