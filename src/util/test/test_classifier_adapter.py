import yaml
import torch
import numpy as np
import sys
from util.tester.classifier_tester import ClassifierTester
from util.tester.classifier_adapter_tester import ClassifierAdapterTester


def main():
    torch.manual_seed(3333)
    np.random.seed(3333)

    config_file_path = sys.argv[1]

    print('read config')
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config['Test']['stage'] == 'classify':
        tester = ClassifierTester(config, device, model_name='transformer_with_classifier_adapter')
        tester.test()
    else:
        tester = ClassifierAdapterTester(config, device, model_name='transformer_with_classifier_adapter')
        tester.decoding()


if __name__ == "__main__":
    main()
