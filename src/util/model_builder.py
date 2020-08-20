from util.model_build.make_model.make_transformer import make_transformer
from util.model_build.make_model.make_transformer_with_adapter import make_transformer_with_adapter
from util.model_build.make_model.make_transformer_with_split_position import make_transformer_with_split_position
from util.model_build.make_model.make_transformer_with_houlsby_adapter import make_transformer_with_houlsby_adapter
from util.model_build.make_model.make_transformer_with_parallel_adapter import make_transformer_with_parallel_adapter
from util.model_build.make_model.make_transformer_with_stacked_adapter import make_transformer_with_stacked_adapter
from util.model_build.make_model.make_transformer_with_stacked_multi_head_adapter import make_transformer_with_stacked_multi_head_adapter
from util.model_build.make_model.make_transformer_with_diff_size_stacked_adapter import make_transformer_with_diff_size_stacked_adapter
from util.model_build.make_model.make_transformer_with_full_model import make_transformer_with_full_adapter
from util.model_build.make_model.make_transformer_with_synthesizer_adapter import make_transformer_with_synthesizer_adapter
from util.model_build.make_model.make_transformer_classifier_adapter import make_transformer_with_classifier_adapter

from util.lr_scheduler.get_lr_scheduler import get_lr_scheduler

from util.make_criterion import make_criterion
import torch
import torch.nn as nn


class ModelBuilder:

    def build_model(self, model_name, model_config, vocab, device, load_pretrained=False, pretrain_path=None):
        if model_name == 'transformer':
            model = make_transformer(model_config=model_config, vocab=vocab)
        elif model_name == 'transformer_with_adapter':
            model = make_transformer_with_adapter(model_config=model_config, vocab=vocab)
        elif model_name == 'transformer_with_split_position':
            model = make_transformer_with_split_position(model_config=model_config, vocab=vocab)
        elif model_name == 'transformer_with_houlsby_adapter':
            model = make_transformer_with_houlsby_adapter(model_config=model_config, vocab=vocab)
        elif model_name == 'transformer_with_parallel_adapter':
            model = make_transformer_with_parallel_adapter(model_config=model_config, vocab=vocab)
        elif model_name == 'transformer_with_stacked_adapter':
            model = make_transformer_with_stacked_adapter(model_config=model_config, vocab=vocab)
        elif model_name == 'transformer_with_stacked_multi_head_adapter':
            model = make_transformer_with_stacked_multi_head_adapter(model_config=model_config, vocab=vocab)
        elif model_name == 'transformer_with_diff_size_stacked_adapter':
            model = make_transformer_with_diff_size_stacked_adapter(model_config=model_config, vocab=vocab)
        elif model_name == 'transformer_with_full_adapter':
            model = make_transformer_with_full_adapter(model_config=model_config, vocab=vocab)
        elif model_name == 'transformer_with_synthesizer_adapter':
            model = make_transformer_with_synthesizer_adapter(model_config=model_config, vocab=vocab)
        elif model_name == 'transformer_with_classifier_adapter':
            model = make_transformer_with_classifier_adapter(model_config=model_config, vocab=vocab)
        else:
            model = None

        if load_pretrained and pretrain_path is not None:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(pretrain_path)

            if model_name == 'transformer_with_adapter' and 'replace' in model_config.keys():
                pretrained_dict = {k.replace(model_config['replace']['src'], model_config['replace']['trg']): v
                                   for k, v in pretrained_dict.items()}

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        return model

    def build_optimizer(self, parameters, optimizer_config, load_pretrained=False, pretrain_path=None):
        optimizer = torch.optim.Adam(parameters,
                                     betas=(0.9, optimizer_config['beta2']),
                                     lr=optimizer_config['lr_rate'])

        if load_pretrained and pretrain_path is not None:
            optimizer.load_state_dict(torch.load(pretrain_path))

        return optimizer

    def build_lr_scheduler(self, optimizer, lr_scheduler_config, load_pretrained=False, pretrain_path=None):
        lr_scheduler = get_lr_scheduler(
            optimizer=optimizer,
            name=lr_scheduler_config['name'],
            config=lr_scheduler_config)

        if load_pretrained and pretrain_path is not None:
            lr_scheduler.load_state_dict(torch.load(pretrain_path))

        return lr_scheduler

    def build_criterion(self, criterion_config, vocab):
        criterion = make_criterion(criterion_config, vocab)
        return criterion
