from util.lr_scheduler.noam_lr_scheduler import NoamLR
from util.lr_scheduler.inverse_sqrt_lr_scheduler import InverseSqrtLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_lr_scheduler(name, optimizer, config):

    if name == 'none':
        return None

    elif name == 'noam':
        return {'name': name,
                'base_on': 'step',
                'scheduler': NoamLR(optimizer,
                                    model_size=config['model_size'],
                                    factor=config['factor'],
                                    warmup_steps=config['warmup_steps'])}

    elif name == 'inverse_sqrt':
        return {'name': name,
                'base_on': 'step',
                'scheduler': InverseSqrtLR(optimizer,
                                           warmup_steps=config['warmup_steps'],
                                           init_lr=config['init_lr'],
                                           max_lr=config['max_lr'],
                                           min_lr=config['min_lr'])}

    elif name == 'reduce_lr_on_loss':
        return {'name': name,
                'base_on': 'metric',
                'scheduler': ReduceLROnPlateau(optimizer,
                                               mode='min',
                                               factor=config['factor'],
                                               patience=config['patience'],
                                               min_lr=config['min_lr'])}

    elif name == 'reduce_lr_on_bleu':
        return {'name': name,
                'base_on': 'metric',
                'scheduler': ReduceLROnPlateau(optimizer,
                                               mode='max',
                                               factor=config['factor'],
                                               patience=config['patience'],
                                               min_lr=config['min_lr'])}
