from module.criterion.label_smoothed_kl_divergence import LabelSmoothedKLDivergence
from module.criterion.label_smoothed_nll_loss import LabelSmoothedNLLLoss


def make_criterion(config, vocab):
    if config['name'] == 'kl_divergence':
        criterion = LabelSmoothedKLDivergence(size=len(vocab['trg']),
                                              padding_idx=vocab['trg'].stoi['<pad>'],
                                              smoothing=config['label_smoothing'],
                                              reduction=config['reduction'])
    elif config['name'] == 'nll':
        criterion = LabelSmoothedNLLLoss(size=len(vocab['trg']),
                                         padding_idx=vocab['trg'].stoi['<pad>'],
                                         smoothing=config['label_smoothing'],
                                         reduction=config['reduction'])
    else:
        criterion = LabelSmoothedKLDivergence(size=len(vocab['trg']),
                                              padding_idx=vocab['trg'].stoi['<pad>'],
                                              smoothing=config['label_smoothing'],
                                              reduction=config['reduction'])

    return criterion
